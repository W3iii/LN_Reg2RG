import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import monai.transforms as transforms
from monai.data import PersistentDataset
import nibabel as nib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from functools import partial
import torch.nn.functional as F
import tqdm
import random
import pickle

CONDITIONS = [
    'Medical material',
    'Arterial wall calcification',
    'Cardiomegaly',
    'Pericardial effusion',
    'Coronary artery wall calcification',
    'Hiatal hernia',
    'Lymphadenopathy',
    'Emphysema',
    'Atelectasis',
    'Lung nodule',
    'Lung opacity',
    'Pulmonary fibrotic sequela',
    'Pleural effusion',
    'Mosaic attenuation pattern',
    'Peribronchial thickening',
    'Consolidation',
    'Bronchiectasis',
    'Interlobular septal thickening',
]

from regions import REGIONS, MAX_LESION_PER_REGION, lesion_token_names
from lesion_utils import instance_bboxes, nodules_by_region, crop_lesion

class RadGenomeDataset_Train(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir, max_region_size=5, max_img_size = 1, image_num = 32, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000, force_num_frames=True):
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}
        # 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        # docs/LESION_TOKENS.md §1: must be appended after every <region*> entry,
        # and this list must match Reg2RG.py and radgenome_dataset_test.py exactly
        # (rule 1 there) -- a mismatch doesn't raise, it just selects the wrong
        # embedding row.
        special_token["additional_special_tokens"].extend(
            lesion_token_names(max_region_size))

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2

        self.max_region_size = max_region_size
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder

        self.csv_file = csv_file
        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            # 将后缀为.nii.gz替换为.npz
            # accession = accession.replace('.nii.gz', '.npz')
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
        
        # 获取当前文件的绝对路径
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # NOTE: cache name is derived from the report CSV so that train/val/test
        # (and different datasets) never silently reuse each other's sample list.
        csv_tag = os.path.splitext(os.path.basename(self.csv_file))[0]
        cache_file = os.path.join(current_file_dir, f'samples_{csv_tag}.pkl')

        if os.path.exists(cache_file):
            samples = pickle.load(open(cache_file, 'rb'))
        else:
            for patient_folder in tqdm.tqdm(patient_folders):
                accession_folders = glob.glob(os.path.join(patient_folder, '*'))

                for accession_folder in accession_folders:
                    nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))
                    for nii_file in nii_files:
                        accession_number = nii_file.split("/")[-1]

                        if accession_number not in self.accession_to_sentences:
                            continue
                            
                        single_sample = {}
                        volume_name = accession_number.split(".")[0]
                        mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                        # add nii_file to single_sample
                        single_sample['image'] = nii_file

                        flag = False
                        for region in REGIONS:
                            # NOTE: only use the samples with the corresponding region report
                            if region in self.accession_to_sentences[accession_number]:
                                mask_file = os.path.join(mask_path, region + '.nii.gz')
                                region_report = self.accession_to_sentences[accession_number][region]
                                single_sample[region] = [mask_file, region_report]
                                flag = True
                        if not flag: # NOTE: if there is no corresponding region report, skip this sample
                            continue

                        samples.append(single_sample)
                        self.paths.append(nii_file)

            # save the samples to a file
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)

        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        img_data = nib.load(img_path).get_fdata()

        mask_img_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            
            mask_data = nib.load(mask_path).get_fdata()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if np.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img[np.newaxis, ...]

            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            mask_img_tensors[key] = tensor
            flag = True

        if not flag:
            print('No mask: ', img_path)

        # Lesion tokens (docs/LESION_TOKENS.md §5): read the nodule instance mask
        # if it exists next to the region masks. It doesn't exist for any patient
        # yet -- see HANDOFF_TO_LOCAL.md -- so this is a no-op today and starts
        # producing lesion crops the moment nodules.nii.gz lands on disk. Crops
        # come from `img_data` before CropForeground/Resize, i.e. the same native
        # (0.8, 0.8, 1.0) mm frame export_reg2rg.py crops the instance mask into.
        lesion_crops = {}
        if mask_paths:
            seg_dir = os.path.dirname(next(iter(mask_paths.values())))
            instance_path = os.path.join(seg_dir, 'nodules.nii.gz')
            if os.path.exists(instance_path):
                instance_mask = nib.load(instance_path).get_fdata().astype(np.int32)
                region_binary_masks = {key: masks[i] for i, key in enumerate(mask_keys)}
                assignment = nodules_by_region(instance_mask, region_binary_masks)
                boxes = instance_bboxes(instance_mask)

                per_region_nodules = {}
                for nid, region in assignment.items():
                    voxel_count = int(np.count_nonzero(instance_mask == nid))
                    per_region_nodules.setdefault(region, []).append((nid, voxel_count))

                lesion_hu_min, lesion_hu_max = -1000, 200
                for region, nodules in per_region_nodules.items():
                    # NOTE: ranked by voxel count as a size proxy.
                    # docs/LESION_TOKENS.md §2 specifies tw_lung_rads desc, then
                    # eq_diam_mm desc from nodule_metadata.csv -- switch to that
                    # once the metadata carries an id joinable to this instance
                    # mask's labels.
                    nodules.sort(key=lambda t: t[1], reverse=True)
                    crops = []
                    for nid, _ in nodules[:MAX_LESION_PER_REGION]:
                        crop = crop_lesion(img_data, boxes[nid])
                        crop = np.clip(crop, lesion_hu_min, lesion_hu_max)
                        crop = ((crop + 400) / 600).astype(np.float32)
                        crops.append(crop)
                    lesion_crops[region] = crops

        # process img_data
        img_data = img_data[np.newaxis, ...]
        masks_data = np.stack(masks, axis=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors, lesion_crops

    def text_add_image_tokens(self, text):

        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions, lesion_tokens_per_slot=None):
        lesion_tokens_per_slot = lesion_tokens_per_slot or {}
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>'
            lesion_tokens = lesion_tokens_per_slot.get(i, [])
            if lesion_tokens:
                region_text = region_text + " with lesions " + "".join(lesion_tokens)
            region_text = region_text + '. '
        text = region_text + text

        return text

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors, lesion_crops = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                region_reports.pop(key)

        region2area = {}
        shuffled_areas = list(region_reports.keys())
        random.shuffle(shuffled_areas)

        for i in range(len(shuffled_areas)):
            region2area[i] = shuffled_areas[i]

        # Lesion tokens: slot indexing must follow region2area's *post-shuffle*
        # slot i (docs/LESION_TOKENS.md §3) -- getting this wrong attaches a
        # lobe's lesions to a different lobe's region token and nothing errors.
        lesion_crop_list = []
        lesion_slot_list = []
        lesion_tokens_per_slot = {}
        for i in range(len(region2area)):
            crops = lesion_crops.get(region2area[i], [])
            if crops:
                toks = []
                for k, crop in enumerate(crops):
                    slot = i * MAX_LESION_PER_REGION + k
                    lesion_crop_list.append(crop)
                    lesion_slot_list.append(slot)
                    toks.append(f'<lesion{slot}>')
                lesion_tokens_per_slot[i] = toks

        if lesion_crop_list:
            lesion_x = torch.stack(
                [torch.from_numpy(c).unsqueeze(0) for c in lesion_crop_list], dim=0)
        else:
            lesion_x = torch.zeros((0, 1, 64, 64, 32), dtype=torch.float32)
        lesion_slots = torch.tensor(lesion_slot_list, dtype=torch.long)

        instruction = "Given the provided global and regional information from this CT scan, please generate a comprehensive medical report for each region. First, identify the anatomical area corresponding to each region, then provide detailed information about these anatomical structures and any abnormalities that are essential. You can refer to the global information as the context and take it as a supplement when generating each region report."

        # prompt = self.text_add_image_tokens(instruction)
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area), lesion_tokens_per_slot=lesion_tokens_per_slot)

        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for i in range(len(region2area)):
            area = region2area[i]
            region_report = region_reports[area]
            combined_report = combined_report + "The region " + str(i) + " is " + area + ": " + region_report + " "

        # make text input
        self.text_tokenizer.padding_side = "right"
            
        text_tensor = self.text_tokenizer(
            prompt + ' ' + combined_report, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
        )
        text_input = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        text_input[torch.sum(attention_mask)] = self.text_tokenizer.eos_token_id  # set the last token id to be eos id

        # make label
        prompt_tensor = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt"
        )
        prompt_length = torch.sum(prompt_tensor["attention_mask"][0])

        label = text_input.clone()
        label[label == self.text_tokenizer.pad_token_id] = -100
        # remove the additional special tokens from the label
        label[label >= self.voc_size] = -100
        label[:prompt_length] = -100  # only focus on answer part

        return {'lang_x': text_input,
                'vision_x': mask_img_tensors,
                'mask_x': mask_tensors,
                'region2area': region2area,
                'lesion_x': lesion_x,
                'lesion_slots': lesion_slots,
                'attention_mask': attention_mask,
                'label': label
                }