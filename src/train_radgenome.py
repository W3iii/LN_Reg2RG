import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence
import transformers
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer
from dataclasses import dataclass, field
from Model.Reg2RG import Reg2RG
from Dataset.radgenome_dataset_train import RadGenomeDataset_Train
from args.train_radgenome.jhcpu7 import ModelArguments, DataArguments, TrainingArguments
import numpy as np
import torch              
import random

from regions import REGIONS

@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print(instances) 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words
        lang_xs, vision_xs, mask_xs, region2areas, lesion_xs, lesion_slots_list, attention_masks, labels = tuple([instance[key] for instance in instances] for key in ('lang_x', 'vision_x', 'mask_x', 'region2area', 'lesion_x', 'lesion_slots', 'attention_mask', 'label'))

        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim = 0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks], dim = 0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim = 0)

        # docs/LESION_TOKENS.md §4: pad lesion_x / lesion_slots to this batch's max
        # lesion count L. -1 slots are padding (MyEmbedding.forward's `valid` mask
        # skips them), matching how absent regions are already zero-filled above.
        max_lesions = max((lx.shape[0] for lx in lesion_xs), default=0)
        if max_lesions == 0:
            lesion_x = torch.zeros((len(instances), 0, 1, 64, 64, 32), dtype=torch.float32)
            lesion_slots = torch.zeros((len(instances), 0), dtype=torch.long)
        else:
            lesion_x = torch.stack([
                F.pad(lx, (0, 0, 0, 0, 0, 0, 0, 0, 0, max_lesions - lx.shape[0]))
                for lx in lesion_xs
            ], dim=0)
            lesion_slots = torch.stack([
                F.pad(ls, (0, max_lesions - ls.shape[0]), value=-1)
                for ls in lesion_slots_list
            ], dim=0)

        vision_temp = {area: [] for area in REGIONS}
        mask_temp = {area: [] for area in REGIONS}
        # get the shape of the vision tensor
        # NOTE: carry the dtype too — the padding below must match the real
        # tensors, or absent regions arrive at a different precision than present
        # ones within the same batch.
        vision_ref = next(iter(vision_xs[0].values()))
        mask_ref = next(iter(mask_xs[0].values()))
        vision_shape, vision_dtype = vision_ref.shape, vision_ref.dtype
        mask_shape, mask_dtype = mask_ref.shape, mask_ref.dtype

        useless_regions = []
        
        for area in REGIONS:
            flag = False
            for i in range(len(vision_xs)):
                if area in vision_xs[i]:
                    vision_temp[area].append(vision_xs[i][area])
                    mask_temp[area].append(mask_xs[i][area])
                    flag = True
                else:
                    vision_temp[area].append(torch.zeros(vision_shape, dtype=vision_dtype))
                    mask_temp[area].append(torch.zeros(mask_shape, dtype=mask_dtype))
            if not flag:
                useless_regions.append(area)

        images = torch.cat([vision['image'].unsqueeze(0) for vision in vision_xs], dim = 0)
        
        # drop the useless regions from vision_temp
        for area in useless_regions:
            vision_temp.pop(area)
            mask_temp.pop(area)
        useful_regions = list(vision_temp.keys())
    
        vision_xs = {area: torch.cat([_.unsqueeze(0) for _ in vision_temp[area]], dim = 0) for area in useful_regions}
        # add image 
        vision_xs['image'] = images

        mask_xs = {area: torch.cat([_.unsqueeze(0) for _ in mask_temp[area]], dim = 0) for area in useful_regions}
        
        # print(vision_xs.shape,vision_xs.dtype)
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            mask_x=mask_xs,
            region2area = region2areas,
            lesion_x = lesion_x,
            lesion_slots = lesion_slots,
            attention_mask=attention_masks,
            labels = labels,
        )
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
                 
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # NOTE: seed comes from --seed (HF TrainingArguments already defines it, default
    # 42) rather than being hardcoded, so a run can be repeated under several seeds
    # to establish the run-to-run noise floor. Two architecturally identical runs of
    # this config gave micro-F1 0.235 and 0.439 (HANDOFF.md §6), i.e. the variance
    # is larger than any effect worth claiming -- no single run means anything on its
    # own. Give each seed its own output_dir; HF Trainer's checkpoint rotation sorts
    # by mtime, so sharing one directory makes later runs delete earlier ones.
    set_seed(training_args.seed)
    print(f'[train] seed={training_args.seed} output_dir={training_args.output_dir}')

    print("Setup Data")
    Train_dataset = RadGenomeDataset_Train(
        text_tokenizer=model_args.tokenizer_path,
        data_folder=data_args.data_folder,
        mask_folder=data_args.mask_folder,
        csv_file=data_args.report_file,
        cache_dir=data_args.monai_cache_dir,
    )

    print("Setup Model")
    model = Reg2RG(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
        pretrained_visual_encoder=model_args.pretrained_visual_encoder,
        pretrained_adapter=model_args.pretrained_adapter,
    )
    
    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      args = training_args,
                      data_collator=DataCollator(),
                      )

    trainer.train()
    trainer.save_state()
      
if __name__ == "__main__":
    main()