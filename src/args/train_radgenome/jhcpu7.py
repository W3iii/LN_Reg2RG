import transformers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf")
    tokenizer_path: str = field(default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf",
                                metadata={"help": "Path to the tokenizer data."})
    pretrained_visual_encoder: Optional[str] = field(
        default="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadFM_vit3d.pth")
    pretrained_adapter: Optional[str] = field(
        default="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadFM_perceiver_fc.pth")
    # Lesion encoder (src/Model/my_embedding_layer.py). gap = original B2 encoder.
    lesion_arch: str = field(default='gap', metadata={'help': 'gap | center | attn'})
    lesion_encoder_ckpt: Optional[str] = field(default='', metadata={'help': 'best.pt from evaluation/train_lesion_encoder.py'})
    freeze_lesion_encoder: bool = field(default=False)

@dataclass
class DataArguments:
    data_folder: Optional[str] = field(default='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_preprocessed')
    mask_folder: Optional[str] = field(default='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_region_mask')
    report_file: Optional[str] = field(default='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv')
    monai_cache_dir: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/cache')
    # Empty = lesion tokens off (the B0 baseline). Point it at nodule_metadata.csv
    # to run the B2 lesion arm; see docs/LESION_TOKENS.md.
    nodule_metadata: Optional[str] = field(default='')
    # B3c: predicted sizes in the prompt ('<lesionK> (~X mm)') and label-free
    # lesion ordering. Empty csv + 'truth' = B2/B3a behaviour.
    lesion_pred_csv: Optional[str] = field(default='')
    lesion_order: str = field(default='truth', metadata={'help': 'truth | pred'})
    # B3d: 'cm' writes '(~0.7cm)' like the reports; 'mm' = B3c.
    lesion_size_unit: str = field(default='mm', metadata={'help': 'mm | cm'})
    # B3e: sizes in the report targets become '[SIZE]' (src/size_placeholder.py);
    # at inference they are filled from lesion_pred_csv. Needs lesion_pred_csv.
    size_placeholder: bool = field(default=False)
    # B3f: also write the encoder's predicted nodule type into the prompt,
    # '<lesionK> (~0.6cm, GGO)'. Needs a lesion_pred_csv with a pred_type column.
    lesion_type: bool = field(default=False)
    # B3g: write the type only when the encoder's confidence reaches this (0 = always, B3f)
    lesion_type_min_conf: float = field(default=0.0)
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(
        default="/jhcnas5/chenzhixuan/checkpoints/Reg2RG/outputs")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    pin_memory: bool = field(default=True)