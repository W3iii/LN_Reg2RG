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
    ckpt_path: Optional[str] = field(   
        default="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/pytorch_model.bin")

@dataclass
class DataArguments:
    data_folder: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_preprocessed')
    mask_folder: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/dataset/valid_region_mask')
    report_file: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv')
    monai_cache_dir: Optional[str] = field(default='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/cache')
    # Must match what the checkpoint was trained with: a model trained with lesion
    # tokens and evaluated without them (or the reverse) gets a prompt whose
    # <lesion*> ids point at zero embeddings, and nothing raises.
    nodule_metadata: Optional[str] = field(default='')
    result_path: Optional[str] = field(default='/home/chenzhixuan/Workspace/Reg2RG/results/radgenome_combined_reports.csv')