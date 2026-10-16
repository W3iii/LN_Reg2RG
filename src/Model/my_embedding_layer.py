import torch.nn as nn
import torch.nn.functional as F
import torch
from .helpers import PerceiverResampler
from .utils import get_visual_encoder
from einops import rearrange, repeat
from einops_exts import rearrange_many
import torchvision
from .vit_3d import ViT
from einops.layers.torch import Rearrange
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import random
from transformers import AutoTokenizer, AutoModel
from monai.networks.nets.swin_unetr import SwinTransformer
from .cross_attention import TwoWayTransformer

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]

from regions import REGIONS, MAX_LESION_PER_REGION


class LesionEncoder(nn.Module):
    """One token per nodule, from a native-resolution crop (docs/LESION_TOKENS.md).

    Deliberately not the ViT3D: that encoder's patch grid is tied to the lobe-level
    (256, 256, 64) resize, which is exactly what makes a ~5 mm nodule occupy 0.117
    of one patch (HANDOFF.md §6). The crop here is already ~64x64x32 at native
    (0.8, 0.8, 1.0) mm — no resize — so a handful of strided 3D convs plus global
    pooling is enough, and cheap: ~131k voxels per nodule against 12.6M for one
    resized region volume (docs/LESION_TOKENS.md §2).
    """

    def __init__(self, in_ch=1, dim=4096, width=32):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.GELU(),
            )

        self.net = nn.Sequential(
            block(in_ch, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 4),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(width * 4, dim)

    def forward(self, x):
        # x: (N, in_ch, 64, 64, 32) -> (N, dim)
        x = self.net(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _conv_block(c_in, c_out, stride):
    return nn.Sequential(nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1),
                         nn.GroupNorm(min(8, c_out), c_out), nn.GELU())


class PooledLesionEncoder(nn.Module):
    """Lesion encoder that keeps spatial resolution and pools around the centre.

    Why it exists (evaluation/train_lesion_encoder.py, val, 142 patients):
    LesionEncoder's global average pool over the 64x64x32 crop dilutes a nodule
    that fills <0.2% of it -- the B2 token probed at size rho 0.175, no better
    than a random encoder. Trained alone on size + TW_Lung_RADS, the same convs
    reach rho 0.69 with the global pool, 0.79 with `center` and 0.81 with `attn`
    (RADS kappa 0.55 / 0.67 / 0.69; mask-derived size alone gives 0.68).

    Two stride-2 then two stride-1 stages give a 16x16x8 map (3.2 mm/cell
    in-plane) instead of 4x4x2, then:
        center  fixed Gaussian weights centred on the nodule
        attn    learned spatial attention plus a learned centre-distance bias
    The local pool is concatenated with the global pool before the projection.

    Attribute names must stay as they are: train_lesion_encoder.py saves
    state_dicts from this exact class, and MyEmbedding loads them by key.
    """

    def __init__(self, mode, dim=4096, width=32, sigma=2.0):
        super().__init__()
        assert mode in ('center', 'attn'), mode
        self.mode = mode
        self.net = nn.Sequential(_conv_block(1, width, 2), _conv_block(width, width * 2, 2),
                                 _conv_block(width * 2, width * 4, 1), _conv_block(width * 4, width * 4, 1))
        c = width * 4
        g = torch.stack(torch.meshgrid(torch.arange(16.), torch.arange(16.), torch.arange(8.), indexing='ij'))
        centre = torch.tensor([7.5, 7.5, 3.5]).view(3, 1, 1, 1)
        scale = torch.tensor([1., 1., 2.]).view(3, 1, 1, 1)   # z cells are 2x as thick (64x64x32 crop)
        d2 = (((g - centre) * scale) ** 2).sum(0)
        self.register_buffer('d2', d2)
        w = torch.exp(-d2 / (2 * sigma ** 2))
        self.register_buffer('gauss', w / w.sum())
        if mode == 'attn':
            self.att = nn.Conv3d(c, 1, 1)
            self.dist_scale = nn.Parameter(torch.tensor(-0.1))
        self.fc = nn.Linear(2 * c, dim)

    def forward(self, x):
        # x: (N, 1, 64, 64, 32) -> (N, dim)
        f = self.net(x)                                     # (N, C, 16, 16, 8)
        glob = f.mean((2, 3, 4))
        if self.mode == 'center':
            loc = (f * self.gauss.to(f.dtype)).sum((2, 3, 4))
        else:
            logits = self.att(f)[:, 0] + self.dist_scale * self.d2.to(f.dtype)
            w = torch.softmax(logits.flatten(1).float(), 1).to(f.dtype).view_as(logits)
            loc = (f * w.unsqueeze(1)).sum((2, 3, 4))
        return self.fc(torch.cat([loc, glob], 1))


LESION_ARCHS = ('gap', 'center', 'attn')


def build_lesion_encoder(arch='gap', dim=4096):
    if arch == 'gap':
        return LesionEncoder(dim=dim)
    return PooledLesionEncoder(arch, dim=dim)


def load_pretrained_lesion_encoder(encoder, path, arch):
    """Load a best.pt written by evaluation/train_lesion_encoder.py."""
    # weights_only=False: our own file, and its metrics dict holds numpy scalars,
    # which torch>=2.6's default weights_only=True refuses to unpickle
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if blob.get('arch') != arch:
        raise ValueError(f'{path} was trained as arch={blob.get("arch")!r} but lesion_arch={arch!r}')
    encoder.load_state_dict(blob['encoder'], strict=True)
    m = blob.get('metrics', {})
    print(f'[lesion] loaded pretrained {arch} encoder from {path} '
          f'(val rho {m.get("rho", float("nan")):.3f}, kappa {m.get("kappa", float("nan")):.3f})')


class MyEmbedding(nn.Module):
    def __init__(self, pretrained_visual_encoder=None, pretrained_adapter=None, num_embeddings=32000, embedding_dim=4096, perceiver_num=32, vis_dim=768, patch_size=32, frame_patch_size=4, seg_channel=256,
                 lesion_arch='gap', lesion_encoder_ckpt=None, freeze_lesion_encoder=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.torch.randn((num_embeddings, embedding_dim)), requires_grad=True)  # NOTE: will be initialized using the weight from MedLLaMA
        self.image_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True)
        self.region_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True)
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.seg_channel = seg_channel

        self.vision_encoder = ViT(
            image_size=512,          # image size
            frames=512,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=frame_patch_size,      # frame patch size
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.mask_encoder = ViT(
            image_size=256,          # image size
            frames=64,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=16,      # frame patch size
            dim=255,
            depth=3,
            heads=8,
            mlp_dim=512,
            channels = 1,
            dropout=0.1,
            emb_dropout=0.1
        )

        # load pretrained vision encoder from RadFM
        if pretrained_visual_encoder is not None:
            vit3d_ckpt = torch.load(pretrained_visual_encoder, map_location='cpu')
            self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)
        # vit3d_ckpt = torch.load(
        #     '/data/chenzhixuan/checkpoints/LLM4CTRG/t3d_swin3d.pth', map_location='cpu')
        # self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)
        # print('load pretrained vision encoder from T3D')

        # frozen the vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.vis_dim = vis_dim

        self.perceiver = PerceiverResampler(
            dim=self.vis_dim, num_latents=perceiver_num)
        # load pretrained perceiver and fc from RadFM
        if pretrained_adapter is not None:
            state_dict = torch.load(pretrained_adapter, map_location='cpu')
            self.perceiver.load_state_dict(state_dict['perceiver'])
            # self.fc.load_state_dict(state_dict['fc'])
        
        # self.cross_attn = TwoWayTransformer(
        #     depth=3,
        #     embedding_dim=self.vis_dim,
        #     num_heads=8,
        #     mlp_dim=1024,
        # )
        
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        self.mask_fc = nn.Linear(255, self.embedding_dim)

        # docs/LESION_TOKENS.md; lesion_arch='gap' is the original B2 encoder.
        # B3a: a pooled encoder pretrained on size + RADS, optionally frozen so the
        # LM loss cannot wash out what it learned.
        self.lesion_arch = lesion_arch
        self.lesion_encoder = build_lesion_encoder(lesion_arch, dim=self.embedding_dim)
        if lesion_encoder_ckpt:
            load_pretrained_lesion_encoder(self.lesion_encoder, lesion_encoder_ckpt, lesion_arch)
        if freeze_lesion_encoder:
            if not lesion_encoder_ckpt:
                print('[lesion] WARNING: freezing a randomly initialised lesion encoder')
            for p in self.lesion_encoder.parameters():
                p.requires_grad = False
        print(f'[lesion] arch={lesion_arch} pretrained={bool(lesion_encoder_ckpt)} frozen={bool(freeze_lesion_encoder)}')

    def forward(self, vision_x, mask_x, text_input, region2areas, lesion_x=None, lesion_slots=None):
        # 获取输入张量vision_x的形状，并将各个维度的大小赋值给相应的变量
        # B: 批量大小，一次处理的样本数量
        # S: 序列长度，例如在处理文本时，这可能是句子的长度
        # C: 通道数，例如在处理图像时，这可能是颜色通道的数量（红、绿、蓝）
        # H: 高度，例如在处理图像时，这可能是图像的高度
        # W: 宽度，例如在处理图像时，这可能是图像的宽度
        # D: 深"度，例如在处理三维数据（如3D图像或点云）时，这可能是数据的深度

        B, S, C, H, W, D = next(iter(vision_x.values())).shape

        # NOTE: the dataset emits fp32 volumes while this module runs in bf16.
        # Training gets away with it because HF Trainer wraps the forward in an
        # autocast context; generate() does not, so the ViT's first LayerNorm
        # raises "expected scalar type Float but found BFloat16". Cast here so
        # both paths behave the same rather than relying on an ambient autocast.
        dtype = self.weight.dtype
        vision_x = {k: v.to(dtype) for k, v in vision_x.items()}
        mask_x = {k: v.to(dtype) for k, v in mask_x.items()}
        if lesion_x is not None:
            lesion_x = lesion_x.to(dtype)

        vision_temp = vision_x['image']
        vision_temp = rearrange(vision_temp, "b S c h w d-> (b S) c h w d")
        vision_temp, pos_embedding = self.vision_encoder(vision_temp)
        vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
        vision_temp = vision_temp.unsqueeze(2)
        vision_temp = self.perceiver(vision_temp)
        n = vision_temp.shape[2]
        vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
        vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n*S)
        image_embedding = vision_temp
        
        del vision_x['image']
    
        region_embeddings = vision_x
        mask_embeddings = mask_x
                
        for key in region_embeddings.keys():
            vision_temp = region_embeddings[key]
            vision_temp = rearrange(vision_temp, "b S c h w d-> (b S) c h w d")
            vision_temp, _ = self.vision_encoder(vision_temp)
            vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
            vision_temp = vision_temp.unsqueeze(2)
            vision_temp = self.perceiver(vision_temp)
            n = vision_temp.shape[2]
            vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
            vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n*S)
            
            region_embeddings[key] = vision_temp

            mask_embedding, _ = self.mask_encoder(mask_x[key])
            mask_embedding = torch.mean(mask_embedding, dim=1)
            mask_embeddings[key] = mask_embedding

        # combined_region_embedding = torch.cat([region_embeddings[key].unsqueeze(1) for key in region_embeddings.keys()], dim=1)
        
        # image_embedding, _ = self.cross_attn(
        #     image_embedding=image_embedding, 
        #     region_embedding=combined_region_embedding, 
        #     )
        
        image_embedding = self.fc(image_embedding)
        
        for key in region_embeddings.keys():
            region_embeddings[key] = self.fc(region_embeddings[key])
            mask_embeddings[key] = self.mask_fc(mask_embeddings[key])

        # fuse region embeddings and mask embeddings
        for key in region_embeddings.keys():
            region_embeddings[key] = torch.cat([region_embeddings[key], mask_embeddings[key].unsqueeze(1)], dim=1) 
        
        max_region = len(region_embeddings)
        # zero init the vision embedding
        # NOTE: dtype must follow the module, not torch's fp32 default. Otherwise
        # this buffer is fp32 while self.weight is bf16, and the torch.cat below
        # promotes the whole embedding table back to fp32 mid-graph.
        vision_region_embedding = torch.zeros(
            (B, 33*max_region, self.embedding_dim),
            device=text_input.device, dtype=self.weight.dtype) # NOTE: 2 means 1 region and 1 mask embeddings
        
        for i in range(B):
            for j in range(len(region2areas[i])):
                region = region2areas[i][j]
                vision_region_embedding[i, j*33:(j+1)*33, :] = region_embeddings[region][i, :, :]

        # NOTE: lesion tokens (docs/LESION_TOKENS.md). Fixed grid of
        # max_region * MAX_LESION_PER_REGION slots, same zero-fill-if-absent
        # convention as vision_region_embedding above. When the dataset is built
        # without nodule_metadata (the B0 baseline arm), lesion_x is empty and
        # this stays all zeros, making the forward pass identical to the
        # pre-lesion-token one.
        # lesion_x     : (B, L, C, 64, 64, 32), L = max lesions in this batch
        # lesion_slots : (B, L) int, flat region*MAX_LESION_PER_REGION+lesion
        #                index; -1 marks padding
        lesion_embedding = torch.zeros(
            (B, max_region * MAX_LESION_PER_REGION, self.embedding_dim),
            device=text_input.device, dtype=self.weight.dtype)
        if lesion_x is not None and lesion_slots is not None:
            for i in range(B):
                valid = lesion_slots[i] >= 0
                if valid.any():
                    tok = self.lesion_encoder(lesion_x[i][valid])
                    lesion_embedding[i, lesion_slots[i][valid]] = tok

        embedding_weight = torch.cat([self.weight, self.image_token_weight, self.region_token_weight], dim=0)  # num_embeddings+2, embedding_dim
        embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)  # B, num_embeddings+2, embedding_dim
        # B, num_embeddings+2+n, embedding_dim

        embedding_weight = torch.cat([embedding_weight, image_embedding, vision_region_embedding, lesion_embedding], dim=1)
        text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(
            vision_region_embedding.dtype).to(text_input.device)  # B, N, num_embeddings+2+n
        out_put = torch.matmul(text_input, embedding_weight)

        return out_put
