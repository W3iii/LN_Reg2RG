"""Cache per-region visual features at three points along the encoder pathway.

This is the measurement instrument for the "does local (lesion-scale) information
survive region pooling" question. It deliberately stops before the LLM: the
region representation is fully formed inside MyEmbedding, so nothing about this
question requires loading LLaMA, generating text, or parsing reports. That makes
the experiment cheap and removes every noise source downstream of the encoder
(sampling, decoding, regex parsing).

Three tap points per region, chosen to separate the candidate bottlenecks:

    pre_perceiver   ViT3D patch tokens        (N_patches, 768)
    post_perceiver  PerceiverResampler out    (32, 768)     <- fixed-size bottleneck
    post_fc         what the LLM finally sees (32, 4096)

`pre` vs `post` is the decisive comparison. The lobe volume is resized to
(256, 256, 64) and patched at (32, 32, 4), giving 8*8*16 = 1024 patch tokens,
which the perceiver then compresses to a *fixed* 32 latents regardless of input
size. If lesion-scale information is measurable at `pre_perceiver` but not at
`post_perceiver`, the resampler is the bottleneck -- and changing the resize
schedule (the B1 ablation in docs/LESION_TOKENS.md) cannot help, because 32
latents in, 32 latents out.

Each tap is stored mean- and max-pooled over its token axis. Both matter: a
nodule occupying well under one patch is diluted ~1000x by mean pooling but can
survive max pooling, so reporting only one of them would beg the question.

Outputs an .npz consumed by probe_local_info.py.
"""
import argparse
import os
import sys

import numpy as np
import torch
import nibabel as nib
from einops import rearrange

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from Model.my_embedding_layer import MyEmbedding          # noqa: E402
from Dataset.radgenome_dataset_test import RadGenomeDataset_Test  # noqa: E402
from regions import REGIONS                                # noqa: E402


def build_encoder(ckpt_path, device, dtype):
    """MyEmbedding alone, with the encoder weights from a training checkpoint.

    `weight` (the tied word-embedding table) is 32000x4096 and irrelevant here --
    it is only used for the one-hot token lookup that happens after the region
    features are built. Skipping it keeps this to the vision path.
    """
    model = MyEmbedding()
    ckpt = torch.load(ckpt_path, map_location='cpu')

    state = {}
    for k, v in ckpt.items():
        if not k.startswith('embedding_layer.'):
            continue
        sub = k[len('embedding_layer.'):]
        if sub == 'weight':
            continue
        state[sub] = v

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != 'weight']
    if missing:
        raise RuntimeError(f'checkpoint is missing encoder weights: {missing[:8]}')
    if unexpected:
        raise RuntimeError(f'unexpected keys for MyEmbedding: {unexpected[:8]}')

    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def region_taps(model, region_volume, mask_volume, device, dtype):
    """One region -> pooled features at each tap point.

    Mirrors the region loop in MyEmbedding.forward exactly; if that changes, this
    must change with it.
    """
    v = region_volume.unsqueeze(0).to(device=device, dtype=dtype)  # (B=1,S=1,C,H,W,D)
    B, S = v.shape[0], v.shape[1]

    x = rearrange(v, "b S c h w d-> (b S) c h w d")
    patches, _ = model.vision_encoder(x)                    # (B*S, N_patch, 768)
    pre = patches

    x = rearrange(patches, "(b s) v d -> b s v d", b=B, s=S).unsqueeze(2)
    latents = model.perceiver(x)                            # (B, S, 32, 768)
    post = rearrange(latents, "b s n d -> (b s n) d").unsqueeze(0)

    fc_out = model.fc(post)                                 # (1, 32, 4096)

    m = mask_volume.unsqueeze(0).to(device=device, dtype=dtype)
    mask_emb, _ = model.mask_encoder(m.squeeze(0))
    mask_emb = model.mask_fc(mask_emb.mean(dim=1))          # (1, 4096)

    def pooled(t):
        t = t.reshape(-1, t.shape[-1]).float()
        return torch.cat([t.mean(0), t.amax(0)]).cpu().numpy().astype(np.float16)

    return {
        'pre_perceiver': pooled(pre),
        'post_perceiver': pooled(post),
        'post_fc': pooled(fc_out),
        'mask_token': mask_emb.reshape(-1).float().cpu().numpy().astype(np.float16),
        'n_patches': int(pre.shape[-2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_root', required=True,
                    help='directory holding images/, masks/, region_report_<split>.csv')
    ap.add_argument('--tokenizer_path', required=True,
                    help='only used to satisfy the dataset constructor')
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--cache_dir', default=None,
                    help='MONAI cache dir. Defaults inside the repo so this never '
                         'writes to the shared clinical data volume, which we only read.')
    a = ap.parse_args()

    cache_dir = a.cache_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'monai_cache')
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float32
    print(f'device={device} dtype={dtype}')

    model = build_encoder(a.ckpt, device, dtype)
    print('encoder loaded')

    ds = RadGenomeDataset_Test(
        text_tokenizer=a.tokenizer_path,
        data_folder=os.path.join(a.data_root, 'images'),
        mask_folder=os.path.join(a.data_root, 'masks'),
        csv_file=os.path.join(a.data_root, f'region_report_{a.split}.csv'),
        cache_dir=cache_dir,
        inferenced_id=[],
    )
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    print(f'{n} samples')

    rows = []
    for i in range(n):
        sample = ds[i]
        acc = sample['acc_num']
        vision_x, mask_x = sample['vision_x'], sample['mask_x']

        # raw (pre-resize) lobe voxel count -- the trivial control feature.
        # mask_x is post-resize and therefore the same size for every lobe, so it
        # cannot carry absolute volume; read it off the mask file instead.
        raw_counts = {}
        for key in ds.data[i]:
            if key in ('image', 'accnum'):
                continue
            mask_file = ds.data[i][key][0]
            if os.path.exists(mask_file):
                raw_counts[key] = float(np.asarray(nib.load(mask_file, mmap=True).dataobj).sum())

        for region in REGIONS:
            if region not in vision_x:
                continue
            taps = region_taps(model, vision_x[region], mask_x[region], device, dtype)
            rows.append({
                'acc_num': acc,
                'region': region,
                'lobe_voxels': raw_counts.get(region, np.nan),
                **taps,
            })

        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{n}', flush=True)

    out = {
        'acc_num': np.array([r['acc_num'] for r in rows]),
        'region': np.array([r['region'] for r in rows]),
        'lobe_voxels': np.array([r['lobe_voxels'] for r in rows], dtype=np.float64),
        'n_patches': np.array([r['n_patches'] for r in rows], dtype=np.int32),
    }
    for tap in ('pre_perceiver', 'post_perceiver', 'post_fc', 'mask_token'):
        out[tap] = np.stack([r[tap] for r in rows])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f'wrote {a.out}: {len(rows)} region rows')
    for tap in ('pre_perceiver', 'post_perceiver', 'post_fc'):
        print(f'  {tap}: {out[tap].shape}')


if __name__ == '__main__':
    main()
