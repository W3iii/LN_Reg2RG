"""Per-nodule size / TW_Lung_RADS / type predictions from the lesion encoder, cross-fitted.

Why: B3a showed the lesion token carries size (probe rho ~0.8) but the LLM turns
it into digits badly (generated-size rho 0.21; every prediction stays in 3-8 mm,
mean 5.4 mm even for >=8 mm nodules). The next step is to stop asking the LLM to
read a number out of a 4096-d vector: let the encoder's own regression head
produce the number, and either

  (a) substitute it into the generated sentence after the fact (no retraining --
      an upper bound on a <SIZE>-placeholder decoder), or
  (b) write it into the prompt, "<lesion3> (~7 mm)", and let the LLM copy it.

(b) needs predictions on the *training* patients that are as noisy as the ones
the model will meet at inference. An encoder scored on its own training patients
is far too accurate, and an LLM trained on those prompts learns to trust them
blindly. So train-split predictions are out-of-fold: K encoders, each blind to
one fold of train patients. Val and test get the encoder fitted on all of train.

The type head (GGO / solid / part-solid / calcified; see train_lesion_encoder.py
for its masked-CE + MIL loss) is cross-fitted the same way, for the same reason:
"<lesion3> (~0.6cm, GGO)" in a training prompt must be as unreliable as it will be
at inference.

No model selection on val anywhere here: every fit runs a fixed number of epochs
and keeps the last weights (OneCycle ends at a low LR; the val-selected best
epochs were 34-38 of 40 anyway). Test labels are never read.

Outputs (<tag> = <arch>_<type_loss>; the B3e file lesion_pred_<arch>_seed<seed>.csv is never rewritten)
  outputs/lesion_pred_<tag>_seed<seed>.csv       one row per nodule, all splits
  outputs/lesion_pred_<tag>_seed<seed>_full.pt   encoder + heads fitted on all train
                                                 (MyEmbedding-loadable, like best.pt)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_lesion_encoder import (Net, Objective, TypeTargets, TYPES, add_loss_args, build_cache,   # noqa: E402
                                  checkpoint, concept_suffix, eval_loader, evaluate, fmt_cbm, fmt_type, make_optim, one_cycle,
                                  pretrain_g, train_loader)


def fit(a, X, logd, rads, tt, idx, device, tag):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dl = train_loader(X, logd, rads, idx, tt.bag, a.bs, a.workers, a.seed)
    model = Net(a.arch).to(device)
    if a.concept == 'frozen':
        pretrain_g(model, logd, rads, tt, idx, device, seed=a.seed)   # this fold's training nodules only
    obj = Objective(a, rads[idx], tt, idx, device, model)
    opt = make_optim(model, a.lr)
    sched = one_cycle(opt, a.lr, a.epochs, dl)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for x, ld, r, rows in dl:
            x, ld, r, rows = x.to(device), ld.to(device), r.to(device), rows.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                o = model(x)
            loss, _ = obj(o, ld, r, rows, ep)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    print(f'  [{tag}] fitted on {len(idx)} nodules, {a.epochs} epochs, {time.time()-t0:.0f}s', flush=True)
    return model


@torch.no_grad()
def predict(model, X, idx, device, bs=128):
    model.eval()
    S, P, T, C = [], [], [], []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(np.asarray(X[idx[s:s + bs]]).astype(np.float32))[:, None].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            sz, logits, tlog, _ = model(x)
        tp = F.softmax(tlog.float(), 1)
        S.append(sz.float().cpu()); P.append(F.softmax(logits.float(), 1).cpu()); T.append(tp.cpu())
        C.append(F.softmax(model.concept(sz, tp), 1).cpu())
    return np.exp(torch.cat(S).numpy()), torch.cat(P).numpy(), torch.cat(T).numpy(), torch.cat(C).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='attn', choices=['gap', 'center', 'attn'])
    ap.add_argument('--data_root', default='/root/notebooks/groups/BME/reg2rg_nifti')
    ap.add_argument('--cache', default=os.path.join(HERE, '..', 'outputs', 'lesion_crops_all.npy'),
                    help='crop cache incl. test (separate from the train/val-only cache)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    add_loss_args(ap)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    if not os.path.exists(a.cache):
        build_cache(a.data_root, os.path.join(a.data_root, 'nodule_metadata.csv'), a.cache,
                    splits=('train', 'val', 'test'))
    X = np.load(a.cache, mmap_mode='r')
    meta = pd.read_csv(a.cache + '.csv')
    logd = np.log(meta['eq_diam_mm'].to_numpy(np.float32))
    rads = meta['tw_lung_rads'].to_numpy(np.int64) - 1
    split = meta['split'].to_numpy()
    tr, va, te = (np.flatnonzero(split == s) for s in ('train', 'val', 'test'))
    pid = meta['pid'].to_numpy()
    tt = TypeTargets(meta, a.type_csv, a.data_root)
    if (tt.tset[te].sum() > 0) or (tt.bag[te] >= 0).any():
        raise ValueError(f'{a.type_csv} carries type labels for test nodules; it must not')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'arch={a.arch} type_loss={a.type_loss} device={device}  train={len(tr)} val={len(va)} test={len(te)}  folds={a.folds}')

    # type_loss always in the name: '--type_loss none' must not overwrite B3e's
    # outputs/lesion_pred_<arch>_seed<seed>.csv, which B3e/B3e_clean inference still reads
    tag = f'{a.arch}_{a.type_loss}' + concept_suffix(a)
    out = a.out or os.path.join(HERE, '..', 'outputs', f'lesion_pred_{tag}_seed{a.seed}.csv')
    pred_d = np.full(len(meta), np.nan)
    pred_p = np.full((len(meta), 4), np.nan)
    pred_t = np.full((len(meta), len(TYPES)), np.nan)
    pred_c = np.full((len(meta), 4), np.nan)
    source = np.array([''] * len(meta), dtype=object)

    # out-of-fold on train patients (a MIL bag is one lobe of one patient, so it never straddles folds)
    oof_metrics = []
    for k, (fit_i, ho_i) in enumerate(GroupKFold(n_splits=a.folds).split(tr, groups=pid[tr])):
        m = fit(a, X, logd, rads, tt, tr[fit_i], device, f'fold {k+1}/{a.folds}')
        d, p, t, c = predict(m, X, tr[ho_i], device)
        pred_d[tr[ho_i]], pred_p[tr[ho_i]], pred_t[tr[ho_i]], pred_c[tr[ho_i]], source[tr[ho_i]] = d, p, t, c, f'oof{k+1}'
        oof_metrics.append(evaluate(m, eval_loader(X, logd, rads, tr[ho_i], a.bs), device, tt, a.concept))
        del m; torch.cuda.empty_cache()

    # all of train -> val, test
    m = fit(a, X, logd, rads, tt, tr, device, 'full')
    for idx in (va, te):
        if len(idx):
            d, p, t, c = predict(m, X, idx, device)
            pred_d[idx], pred_p[idx], pred_t[idx], pred_c[idx], source[idx] = d, p, t, c, 'full'
    vm = evaluate(m, eval_loader(X, logd, rads, va, a.bs), device, tt, a.concept)
    torch.save(checkpoint(m, a.arch, vm, type_loss=a.type_loss, concept=a.concept,
                          note='fitted on all train, last epoch, no val selection'),
               out.replace('.csv', '_full.pt'))

    res = meta.copy()
    res['pred_diam_mm'] = pred_d.round(2)
    res['pred_rads'] = np.nanargmax(pred_p, 1) + 1
    for c in range(4):
        res[f'p_rads{c+1}'] = pred_p[:, c].round(4)
    if a.type_loss != 'none':
        res['pred_type'] = [TYPES[i] for i in np.nanargmax(pred_t, 1)]
        res['pred_type_conf'] = np.nanmax(pred_t, 1).round(4)
        for c, n in enumerate(TYPES):
            res[f'p_{n.replace("-", "_")}'] = pred_t[:, c].round(4)
        # report-derived label, for scoring only (empty for test: never parsed)
        res['type_label'] = [TYPES[h] if h >= 0 else '' for h in tt.hard]
    if a.concept != 'none':
        # the concept-path grade: from predicted size + type alone (train_lesion_encoder.Net)
        res['pred_rads_cbm'] = np.nanargmax(pred_c, 1) + 1
        for c in range(4):
            res[f'p_rads_cbm{c+1}'] = pred_c[:, c].round(4)
    res['pred_source'] = source
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    res.to_csv(out, index=False)

    f = lambda m: (f'rho {m["rho"]:.3f}  MAE {m["MAE"]:.2f}  MAE(>=6) {m["MAE_ge6"]:.2f}  kappa {m["kappa"]:.3f}  '
                   f'AUC>=3 {m["auc_ge3"]:.3f}' + (fmt_type(m) if a.type_loss != 'none' else '') + fmt_cbm(m))
    print()
    for k, mm in enumerate(oof_metrics):
        print(f'OOF fold {k+1}: {f(mm)}')
    print(f'VAL (full fit, last epoch): {f(vm)}')
    print('The OOF folds should look like VAL: if they are much better, the train-split '
          'predictions are leaking and must not go into training prompts.')
    print(f'wrote {out}  ({len(res)} nodules; test rows are predictions only, labels not scored)')


if __name__ == '__main__':
    main()
