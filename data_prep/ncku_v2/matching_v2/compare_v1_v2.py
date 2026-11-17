import os, re, io
import pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LOB = r'\b(RUL|RML|RLL|LUL|LLL)\b|lingul|right upper lobe|right middle lobe|right lower lobe|left upper lobe|left lower lobe'
def nlobes(t):
    t = t or ''
    s = set()
    for m in re.finditer(LOB, t, re.I):
        g = m.group(0).lower()
        s.add({'lingul': 'lul', 'right upper lobe': 'rul', 'right middle lobe': 'rml', 'right lower lobe': 'rll',
               'left upper lobe': 'lul', 'left lower lobe': 'lll'}.get(g, g))
    return len(s)
def sz(t):
    return [float(m.group(1)) * (10 if m.group(2).lower() == 'cm' else 1)
            for m in re.finditer(r'(\d+(?:\.\d+)?)\s*-?\s*(mm|cm)\b', t or '', re.I) if 0.5 <= float(m.group(1)) * (10 if m.group(2).lower() == 'cm' else 1) <= 120]
def summ(df, txt, key, npat, lobe_agree=None):
    df = df.copy(); df[txt] = df[txt].fillna(''); m = df[df[txt] != '']
    one = m[txt].apply(lambda t: len(sz(t)) == 1)
    d = (m[one].eq_diam_mm - m[one][txt].apply(lambda t: sz(t)[0])).abs()
    out = {'配到句子的 nodule': f"{len(m):,} / {len(df):,} ({100*len(m)/len(df):.1f}%)",
           '至少 1 顆配到的病患': f"{m[key].nunique():,} / {npat:,}",
           '描述混入其他肺葉 (文字提到 ≥2 肺葉)': f"{(m[txt].apply(nlobes) >= 2).sum():,} ({100*(m[txt].apply(nlobes) >= 2).mean():.1f}%)",
           '高信心 (score ≥ 5.5)': f"{100*(m.match_score >= 5.5).mean():.1f}%",
           '單一尺寸描述: |mask 徑 − 報告尺寸| 中位數': f"{d.median():.2f} mm (n={len(d):,})"}
    if lobe_agree is not None:
        la = m[lobe_agree].dropna()
        out['肺葉一致 (分割 vs 句子)'] = f"{100*la.astype(bool).mean():.1f}% (n={len(la):,})"
    return out
# v1
me1 = pd.read_csv(os.path.join(ROOT, 'reg2rg_dataset', 'nodule_sentence.csv'))
ld1 = pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodule_sentence.csv'))
ld1 = ld1[ld1.note.fillna('') != 'zh_cad_template']
me2 = pd.read_csv(os.path.join(HERE, 'me_nodule_sentence_v2.csv'))
ld2 = pd.read_csv(os.path.join(HERE, 'ldct_nodule_sentence_v2.csv'))
T = pd.DataFrame({'ME v1': summ(me1, 'matched_sentence', 'pid', 1400, 'lobe_agree'),
                  'ME v2': summ(me2, 'per_nodule_text', 'key', 1400, 'lobe_agree'),
                  'LDCT v1': summ(ld1, 'matched_sentence', 'folder', 568),
                  'LDCT v2': summ(ld2, 'per_nodule_text', 'key', ld2.key.nunique(), 'lobe_agree' if ld2.lobe_agree.notna().any() else None)})
lines = ['v1 vs v2 配對結果', T.to_string(), '']
for name, df in (('ME', me2), ('LDCT', ld2)):
    m = df[df.per_nodule_text.fillna('') != '']
    lines.append(f'{name} v2 配對來源: ' + str(m.kind.value_counts().to_dict()))
    U = pd.read_csv(os.path.join(HERE, f'{name.lower()}_report_lesion_without_mask.csv'))
    lines.append(f'{name} 報告有描述但沒有對應 mask 的病灶片段: {len(U):,} 個, 涉及 {U.key.nunique():,} 位病患')
lines.append('')
for name, df in (('ME', me2), ('LDCT', ld2)):
    if df.lobe.fillna('').eq('').all(): lines.append(f'{name}: 尚無肺葉 mask, 略過獨立肺葉驗證'); continue
    one = df.groupby('key').filter(lambda g: len(g) == 1)
    one = one[(one.per_nodule_text.fillna('') != '') & one.piece_lobe.fillna('').str.fullmatch(r'RUL|RML|RLL|LUL|LLL') & (one.lobe.fillna('') != '')]
    lines.append(f'{name} 獨立肺葉驗證 (單顆 nodule + 句子只講 1 個肺葉, n={len(one)}): 左右一致 {100*(one.lobe.str[0]==one.piece_lobe.str[0]).mean():.1f}%  肺葉一致 {100*(one.lobe==one.piece_lobe).mean():.1f}%')
txt = '\n'.join(lines); print(txt)
io.open(os.path.join(HERE, 'compare_v1_v2.txt'), 'w', encoding='utf-8').write(txt + '\n')
