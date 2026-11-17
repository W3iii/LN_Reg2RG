"""v2 配對結果 -> Excel (ME / LDCT 各一份, v1 檔案不動)"""
import os, re
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
L5 = r'RUL|RML|RLL|LUL|LLL'
LOSS = {'A': 'target', 'B': 'target_low_weight', 'C': 'ignore', 'X': 'ignore', 'none': 'no_description'}
NOD = r'nodul|GGN|GGO|ground[\s-]?glass|opacit|mass\b|lesion|densit|calcif|tumou?r|granuloma|LN\b|lymph'


def tier(r):
    if not isinstance(r.per_nodule_text, str) or not r.per_nodule_text: return 'none'
    if not re.search(NOD, r.per_nodule_text, re.I): return 'X'          # 配到的是肺塌陷/支氣管擴張等非結節發現
    if r.kind != 'shared': return 'A'
    return 'B' if re.fullmatch(L5, r.piece_lobe or '') else 'C'


def prep(v2):
    v2 = v2.copy(); v2['piece_lobe'] = v2.piece_lobe.fillna('')
    v2['match_tier'] = v2.apply(tier, axis=1)
    v2['nodule_loss'] = v2.match_tier.map(LOSS)
    v2.loc[v2.match_tier == 'none', 'per_nodule_text'] = 'No specific description.'
    return v2


README = [
    ('match_tier', 'A = 具體描述 (一句對一顆, 含拆句後的單一肺葉/尺寸片段); B = 指定單一肺葉的複數描述 ("RLL GGOs"); '
                   'C = 雙側/全肺的整體描述 ("nodules in bilateral lungs"); X = 配到的句子是非結節發現 (atelectasis / bronchiectasis / infiltration…), 不確定是否同一病灶; '
                   'none = 報告沒提到; excluded = 不參與配對'),
    ('nodule_loss', '建議的逐顆訓練用法: target (A) / target_low_weight (B) / ignore (C, 不算 loss, 句子留在整份報告) / '
                    'no_description (none, 目標為 "No specific description.") / ignore 也用於 X / 空白 (excluded)'),
    ('kind', 'single = 原句直接配對; lobe_split = 多肺葉句依肺葉拆開; size_split = 同肺葉多尺寸依尺寸拆開; '
             'count_copy = "Two ..." 依數量複製; shared = 複數通用句給多顆'),
    ('per_nodule_text', '給這顆 nodule 的描述 (拆句時已改寫成單數 / 單一肺葉 / 單一尺寸; Findings 與 Impression 講同一病灶時已合併)'),
    ('matched_sentence', '報告原句 (未拆, 僅做過清理: 去樣板、病史、前次比較、Srs/Img 影像編號)'),
    ('per_nodule_lobe / per_nodule_size_mm', '句子片段指定的肺葉 / 尺寸 (mm); 可直接換成 [lobe_k] / [size_k] placeholder'),
    ('lobe_seg', 'nodule 中心所在肺葉 (TotalSegmentator 五肺葉 mask)'),
    ('eq_diam_mm', 'mask 等效球徑 2*(3V/4π)^(1/3), V = voxel 數 × 0.8×0.8×1.0 mm³'),
    ('match_score', '側別 3 (+0.5 單側) + 肺葉 (+3 同葉 / −1 不同葉) + 大小 (≤2mm +2 / ≤4mm +1 / >8mm −0.5); ≥5.5 = 高信心'),
    ('report_lesion_without_mask (工作表)', '報告有描述、但該病患沒有對應 mask 的病灶片段 -> 可能漏標; 訓練 lesion-grounded 生成時建議從目標拿掉或用分割模型補 mask'),
]


def write_book(path, sheets, wrap_cols, widths):
    """單次以 xlsxwriter 寫出 (含格式), 不再用 openpyxl 讀回重存 —
    重存會產生 inlineStr + 空的 t="n" 儲存格, Excel 判定為損壞。"""
    with pd.ExcelWriter(path, engine='xlsxwriter',
                        engine_kwargs={'options': {'strings_to_formulas': False, 'strings_to_urls': False}}) as w:
        wb = w.book
        hdr = wb.add_format({'bold': True, 'bg_color': '#D0CECE', 'border': 1, 'text_wrap': True, 'valign': 'vcenter'})
        wrap = wb.add_format({'text_wrap': True, 'valign': 'top'})
        for name, df in sheets:
            df = df.copy()
            for c in df.columns:                          # NaN -> 空白, 數字欄保持數字
                if df[c].dtype == object: df[c] = df[c].where(df[c].notna(), None)
            df.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            for j, h in enumerate(df.columns):
                ws.write(0, j, h, hdr)
                ws.set_column(j, j, widths.get(h, 12), wrap if h in wrap_cols else None)
            ws.freeze_panes(1, 0)


WRAP = {'Report', 'per_nodule_text', 'matched_sentence', 'text', 'description'}
WID = {'Report': 60, 'per_nodule_text': 55, 'matched_sentence': 55, 'text': 70, 'description': 110, 'column': 32,
       'match_tier': 10, 'nodule_loss': 18, 'kind': 12, 'exclude_reason': 22, 'Img name': 20}

# ============================================================ ME
v2 = prep(pd.read_csv(os.path.join(HERE, 'me_nodule_sentence_v2.csv')))
N = pd.read_csv(os.path.join(ROOT, 'reg2rg_dataset', 'nodules.csv'))[['pid', 'folder', 'nodule_id', 'region_source']]
v2 = v2.rename(columns={'key': 'folder'}).merge(N, on=['folder', 'nodule_id'], how='left')
xl = pd.read_excel(os.path.join(ROOT, 'lung_M_class_0001-1800_report.xlsx'), sheet_name=0)
xl.columns = ['patient ID', 'nodule', 'Img name', 'Total label NO.', 'TW_Lung_RADS', 'Report']
X = xl.merge(v2.rename(columns={'pid': 'patient ID', 'nodule_id': 'nodule'}), on=['patient ID', 'nodule'], how='left')
sk = pd.read_csv(os.path.join(ROOT, 'reg2rg_dataset', 'skipped.csv')).set_index('pid').reason.to_dict()
X['exclude_reason'] = ''
miss = X.match_tier.isna()
X.loc[miss, 'exclude_reason'] = X.loc[miss, 'patient ID'].map(sk).fillna('not_in_ME_dataset')
X.loc[miss, 'match_tier'] = 'excluded'
X = X.rename(columns={'lobe': 'lobe_seg', 'piece_lobe': 'per_nodule_lobe', 'piece_size_mm': 'per_nodule_size_mm'})
cols = ['patient ID', 'nodule', 'folder', 'Img name', 'Total label NO.', 'TW_Lung_RADS', 'lobe_seg', 'eq_diam_mm',
        'match_tier', 'nodule_loss', 'kind', 'per_nodule_lobe', 'per_nodule_size_mm', 'per_nodule_text',
        'matched_sentence', 'match_score', 'exclude_reason', 'Report']
U = pd.read_csv(os.path.join(HERE, 'me_report_lesion_without_mask.csv')).rename(columns={'key': 'folder', 'lobes': 'lobe', 'size': 'size_mm'})
out = os.path.join(ROOT, 'lung_M_class_0001-1800_report_with_perNodule_text_v2.xlsx')
write_book(out, [('per_nodule', X[cols]), ('report_lesion_without_mask', U[['folder', 'lobe', 'size_mm', 'kind', 'text']]),
                 ('README', pd.DataFrame(README, columns=['column', 'description']))], WRAP, WID)
print('ME', out); print(X.match_tier.value_counts().to_dict()); print(X[X.match_tier == 'excluded'].exclude_reason.value_counts().to_dict())

# ============================================================ LDCT
v2 = prep(pd.read_csv(os.path.join(HERE, 'ldct_nodule_sentence_v2.csv'))).rename(columns={'key': 'Rename', 'nodule_id': 'nodule'})
Nd = pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodules.csv')).rename(columns={'folder': 'Rename', 'nodule_id': 'nodule'})
A = pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodule_lobe.csv')).rename(columns={'folder': 'Rename', 'nodule_id': 'nodule', 'lobe': 'lobe_seg'})
sk = pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_skipped.csv')).set_index('folder').reason.to_dict()
X = Nd[['Rename', 'nodule', 'cy', 'cx', 'cz', 'size_vox', 'eq_diam_mm', 'side']].merge(A[['Rename', 'nodule', 'lobe_seg', 'prone', 'flag']], on=['Rename', 'nodule'], how='left')
X = X.merge(v2.drop(columns=['eq_diam_mm', 'side', 'lobe', 'zrel']), on=['Rename', 'nodule'], how='left')
X['side'] = np.where(X.lobe_seg.fillna('') != '', X.lobe_seg.fillna('').str[0], X.side)
X['exclude_reason'] = ''
miss = X.match_tier.isna()
X.loc[miss, 'exclude_reason'] = X.loc[miss].apply(lambda r: r.flag if isinstance(r.flag, str) and r.flag else sk.get(r.Rename, 'unknown'), axis=1)
X.loc[miss, 'match_tier'] = 'excluded'
X['note'] = np.where(X.prone == True, 'prone_scan', '')
rep = pd.read_excel(os.path.join(ROOT, 'LDCT_test1000_report.xlsx')).set_index('Rename').Report
X['Report'] = X.Rename.map(rep); X.loc[X.duplicated('Rename'), 'Report'] = None
X = X.rename(columns={'piece_lobe': 'per_nodule_lobe', 'piece_size_mm': 'per_nodule_size_mm'})
cols = ['Rename', 'nodule', 'cy', 'cx', 'cz', 'size_vox', 'eq_diam_mm', 'side', 'lobe_seg', 'match_tier', 'nodule_loss', 'kind',
        'per_nodule_lobe', 'per_nodule_size_mm', 'per_nodule_text', 'matched_sentence', 'match_score', 'exclude_reason', 'note', 'Report']
U = pd.read_csv(os.path.join(HERE, 'ldct_report_lesion_without_mask.csv')).rename(columns={'key': 'Rename', 'lobes': 'lobe', 'size': 'size_mm'})
out = os.path.join(ROOT, 'LDCT_test1000_report_with_perNodule_text_v2.xlsx')
write_book(out, [('per_nodule', X[cols]), ('report_lesion_without_mask', U[['Rename', 'lobe', 'size_mm', 'kind', 'text']]),
                 ('README', pd.DataFrame(README, columns=['column', 'description']))], WRAP, WID)
print('LDCT', out); print(X.match_tier.value_counts().to_dict()); print(X[X.match_tier == 'excluded'].exclude_reason.value_counts().to_dict())
