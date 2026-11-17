import os, re, io, json, sys
import numpy as np, pandas as pd
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
src=io.open(os.path.join(ROOT,'reg2rg_dataset','build_dataset.py'),encoding='utf-8').read()
exec(src[src.index('# ============================================================ 報告正規化'):src.index('# ============================================================ 主流程')])

def parse(t):
    body=strip_boiler(t); f,i=split_sections(body); out=[]; drop=0
    for sec,blk in (('findings',f),('impression',i)):
        for s0 in to_sentences(blk):
            s=clean_sentence(s0)
            if s is None: drop+=1; continue
            out.append(dict(section=sec,text=s,lobes=sentence_lobes(s),
                is_lung_lesion=bool(re.search(LESION,s,re.I)) and not re.search(EXTRA_PULM,s,re.I)))
    return out,drop

# ---- LDCT ----
xl=pd.read_excel(os.path.join(ROOT,'LDCT_test1000_report.xlsx')); xl.columns=['folder','report']; xl.report=xl.report.fillna('')
R=pd.read_csv(os.path.join(HERE,'ldct_nodule_sentence.csv')); R['matched_sentence']=R.matched_sentence.fillna('')
nn=R.groupby('folder').size(); nm=R[R.matched_sentence!=''].groupby('folder').size()
L=[]
for r in xl.itertuples():
    t=r.report; nzh=len(re.findall(r'[一-鿿]',t))
    zh=bool(re.search(r'Syngo\.Lung CAD',t)) or (nzh>20 and not re.search(r'lung apex|Low dose chest CT|Chest CT',t,re.I))
    raw=t
    if zh: L.append(dict(folder=r.folder,zh=True,n_nod=nn.get(r.folder,0))); continue
    s,drop=parse(t); clean=' '.join(x['text'] for x in s)
    L.append(dict(folder=r.folder,zh=False,n_nod=nn.get(r.folder,0),n_match=nm.get(r.folder,0),sents=s,clean=clean,
                  n_sent=len(s),drop=drop,chars=len(clean),raw=raw))
L=pd.DataFrame(L)
# ---- ME ----
ME=[json.loads(l) for l in io.open(os.path.join(ROOT,'reg2rg_dataset','reports.jsonl'),encoding='utf-8')]
man=pd.read_csv(os.path.join(ROOT,'reg2rg_dataset','manifest.csv'))
xm=pd.read_excel(os.path.join(ROOT,'lung_M_class_0001-1800_report.xlsx')); xm.columns=['pid','nodule','img','nlabel','rads','report']
rawme=xm.dropna(subset=['report']).drop_duplicates('pid').set_index('pid')['report'].to_dict()

print('================ LDCT 可用於 reg2rg 的病患 ================')
en=L[~L.zh]
print('英文報告              :',len(en))
print('  且 clean 後 >=60 字 :',(en.chars>=60).sum())
ok=en[en.chars>=60]
print('  且有 nodule mask    :',(ok.n_nod>0).sum(),' nodule',ok.n_nod.sum())
print('  且至少一顆配到句子  :',(ok.n_match>0).sum(),' 配到的 nodule',ok.n_match.fillna(0).sum())
print('  無 nodule (純陰性)  :',(ok.n_nod==0).sum())
print('ME 對照: 收 1400 位 / nodule 5392 / 配到 2417')

def stats(name, sentlists, chars, raws):
    allS=[s for ss in sentlists for s in ss]
    ll=[s for s in allS if s['is_lung_lesion']]
    nsent=np.array([len(ss) for ss in sentlists])
    cov={Lb:np.mean([any(Lb in s['lobes'] for s in ss) for ss in sentlists]) for Lb in LOBES5}
    rawj=pd.Series(raws)
    d=dict(n=len(sentlists),chars_med=np.median(chars),chars_p10=np.percentile(chars,10),chars_p90=np.percentile(chars,90),
           sent_med=np.median(nsent),lung_lesion_frac=len(ll)/len(allS),lobe_tag_frac=np.mean([bool(s['lobes']) for s in ll]),
           size_in_lesion=np.mean([bool(re.search(r'\d(?:\.\d+)?\s*(?:mm|cm)\b',s['text'],re.I)) for s in ll]),
           **{'cov_'+k:v for k,v in cov.items()},
           raw_compare=rawj.str.contains(r'compar|previous|prior',case=False).mean(),
           raw_srsimg=rawj.str.contains(r'Srs\s*/\s*Img|Se\s*/\s*Im',case=False).mean(),
           raw_templ_lowdose=rawj.str.contains(r'Low dose chest CT \(120KVP',case=False).mean(),
           raw_bullet_gt=rawj.str.contains(r'^\s*>',regex=True,flags=re.M).mean(),
           raw_numbered=rawj.str.contains(r'^\s*\d+\.',regex=True,flags=re.M).mean(),
           neg_no_tumor=rawj.str.contains(r'No (?:image )?evidence of lung tumou?r|No lesion over bilateral lungs|No parenchymal lung mass',case=False).mean())
    return d, Counter(w.lower() for s in ll for w in re.findall(r'[A-Za-z]{3,}',s['text']))

me_ids=set(man.pid)
mes=[r for r in ME if r['pid'] in me_ids]
dme,wme=stats('ME',[r['sentences'] for r in mes],[len(r['clean_report']) for r in mes],[rawme[r['pid']] for r in mes])
dl,wl=stats('LDCT',ok.sents.tolist(),ok.chars.tolist(),ok.raw.tolist())
okn=ok[ok.n_nod>0]
dln,wln=stats('LDCT+nod',okn.sents.tolist(),okn.chars.tolist(),okn.raw.tolist())
print('\n================ 報告文字比較 ================')
T=pd.DataFrame({'ME (1400)':dme,'LDCT all-en':dl,'LDCT en+nodule':dln}).round(3); print(T.to_string())
print('\n肺部病灶句 top 詞 (每千詞):')
def norm(c): tot=sum(c.values()); return {k:1000*v/tot for k,v in c.items()}
a,b=norm(wme),norm(wln)
keys=[k for k,_ in (wme+wln).most_common(45)]
print(pd.DataFrame({'ME':[a.get(k,0) for k in keys],'LDCT':[b.get(k,0) for k in keys]},index=keys).round(1).T.to_string())
voc_me=set(k for k,v in wme.items() if v>=3); voc_l=set(k for k,v in wln.items() if v>=3)
print('\nLDCT 病灶句詞彙 (>=3次) 在 ME 出現過的比例: %.1f%%'%(100*len(voc_l&voc_me)/len(voc_l)))
cov_tok=sum(v for k,v in wln.items() if k in set(wme))/sum(wln.values()); print('LDCT 病灶句 token 被 ME 詞彙涵蓋: %.1f%%'%(100*cov_tok))
# 完全重複的句子
sme=set(s['text'].lower() for r in mes for s in r['sentences'])
sl=[s['text'].lower() for ss in okn.sents for s in ss]
print('LDCT 句子與 ME 完全相同: %.1f%%'%(100*np.mean([s in sme for s in sl])))
print('\n隨機 3 份 LDCT clean report:'); 
for t in okn.clean.sample(3,random_state=3): print(' -',t[:400])
print('\n隨機 3 份 ME clean report:')
for r in pd.Series(mes).sample(3,random_state=3): print(' -',r['clean_report'][:400])
