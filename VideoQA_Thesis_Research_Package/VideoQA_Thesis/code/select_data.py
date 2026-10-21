"""Source-video separated human-audit calibration and distribution-matched controls."""
import argparse,json,csv,random,pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from data import read_jsonl,write_jsonl,make_subsets,norm,stratum

TYPES=['descriptive','color','person','count','temporal','causal']
FEATURES=['p_full','p_question','p_wrong_mean','p_wrong_std','utility','agreement']

def matrix(rows):
    return np.asarray([[float(r[k]) for k in FEATURES]+[int(r.get('question_type')==t) for t in TYPES] for r in rows])

def make_audit(a):
    rows=read_jsonl(a.scores); groups=defaultdict(list)
    for r in rows: groups[r['video_id']].append(r)
    ids=sorted(groups); random.Random(a.seed).shuffle(ids)
    if len(ids)<9: raise ValueError('At least 9 videos required for three source-disjoint audit partitions')
    # Separate video pools first; sample up to requested total across them.
    pools=[ids[:len(ids)//2],ids[len(ids)//2:3*len(ids)//4],ids[3*len(ids)//4:]]
    rng=random.Random(a.seed); audit=[]
    for name,pool,n in zip(['fit','dev','eval'],pools,[a.n//2,a.n//4,a.n-a.n//2-a.n//4]):
        candidates=[r for vid in pool for r in groups[vid]]; rng.shuffle(candidates)
        for r in candidates[:n]: audit.append(dict(uid=r['uid'],video_id=r['video_id'],audit_split=name,valid=''))
    with open(a.audit,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['uid','video_id','audit_split','valid']);w.writeheader();w.writerows(audit)
    # Unblinded scores are kept separate from the annotation packet.
    lookup={r['uid']:r for r in rows}
    packet=[{k:lookup[x['uid']].get(k) for k in ['uid','video_id','video_path','question','answer','evidence_times']} for x in audit]
    write_jsonl(str(a.audit)+'.annotation_packet.jsonl',packet)
    print('Annotate valid as 0/1 against actual videos. Use separate rubric fields for detailed audit labels.')


def main(a):
    if a.make_audit: return make_audit(a)
    rows=read_jsonl(a.scores); by_uid={r['uid']:r for r in rows}
    if len(by_uid)!=len(rows): raise ValueError('Duplicate candidate IDs')
    with open(a.audit,newline='',encoding='utf-8') as f: labels=list(csv.DictReader(f))
    groups=defaultdict(list); video_group={}; banned=set()
    for x in labels:
        if x['valid'] not in ['0','1']: raise ValueError(f"Unlabelled/invalid audit label {x['uid']}")
        if x['audit_split'] not in ['fit','dev','eval']: raise ValueError('Unknown audit split')
        row=by_uid[x['uid']]; vid=row['video_id']; group=x['audit_split']; banned.add(vid)
        if x['video_id']!=vid: raise ValueError('Audit video mismatch')
        if vid in video_group and video_group[vid]!=group: raise ValueError('Audit source video crosses partitions')
        video_group[vid]=group; groups[group].append((row,int(x['valid'])))
    for name in ['fit','dev','eval']:
        if len(groups[name])<10: raise ValueError(f'{name}: at least 10 labels required; 100+ recommended')
        if len({y for _,y in groups[name]})<2: raise ValueError(f'{name} must include valid and invalid labels')
    def xy(name):
        return matrix([r for r,_ in groups[name]]),np.array([y for _,y in groups[name]])
    xf,yf=xy('fit'); xd,yd=xy('dev'); xe,ye=xy('eval')
    best=None
    for c in [.01,.1,1.,10.]:
        m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=a.seed)).fit(xf,yf)
        loss=brier_score_loss(yd,m.predict_proba(xd)[:,1])
        if best is None or loss<best[0]:best=(loss,m,c)
    _,model,c=best; pred=model.predict_proba(xe)[:,1]
    report={'n_fit':len(yf),'n_dev':len(yd),'n_eval':len(ye),'C':c,
            'dev_brier':best[0],'eval_brier':brier_score_loss(ye,pred),
            'eval_confidence_brier':brier_score_loss(ye,[r['p_full'] for r,_ in groups['eval']]),'reliability_bins':[]}
    for low in np.arange(0,1,.2):
        mask=(pred>=low)&(pred<(low+.2) if low<.8 else pred<=1)
        report['reliability_bins'].append({'lower':float(low),'n':int(mask.sum()),
            'mean_prediction':float(pred[mask].mean()) if mask.any() else None,
            'valid_fraction':float(ye[mask].mean()) if mask.any() else None})
    pool=[];seen=set()
    for row in rows:
        if row['video_id'] in banned: continue
        key=(row['video_id'],norm(row['question']),norm(row['answer']))
        if key in seen: continue
        seen.add(key);pool.append(row)
    if not pool: raise ValueError('No synthetic pool remains after removing audit videos')
    for row,r in zip(pool,model.predict_proba(matrix(pool))[:,1]):row['reliability']=float(r)
    counts=json.loads(Path(a.counts).read_text()); selected,control=make_subsets(pool,counts,a.fraction,a.seed,a.per_video)
    # Confidence-only control uses the same strata and size as the calibrated arm.
    confidence_pool=[dict(r,reliability=r['p_full']) for r in pool]
    confidence,_=make_subsets(confidence_pool,counts,a.fraction,a.seed,a.per_video)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for name,data in [('selected',selected),('random',control),('confidence',confidence),('pool',pool)]:write_jsonl(out/f'{name}.jsonl',data)
    report.update(pool_size=len(pool),selected_size=len(selected),requested_fraction=a.fraction,
                  actual_fraction=len(selected)/len(pool),excluded_audit_videos=len(banned),per_video_matching=a.per_video)
    (out/'calibration_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'selector.pkl').open('wb') as f:pickle.dump({'model':model,'features':FEATURES,'types':TYPES},f)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--counts',default='data/answer_counts.json');p.add_argument('--output',default='selection')
    p.add_argument('--fraction',type=float,default=.5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--make-audit',action='store_true');p.add_argument('--n',type=int,default=400)
    p.add_argument('--per-video',action='store_true');main(p.parse_args())
