"""Frozen supervised scorers produce reproducible synthetic-selection signals."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from data import read_jsonl,write_jsonl,norm,digest
from train import build_from_checkpoint

@torch.inference_mode()
def main(a):
    device='cuda' if torch.cuda.is_available() else 'cpu'; root=Path(a.cache)
    full,ck=build_from_checkpoint(a.full,device); blind,bk=build_from_checkpoint(a.blind,device)
    if ck['config']['question_only'] or not bk['config']['question_only']: raise ValueError('Wrong scorer modalities')
    if ck['vocab']!=bk['vocab']: raise ValueError('Scorer vocabularies differ')
    full.eval(); blind.eval(); vocab=ck['vocab']; training=read_jsonl(a.train_manifest)
    if any(r['split']!='train' for r in training): raise ValueError('Wrong-video pool must be training only')
    vids=sorted({r['video_id'] for r in training}); rng=random.Random(a.seed); out=[]
    def video(vid):
        with np.load(root/f'v_{vid}.npz') as f: return torch.from_numpy(f['features'].copy())[None].to(device)
    for row in read_jsonl(a.candidates):
        if row['split']!='train' or row['video_id'] not in vids: raise ValueError('Candidate not in training split')
        y=vocab.get(norm(row['answer']))
        if y is None: raise ValueError('OOV synthetic answer; filter/log before scoring')
        q=torch.from_numpy(np.load(root/f"q_{digest(row['question'])}.npy"))[None].to(device)
        v=video(row['video_id'])
        pF=float(full(v,q).softmax(-1)[0,y]); pQ=float(blind(v,q).softmax(-1)[0,y])
        available=[vid for vid in vids if vid!=row['video_id']]
        if len(available)<a.wrong: raise ValueError('Too few distinct training videos')
        wrong_ids=rng.sample(available,a.wrong)
        pW=[float(full(video(vid),q).softmax(-1)[0,y]) for vid in wrong_ids]
        # This baseline samples random training negatives, not semantic hard negatives.
        out.append(dict(row,p_full=pF,p_question=pQ,p_wrong_mean=float(np.mean(pW)),p_wrong_std=float(np.std(pW)),
                        utility=pF-max(pQ,float(np.mean(pW))),wrong_video_ids=wrong_ids,
                        scorer_full=str(a.full),scorer_blind=str(a.blind),negative_policy='random_train_source_ids'))
    write_jsonl(a.output,out); print(f'Scored {len(out)} candidates')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ['candidates','train-manifest','full','blind','output']: p.add_argument('--'+x,required=True)
    p.add_argument('--cache',default='cache'); p.add_argument('--wrong',type=int,default=3);p.add_argument('--seed',type=int,default=42)
    main(p.parse_args())
