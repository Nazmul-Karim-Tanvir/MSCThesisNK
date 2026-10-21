"""Paired video-cluster bootstrap. One run/seed, not training-seed uncertainty."""
import argparse,json,random
from collections import defaultdict
from data import read_jsonl

def paired_cluster(a,b,repetitions=2000,seed=42):
    aa={r['uid']:r for r in a};bb={r['uid']:r for r in b}
    if len(aa)!=len(a) or len(bb)!=len(b):raise ValueError('Duplicate prediction IDs')
    if aa.keys()!=bb.keys():raise ValueError('Prediction UID sets differ')
    groups=defaultdict(list)
    for uid,r in aa.items():
        s=bb[uid]
        if r['video_id']!=s['video_id'] or r.get('target')!=s.get('target'):raise ValueError('Pair identity differs')
        groups[r['video_id']].append(r['correct']-s['correct'])
    if len(groups)<2:raise ValueError('Need at least two video clusters')
    rng=random.Random(seed);ids=sorted(groups);differences=[]
    for _ in range(repetitions):
        sampled=rng.choices(ids,k=len(ids));num=0;den=0
        for vid in sampled:num+=sum(groups[vid]);den+=len(groups[vid])
        differences.append(num/den)
    differences.sort();n=len(differences)
    return {'accuracy_difference':sum(sum(g) for g in groups.values())/len(a),
            'ci95':[differences[int(.025*n)],differences[min(n-1,int(.975*n))]],
            'clusters':len(groups),'questions':len(a),'repetitions':repetitions}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a',required=True);p.add_argument('--b',required=True)
    p.add_argument('--repetitions',type=int,default=2000);p.add_argument('--seed',type=int,default=42)
    x=p.parse_args();print(json.dumps(paired_cluster(read_jsonl(x.a),read_jsonl(x.b),x.repetitions,x.seed),indent=2))
