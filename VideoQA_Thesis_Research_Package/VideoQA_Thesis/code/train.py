"""Fixed update training and deterministic input interventions. Trusted local checkpoints only."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm.auto import tqdm
from data import read_jsonl,write_jsonl,norm,digest
from model import VideoQA


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class CachedDataset(Dataset):
    def __init__(self,rows,cache,vocab,training=False):
        self.rows=rows; self.cache=Path(cache); self.vocab=vocab
        if not rows: raise ValueError('Empty dataset')
        if training and any(norm(r['answer']) not in vocab for r in rows): raise ValueError('Unknown training answer')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with np.load(self.cache/f"v_{r['video_id']}.npz") as f: v=f['features'].copy()
        q=np.load(self.cache/f"q_{digest(r['question'])}.npy")
        return torch.from_numpy(v),torch.from_numpy(q),self.vocab.get(norm(r['answer']),-1),i


def load_checkpoint(path,device):
    return torch.load(path,map_location=device,weights_only=False)


def build_from_checkpoint(path,device):
    ck=load_checkpoint(path,device); model=VideoQA(**ck['config']).to(device)
    model.load_state_dict(ck['model']); return model,ck


@torch.inference_mode()
def evaluate(model,loader,device,mode='full',seed=42):
    model.eval(); rows=loader.dataset.rows; rng=np.random.default_rng(seed)
    ids=sorted(set(r['video_id'] for r in rows)); mapping={}
    if mode=='wrong':
        if len(ids)<2: raise ValueError('Wrong-video test requires at least two distinct videos')
        # Seeded cyclic derangement, stable across batch boundaries; repeat using several seeds.
        order=list(rng.permutation(ids)); mapping={x:order[(j+1)%len(order)] for j,x in enumerate(order)}
    out=[]; correct=0
    for v,q,y,index in loader:
        if mode=='zero': v.zero_()
        elif mode=='single': v=v[:,v.shape[1]//2:v.shape[1]//2+1].expand_as(v).clone()
        elif mode=='order':
            v=torch.stack([x[torch.as_tensor(rng.permutation(x.shape[0]))] for x in v])
        elif mode=='wrong':
            values=[]
            for ix in index.tolist():
                vid=mapping[rows[ix]['video_id']]
                with np.load(loader.dataset.cache/f'v_{vid}.npz') as f: values.append(torch.from_numpy(f['features'].copy()))
            v=torch.stack(values)
        logits=model(v.to(device),q.to(device)); pred=logits.argmax(-1).cpu(); probs=logits.softmax(-1).cpu()
        correct+=int((pred==y).sum())
        for j,ix in enumerate(index.tolist()):
            row=rows[ix]
            out.append(dict(uid=row['uid'],video_id=row['video_id'],target=int(y[j]),prediction=int(pred[j]),
                            correct=int(pred[j]==y[j]),mode=mode,confidence=float(probs[j,pred[j]])))
    return correct/len(rows),out


def main(args):
    seed_all(args.seed); device='cuda' if torch.cuda.is_available() else 'cpu'
    vocab=json.loads(Path(args.vocab).read_text()); valrows=read_jsonl(args.val)
    val=DataLoader(CachedDataset(valrows,args.cache,vocab),batch_size=args.batch_size,shuffle=False,num_workers=0)
    if args.eval_checkpoint:
        model,ck=build_from_checkpoint(args.eval_checkpoint,device)
        if ck['vocab']!=vocab: raise ValueError('Vocabulary mismatch')
        acc,preds=evaluate(model,val,device,args.mode,args.seed); write_jsonl(args.output,preds)
        print(json.dumps({'accuracy':acc,'n':len(preds),'mode':args.mode})); return
    trainrows=read_jsonl(args.train)
    ds=CachedDataset(trainrows,args.cache,vocab,True)
    v,q,_,_=ds[0]; config=dict(answers=len(vocab),video_dim=v.shape[-1],text_dim=q.shape[-1],
                              hidden=args.hidden,frames=v.shape[0],question_only=args.question_only)
    model=VideoQA(**config).to(device)
    if args.init:
        ck=load_checkpoint(args.init,device)
        if ck['config']!=config or ck['vocab']!=vocab: raise ValueError('Initialization configuration differs')
        model.load_state_dict(ck['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=generator)
    # A resume restarts at a saved epoch boundary. Use --init for a new synthetic/human stage.
    step=0; best=-1.; best_state=None
    if args.resume:
        ck=load_checkpoint(args.resume,device)
        if ck['config']!=config or ck['vocab']!=vocab: raise ValueError('Resume configuration differs')
        if ck['train_hash']!=digest(json.dumps(trainrows,sort_keys=True)): raise ValueError('Resume data differ')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); step=ck['step']; best=ck['best']
        best_state=ck.get('best_model'); generator.set_state(ck['loader_rng'])
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device=='cuda' and ck.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    bar=tqdm(total=args.steps,initial=step,desc='Updates'); model.train()
    while step<args.steps:
        for v,q,y,_ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(v.to(device),q.to(device)); loss=F.cross_entropy(logits,y.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            step+=1; bar.update(1); bar.set_postfix(loss=f'{loss.item():.4f}')
            if step>=args.steps: break
        # Validation/checkpoint at an epoch boundary (or final partial epoch).
        acc,preds=evaluate(model,val,device); model.train()
        record={'step':step,'validation_accuracy':acc,'last_batch_loss':loss.item()}
        with (out/'metrics.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if acc>best:
            best=acc; best_state=state
            torch.save({'model':state,'config':config,'vocab':vocab,'step':step},out/'best.pt')
            write_jsonl(out/'validation_predictions.jsonl',preds)
        torch.save(dict(model=state,config=config,vocab=vocab,optimizer=optimizer.state_dict(),step=step,best=best,
                        best_model=best_state,train_hash=digest(json.dumps(trainrows,sort_keys=True)),
                        loader_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None,
                        python_rng=random.getstate(),numpy_rng=np.random.get_state()),out/'last.pt')
        (out/'run_config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
        # Resumed run also restores the best checkpoint into its output folder.
        if best_state is not None and not (out/'best.pt').exists():
            torch.save({'model':best_state,'config':config,'vocab':vocab,'step':step},out/'best.pt')
    bar.close(); print('Best validation accuracy:',best)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--train'); p.add_argument('--val',required=True)
    p.add_argument('--cache',default='cache'); p.add_argument('--vocab',default='data/vocab.json'); p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=3000); p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--hidden',type=int,default=256); p.add_argument('--lr',type=float,default=.0003); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--question-only',action='store_true'); p.add_argument('--init'); p.add_argument('--resume')
    p.add_argument('--eval-checkpoint'); p.add_argument('--mode',choices=['full','zero','wrong','order','single'],default='full')
    args=p.parse_args()
    if not args.eval_checkpoint and not args.train: p.error('--train is required for training')
    if args.init and args.resume: p.error('Use only --init or --resume')
    main(args)
