"""Cache CLIP frame and masked-mean DistilBERT question features."""
import argparse, json, importlib.metadata
from pathlib import Path
import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection, AutoImageProcessor, AutoTokenizer, AutoModel
from data import read_jsonl, digest


def sample_frames(path, n=16):
    import cv2
    cap=cv2.VideoCapture(str(path))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS))
    if count<1 or fps<=0: cap.release(); raise ValueError(f'Invalid video {path}')
    indices=np.linspace(0,count-1,n).round().astype(int); frames=[]
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,bgr=cap.read()
            if not ok: raise ValueError(f'Frame decode failed {path} frame {i}')
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    finally: cap.release()
    return frames, indices/fps


@torch.inference_mode()
def main(args):
    rows=read_jsonl(args.manifest); root=Path(args.cache); root.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    revision=args.revision
    common=dict(revision=revision) if revision else {}
    model=CLIPVisionModelWithProjection.from_pretrained(args.vision,**common).to(device).eval()
    processor=AutoImageProcessor.from_pretrained(args.vision,**common)
    vconfig={'model':args.vision,'revision':getattr(model.config,'_commit_hash',None),'frames':args.frames,
             'processor':processor.to_dict()}
    seen=set()
    for r in rows:
        vid=r['video_id']
        if vid in seen: continue
        seen.add(vid); path=Path(r['video_path']); stat=path.stat()
        meta=dict(vconfig,source=str(path.resolve()),size=stat.st_size,mtime=stat.st_mtime_ns)
        fp=digest(json.dumps(meta,sort_keys=True)); dst=root/f'v_{vid}.npz'
        if dst.exists():
            with np.load(dst) as f:
                if str(f['fingerprint'])==fp: continue
            raise ValueError(f'Stale cache {dst}; use a new cache directory')
        frames,times=sample_frames(path,args.frames); outputs=[]
        for i in range(0,len(frames),args.extract_batch):
            x=processor(images=frames[i:i+args.extract_batch],return_tensors='pt').to(device)
            f=model(**x).image_embeds
            outputs.append(torch.nn.functional.normalize(f,dim=-1).cpu().numpy())
        np.savez(dst,features=np.concatenate(outputs).astype('float32'),times=times,fingerprint=fp)
    del model,processor
    if device=='cuda': torch.cuda.empty_cache()
    tokenizer=AutoTokenizer.from_pretrained(args.text)
    model=AutoModel.from_pretrained(args.text).to(device).eval()
    text_meta={'model':args.text,'revision':getattr(model.config,'_commit_hash',None),'max_length':args.max_length,
               'pooling':'attention_mask_mean'}
    config_path=root/'text_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=text_meta:
        raise ValueError('Text model cache differs; use new cache directory')
    config_path.write_text(json.dumps(text_meta,indent=2),encoding='utf-8')
    questions=sorted(set(r['question'] for r in rows))
    for q in questions:
        dst=root/f'q_{digest(q)}.npy'
        if dst.exists(): continue
        inputs=tokenizer(q,return_tensors='pt',truncation=True,max_length=args.max_length).to(device)
        h=model(**inputs).last_hidden_state; mask=inputs['attention_mask'].unsqueeze(-1)
        pooled=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
        np.save(dst,pooled[0].cpu().numpy().astype('float32'))
    (root/'vision_config.json').write_text(json.dumps(vconfig,indent=2),encoding='utf-8')
    (root/'versions.json').write_text(json.dumps({k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},indent=2))
    print(f'Cached {len(seen)} videos and {len(questions)} unique questions')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--cache',default='cache')
    p.add_argument('--vision',default='openai/clip-vit-base-patch32'); p.add_argument('--text',default='distilbert/distilbert-base-uncased')
    p.add_argument('--revision',default=None); p.add_argument('--frames',type=int,default=16)
    p.add_argument('--extract-batch',type=int,default=4); p.add_argument('--max-length',type=int,default=32); main(p.parse_args())
