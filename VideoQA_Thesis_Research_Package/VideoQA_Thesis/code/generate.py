"""Descriptive pilot only: BLIP captions then FLAN-T5 QA and text verification.
No temporal or hidden-content claims. All malformed/OOV outputs are logged.
"""
import argparse,json
from pathlib import Path
import torch
from transformers import BlipProcessor,BlipForConditionalGeneration,AutoTokenizer,AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
from data import read_jsonl,write_jsonl,norm,digest,question_type
from features import sample_frames

@torch.inference_mode()
def main(a):
    rows=read_jsonl(a.manifest)
    if any(r['split']!='train' for r in rows): raise ValueError('Generation is restricted to training videos')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if a.mode=='captions':
        model=BlipForConditionalGeneration.from_pretrained(a.captioner).to(device).eval()
        processor=BlipProcessor.from_pretrained(a.captioner); unique={r['video_id']:r for r in rows}
        output=[]
        for vid,r in tqdm(sorted(unique.items())[:a.limit],desc='Caption videos'):
            frames,times=sample_frames(r['video_path'],a.frames)
            for j,(frame,timestamp) in enumerate(zip(frames,times)):
                inputs=processor(images=frame,return_tensors='pt').to(device)
                text=processor.decode(model.generate(**inputs,max_new_tokens=40,do_sample=False)[0],skip_special_tokens=True)
                output.append(dict(video_id=vid,video_path=r['video_path'],split='train',caption=text,
                                   timestamp=float(timestamp),captioner=a.captioner,
                                   captioner_revision=getattr(model.config,'_commit_hash',None),source_id=f'{vid}-f{j}'))
        write_jsonl(a.output,output); return
    tokenizer=AutoTokenizer.from_pretrained(a.generator)
    model=AutoModelForSeq2SeqLM.from_pretrained(a.generator).to(device).eval()
    vocab=json.loads(Path(a.vocab).read_text()); accepted=[]; rejected=[]; seen=set()
    def ask(prompt):
        x=tokenizer(prompt,return_tensors='pt',truncation=True,max_length=512).to(device)
        return tokenizer.decode(model.generate(**x,max_new_tokens=80,do_sample=False)[0],skip_special_tokens=True).strip()
    for r in tqdm(read_jsonl(a.captions),desc='Candidate QA'):
        caption=r['caption']
        prompt=('Write one short question and its short answer using only the caption. '
                'Ask about a visible object or action. Do not infer liquid identity, intention, time order, or cause. '
                'Use exactly this format: QUESTION: question | ANSWER: answer. Caption: '+caption)
        raw=ask(prompt)
        try:
            qpart,apart=raw.split('| ANSWER:',1); q=qpart.split('QUESTION:',1)[1].strip(); ans=norm(apart)
            if not q or not ans or len(ans.split())>5: raise ValueError('malformed')
            kind=question_type(q)
            if kind in ['temporal','causal','count']: raise ValueError('unsupported question category')
            if ans not in vocab: raise ValueError('out_of_vocabulary')
            key=(r['video_id'],norm(q),ans)
            if key in seen: raise ValueError('duplicate')
            verified=norm(ask('Answer the question using only the caption. If unsupported, answer unknown. '
                              'Give only a short answer. Caption: '+caption+' Question: '+q))
            seen.add(key)
            accepted.append(dict(uid='s-'+digest('|'.join(key))[:24],video_id=r['video_id'],video_path=r['video_path'],
                                 split='train',question=q,answer=ans,question_type=kind,evidence_text=caption,
                                 evidence_times=[r['timestamp']],source_id=r['source_id'],captioner=r['captioner'],
                                 captioner_revision=r.get('captioner_revision'),generator=a.generator,
                                 generator_revision=getattr(model.config,'_commit_hash',None),prompt_version='descriptive-v1',
                                 verifier_answer=verified,agreement=int(verified==ans)))
        except (ValueError,IndexError) as e:
            rejected.append(dict(source_id=r['source_id'],raw=raw,reason=str(e)))
    write_jsonl(a.output,accepted); write_jsonl(str(a.output)+'.rejected.jsonl',rejected)
    print({'accepted':len(accepted),'rejected':len(rejected)})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['captions','qa'],required=True)
    p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--captions'); p.add_argument('--vocab',default='data/vocab.json')
    p.add_argument('--limit',type=int,default=200); p.add_argument('--frames',type=int,default=4)
    p.add_argument('--captioner',default='Salesforce/blip-image-captioning-base'); p.add_argument('--generator',default='google/flan-t5-base')
    a=p.parse_args()
    if a.mode=='qa' and not a.captions: p.error('--captions required in qa mode')
    main(a)
