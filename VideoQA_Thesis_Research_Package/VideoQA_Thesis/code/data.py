"""Manifest preparation and deterministic, dependency-light helpers."""
import argparse, hashlib, json, random, re
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')


def norm(s):
    return ' '.join(str(s).strip().lower().split())


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def question_type(q):
    q = norm(q)
    # Heuristic descriptors, NOT official reasoning annotations.
    if any(w in q.split() for w in ['before', 'after', 'first', 'last']): return 'temporal'
    if q.startswith('why'): return 'causal'
    if q.startswith('how many'): return 'count'
    if 'color' in q or 'colour' in q: return 'color'
    if q.startswith('who'): return 'person'
    return 'descriptive'


def stratum(row, counts):
    n = counts.get(norm(row['answer']), 0)
    frequency = 'rare' if n < 20 else ('medium' if n < 100 else 'common')
    return (row.get('question_type', question_type(row['question'])), frequency)


def make_subsets(rows, counts, fraction=.5, seed=42, per_video=False):
    """Same type/frequency counts; optionally same exact video counts too."""
    if not 0 < fraction <= 1: raise ValueError('fraction must be in (0,1]')
    groups = defaultdict(list)
    for r in rows:
        key = stratum(r, counts)
        if per_video: key += (str(r['video_id']),)
        groups[key].append(r)
    rng = random.Random(seed); selected, control = [], []
    for key in sorted(groups):
        group = groups[key]
        n = max(1, int(len(group) * fraction))
        selected.extend(sorted(group, key=lambda x: (-x['reliability'], x['uid']))[:n])
        control.extend(rng.sample(sorted(group, key=lambda x: x['uid']), n))
    return selected, control


def prepare(args):
    videos = Path(args.videos); ann = Path(args.annotations); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for line in Path(args.mapping).read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 2: continue
        vid = next((x for x in parts if re.fullmatch(r'vid\d+', x)), None)
        if vid:
            name = parts[0] if parts[1] == vid else parts[1]
            mapping[int(vid[3:])] = name
    if not mapping: raise ValueError('No filename vidN pairs parsed from mapping')
    all_rows, split_ids, hashes = {}, {}, {}
    files = {p.name: p for p in videos.iterdir() if p.is_file()}
    for split in ['train', 'val', 'test']:
        path = ann / f'{split}_qa.json'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = json.loads(path.read_text(encoding='utf-8')); rows = []
        for i, item in enumerate(raw):
            value = str(item['video_id']); vid = int(value.removeprefix('vid'))
            name = mapping.get(vid)
            candidates = [name, f'{name}.avi', f'{name}.mp4'] if name else []
            found = next((files[x] for x in candidates if x in files), None)
            if found is None: raise FileNotFoundError(f'Unmapped/missing video_id={vid}, name={name}')
            rows.append(dict(uid=f'{split}-{i}', video_id=str(vid), video_path=str(found.resolve()),
                             question=item['question'], answer=norm(item['answer']), split=split,
                             question_type=question_type(item['question'])))
        split_ids[split] = {r['video_id'] for r in rows}; all_rows[split] = rows
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        if split_ids[a] & split_ids[b]: raise ValueError(f'Video overlap {a}/{b}')
    counts = Counter(r['answer'] for r in all_rows['train'])
    vocab = {a:i for i,a in enumerate(sorted(counts))}
    (out/'vocab.json').write_text(json.dumps(vocab, indent=2), encoding='utf-8')
    (out/'answer_counts.json').write_text(json.dumps(counts), encoding='utf-8')
    report = {'annotation_sha256':hashes, 'vocab_size':len(vocab), 'splits':{}}
    for split, rows in all_rows.items():
        write_jsonl(out/f'{split}.jsonl', rows)
        report['splits'][split] = dict(questions=len(rows), videos=len(split_ids[split]),
                                       answer_coverage=sum(r['answer'] in vocab for r in rows)/len(rows))
    (out/'manifest_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--videos',required=True); p.add_argument('--annotations',required=True)
    p.add_argument('--mapping',required=True); p.add_argument('--out',default='data'); prepare(p.parse_args())
