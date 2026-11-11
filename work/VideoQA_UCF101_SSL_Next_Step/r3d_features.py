"""Create train.py-compatible R3D-18 video and DistilBERT question caches.

Use a different --cache directory for Kinetics-only and UCF101-SSL features.
Each video cache has shape [num_clips, 512], matching model.py/train.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from torchvision.models.video import R3D_18_Weights, r3d_18
from tqdm.auto import tqdm

from data import digest, read_jsonl
from ssl_pretrain import MEAN, STD


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_segment_clips(path: Path, num_clips: int, clip_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or count < 1:
        cap.release()
        raise ValueError(f"Could not decode video: {path}")
    clips = []
    try:
        edges = np.linspace(0, max(0, count - 1), num_clips + 1)
        for i in range(num_clips):
            start = int(round(edges[i]))
            end = int(round(edges[i + 1] - 1)) if i < num_clips - 1 else count - 1
            end = max(start, min(count - 1, end))
            ids = np.linspace(start, end, clip_frames).round().astype(int)
            frames = []
            for idx in ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"Frame decode failed: {path} frame={idx}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
                crop = resized[8:120, 8:120].astype(np.float32) / 255.0
                frames.append((crop - MEAN[None, None, :]) / STD[None, None, :])
            # [C,T,H,W]
            clips.append(np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32))
    finally:
        cap.release()
    return np.stack(clips)


def load_video_model(checkpoint: str | None, device: torch.device):
    weights = R3D_18_Weights.KINETICS400_V1 if not checkpoint else None
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    ckpt_sha = "torchvision:R3D_18_Weights.KINETICS400_V1"
    if checkpoint:
        path = Path(checkpoint)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "encoder" not in ck:
            raise ValueError("Checkpoint must be ssl_pretrain.py best.pt/last.pt with an 'encoder' state")
        model.load_state_dict(ck["encoder"], strict=True)
        ckpt_sha = sha256_file(path)
    model.to(device).eval()
    return model, ckpt_sha


@torch.inference_mode()
def cache_text_features(rows, cache: Path, model_name: str, max_length: int, batch_size: int, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    meta = {"model": model_name, "revision": getattr(model.config, "_commit_hash", None),
            "max_length": max_length, "pooling": "attention_mask_mean"}
    meta_path = cache / "text_config.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise ValueError("Text model cache differs; use a new cache directory")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    questions = sorted({row["question"] for row in rows})
    missing = [q for q in questions if not (cache / f"q_{digest(q)}.npy").exists()]
    for start in tqdm(range(0, len(missing), batch_size), desc="Question embeddings", unit="batch"):
        batch = missing[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        for q, feat in zip(batch, pooled.cpu().numpy()):
            np.save(cache / f"q_{digest(q)}.npy", feat.astype("float32"))
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return len(questions)


@torch.inference_mode()
def main(args):
    rows = read_jsonl(args.manifest)
    if not rows:
        raise ValueError("Manifest is empty")
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_sha = load_video_model(args.checkpoint, device)
    vision_meta = {"encoder": "torchvision.r3d_18", "checkpoint_sha256": ckpt_sha,
                   "num_clips": args.num_clips, "clip_frames": args.clip_frames,
                   "resolution": 112, "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()}}
    config_path = cache / "vision_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != vision_meta:
        raise ValueError("Video feature configuration differs; use a new cache directory")
    config_path.write_text(json.dumps(vision_meta, indent=2), encoding="utf-8")
    seen = {}
    for row in rows:
        vid, path = str(row["video_id"]), Path(row["video_path"])
        if vid in seen and seen[vid] != path:
            raise ValueError(f"video_id={vid} maps to multiple paths in this manifest")
        seen[vid] = path
    for vid, path in tqdm(seen.items(), desc="R3D video features", unit="video"):
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        fingerprint = hashlib.sha256(json.dumps({**vision_meta, "source": str(path.resolve()),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}, sort_keys=True).encode()).hexdigest()
        dst = cache / f"v_{vid}.npz"
        if dst.exists():
            with np.load(dst) as old:
                if str(old["fingerprint"]) == fingerprint:
                    continue
            raise ValueError(f"Stale feature cache {dst}; select a fresh cache directory")
        clips = decode_segment_clips(path, args.num_clips, args.clip_frames)
        feats = []
        for start in range(0, len(clips), args.clip_batch_size):
            x = torch.from_numpy(clips[start:start + args.clip_batch_size]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                h = model(x)
            feats.append(h.float().cpu().numpy())
        features = np.concatenate(feats).astype("float32")
        times = np.linspace(0.0, 1.0, args.num_clips, dtype=np.float32)
        np.savez_compressed(dst, features=features, times=times, fingerprint=fingerprint)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    n_questions = cache_text_features(rows, cache, args.text, args.max_length, args.text_batch_size, device)
    versions = {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "transformers", "numpy"]}
    (cache / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print(f"Cached {len(seen)} videos ({args.num_clips} x 512 features) and {n_questions} questions; device={device}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="train.jsonl, val.jsonl, or test.jsonl")
    p.add_argument("--cache", required=True, help="Use separate folders for Kinetics and UCF101-SSL")
    p.add_argument("--checkpoint", help="UCF101 SSL best.pt; omit to use Kinetics-400 R3D-18 weights")
    p.add_argument("--text", default="distilbert/distilbert-base-uncased")
    p.add_argument("--num-clips", type=int, default=8)
    p.add_argument("--clip-frames", type=int, default=8)
    p.add_argument("--clip-batch-size", type=int, default=2)
    p.add_argument("--text-batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=32)
    main(p.parse_args())
