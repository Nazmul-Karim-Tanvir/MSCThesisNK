"""Self-supervised UCF101 pretraining for an R3D-18 video encoder.

Uses video pixels only (no UCF101 class labels): two augmented views are matched
with a symmetric NT-Xent/InfoNCE objective. The saved encoder is directly
compatible with r3d_features.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18
from tqdm.auto import tqdm

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
MEAN = np.asarray([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.asarray([0.22803, 0.22145, 0.216989], dtype=np.float32)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_videos(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"UCF101 folder not found: {root}")
    paths = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if len(paths) < 4:
        raise ValueError(f"Found only {len(paths)} video files under {root}; check the UCF101 folder path")
    return paths


def decode_uniform(path: Path, frames: int = 8) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or count < 1:
        cap.release()
        raise ValueError(f"Could not decode video: {path}")
    indices = np.linspace(0, count - 1, frames).round().astype(int)
    rgb = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Frame decode failed: {path} frame={idx}")
            rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(rgb)


def make_view(frames: np.ndarray, rng: np.random.Generator, train: bool, flip: bool = False) -> torch.Tensor:
    # Shared crop/color transform across frames preserves clip-level consistency.
    resized = np.stack([cv2.resize(f, (128, 128), interpolation=cv2.INTER_AREA) for f in frames])
    if train:
        top = int(rng.integers(0, 17)); left = int(rng.integers(0, 17))
        if rng.random() < 0.5:
            resized = resized[:, :, ::-1]
        x = resized[:, top:top + 112, left:left + 112].copy().astype(np.float32) / 255.0
        brightness = float(rng.uniform(0.85, 1.15))
        contrast = float(rng.uniform(0.85, 1.15))
        mean = x.mean(axis=(0, 1, 2), keepdims=True)
        x = np.clip((x - mean) * contrast + mean, 0.0, 1.0) * brightness
        x = np.clip(x, 0.0, 1.0)
    else:
        if flip:
            resized = resized[:, :, ::-1]
        x = resized[:, 8:120, 8:120].copy().astype(np.float32) / 255.0
    x = (x - MEAN[None, None, None, :]) / STD[None, None, None, :]
    return torch.from_numpy(x.transpose(3, 0, 1, 2).copy()).float()


class UCFViews(Dataset):
    def __init__(self, paths: list[Path], seed: int, training: bool):
        self.paths, self.seed, self.training = paths, seed, training

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        # Workers are deliberately disabled in the supplied run command for Windows stability.
        rng = np.random.default_rng(self.seed + index * 1009 + (random.randrange(1_000_000) if self.training else 0))
        frames = decode_uniform(self.paths[index], frames=8)
        return make_view(frames, rng, self.training), make_view(frames, rng, self.training, flip=not self.training)


class ContrastiveR3D(nn.Module):
    def __init__(self, pretrained: bool = True, projection_dim: int = 128):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.encoder = r3d_18(weights=weights)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                       nn.Linear(512, projection_dim))

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=-1)
        return h, z


def ntxent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    if z1.shape[0] < 2:
        raise ValueError("InfoNCE requires batch size >= 2; keep --batch-size at least 2")
    z = torch.cat([z1, z2], dim=0)
    logits = (z @ z.T) / temperature
    logits.fill_diagonal_(torch.finfo(logits.dtype).min)
    n = z1.shape[0]
    targets = torch.cat([torch.arange(n, 2 * n, device=z.device), torch.arange(n, device=z.device)])
    return F.cross_entropy(logits, targets)


@torch.inference_mode()
def validation_loss(model, loader, device, temperature, amp):
    model.eval(); losses = []
    for x1, x2 in loader:
        x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            _, z1 = model(x1); _, z2 = model(x2)
            loss = ntxent(z1.float(), z2.float(), temperature)
        losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("nan")


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main(args):
    seed_all(args.seed)
    paths = list_videos(Path(args.videos))
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    if args.max_videos:
        paths = paths[:args.max_videos]
    n_val = max(2, round(len(paths) * args.val_fraction))
    if len(paths) < 6 or n_val >= len(paths):
        raise ValueError("Need at least 6 videos and a non-empty train/validation split")
    val_paths, train_paths = paths[:n_val], paths[n_val:]
    split_hash = hashlib.sha256(("\n".join(map(str, train_paths)) + "\n--VAL--\n" +
                                "\n".join(map(str, val_paths))).encode()).hexdigest()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    split_file = out / "split_manifest.jsonl"
    with split_file.open("w", encoding="utf-8") as f:
        for split, group in (("train", train_paths), ("val", val_paths)):
            for p in group:
                f.write(json.dumps({"path": str(p), "split": split}) + "\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    model = ContrastiveR3D(pretrained=not args.from_scratch).to(device)
    train_ds = UCFViews(train_paths, args.seed, True)
    val_ds = UCFViews(val_paths, args.seed + 1, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=amp)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
                            num_workers=args.workers, pin_memory=amp)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Train and validation each need at least two videos for contrastive batches")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if ck.get("split_hash") != split_hash:
            raise ValueError("Resume checkpoint uses a different UCF101 split; choose a fresh output folder")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck.get("scaler", {})); start_epoch = int(ck["epoch"])
        best_val = float(ck.get("best_val_loss", best_val))
        if "rng" in ck:
            random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
            torch.set_rng_state(ck["rng"]["torch"].cpu())
            if device.type == "cuda" and ck["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    config = vars(args).copy()
    (out / "run_config.json").write_text(json.dumps({**config, "device": str(device),
        "train_videos": len(train_paths), "val_videos": len(val_paths), "class_labels_used": False}, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        model.train(); running = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for x1, x2 in bar:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                _, z1 = model(x1); _, z2 = model(x2)
                loss = ntxent(z1.float(), z2.float(), args.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite InfoNCE loss; inspect video decoding or lower the learning rate")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running += float(loss.detach()); bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
        train_loss = running / max(1, len(train_loader))
        val_loss = validation_loss(model, val_loader, device, args.temperature, amp)
        record = {"epoch": epoch + 1, "train_infonce": train_loss, "val_infonce": val_loss,
                  "train_videos": len(train_paths), "val_videos": len(val_paths)}
        write_jsonl(out / "metrics.jsonl", record)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save = {"model": state, "encoder": {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()},
                "epoch": epoch + 1, "config": config, "split_hash": split_hash, "class_labels_used": False}
        torch.save({**save, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "best_val_loss": min(best_val, val_loss),
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}}, out / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({**save, "best_val_loss": best_val}, out / "best.pt")
        print(json.dumps(record), flush=True)
    print(f"Best UCF101 validation InfoNCE: {best_val:.5f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True, help="Root folder containing UCF101 video files")
    p.add_argument("--output", default="runs/ucf101_r3d18_ssl")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-videos", type=int, default=0, help="Use 32 for the smoke run; 0 uses all videos")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from-scratch", action="store_true", help="Disable Kinetics-400 initialization")
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    p.add_argument("--resume", help="Resume from last.pt")
    main(p.parse_args())
