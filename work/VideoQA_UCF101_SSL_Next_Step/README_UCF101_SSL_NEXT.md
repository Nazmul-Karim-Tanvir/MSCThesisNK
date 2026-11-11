# Next implementation: UCF101 self-supervised video features

This add-on begins the thesis experiment that is still missing from the completed CLIP question-quality pilot. It uses R3D-18 initialized from Kinetics-400 and trains it on UCF101 **without reading UCF101 class labels**. The training objective is symmetric InfoNCE on two augmented views of each video. It then extracts eight 512-dimensional video tokens per MSVD-QA video, which are compatible with the existing `train.py` and `model.py`.

This environment did not have PyTorch or the user's videos, so GPU execution has not been verified here. Run the small smoke cell in Jupyter before the full UCF101 job.

## Add the two scripts

Copy `ssl_pretrain.py` and `r3d_features.py` into the existing thesis `code` folder, beside `train.py` and `data.py`. Keep your existing code and result folders.

Required packages are PyTorch, a matching torchvision build, OpenCV, NumPy, Transformers, and tqdm. Your CUDA-enabled PyTorch is already installed; first verify that `import torch, torchvision` works and `torch.cuda.is_available()` is `True`.

## Run order

1. In `UCF101_SSL_Next_Step.ipynb`, set `CODE`, `DATA`, and `UCF_VIDEOS` to your real folders. The UCF101 folder must contain the 13,320 indexed video files.
2. Run the **32-video smoke run**. It should create `best.pt`, `last.pt`, `metrics.jsonl`, and a split manifest. This checks decoding, GPU memory, and checkpoint writing; it is not a research result.
3. If the smoke run succeeds, train on the full UCF101 folder. The starter budget is 10 epochs. This can take many hours on a 4 GB RTX 2050; keep the full run in its own output folder. If interrupted, resume from `last.pt` to the requested total epoch count.
4. Create two separate MSVD-QA caches: R3D-18 Kinetics-400 features, and R3D-18 UCF101-SSL features. Never reuse the same cache directory for both.
5. Train the same VideoQA classifier on both caches using identical train/validation manifests, update budgets, and seeds 42, 43, and 44. Compare validation results first.
6. Freeze the settings before the single final test evaluation. Do not select epochs or settings based on test results.

## How to interpret the first comparison

The first controlled question is: **does additional unlabeled UCF101 InfoNCE training improve MSVD-QA answer accuracy over the same Kinetics-400 R3D-18 initialization?** Keep the QA model, DistilBERT embeddings, training data, 3,000 updates, and seeds fixed. Report mean and standard deviation across seeds; add video-clustered confidence intervals for the locked test comparison.

This experiment tests contrastive SSL only. It does not yet test a temporal-order loss, another QA dataset, or a new model family. Add those as separately named ablations after this baseline comparison works. Do not describe the earlier CLIP pilot as an R3D/SSL result.

## Files

- `code/ssl_pretrain.py`: UCF101 R3D-18 InfoNCE pretraining, progress display, validation loss, resumable checkpoints.
- `code/r3d_features.py`: builds R3D video features and DistilBERT question features in the cache format used by the existing QA scripts.
- `UCF101_SSL_Next_Step.ipynb`: beginner-friendly Jupyter cells for smoke test, full pretraining, cache extraction, and three-seed validation comparison.

