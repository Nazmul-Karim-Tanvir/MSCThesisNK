# VideoQA thesis research package

Start with Manuscript.docx for the full research plan and equations. Start implementation with code/VideoQA_Workflow.ipynb in JupyterLab. The notebook runs scripts using the active Python kernel, so select the correct CUDA-enabled environment first.

This is a proposed study with a runnable core implementation, not a completed empirical paper. No new benchmark accuracy is claimed. The old approximately 25% validation figure is historical until reproduced. No dataset or pretrained checkpoint is bundled. PyTorch training could not be executed in the authoring environment because PyTorch and the local videos were unavailable. See VALIDATION.md for actual checks.

## Files

- Manuscript.docx and Manuscript.md: full paper scaffold, related-work analysis, method, equations, datasets, ablations, statistical plan, and completion gates.
- code/VideoQA_Workflow.ipynb: ordered Windows/Jupyter workflow.
- code/data.py: MSVD mapping, manifests, split assertions, vocabulary, matched-control helper.
- code/features.py: cached CLIP frame and DistilBERT question embeddings.
- code/model.py and code/train.py: compact temporal cross-attention model, question-only baseline, fixed-update training, checkpointing, interventions.
- code/generate.py: optional BLIP-to-FLAN descriptive synthetic pilot, with rejection logging.
- code/score.py: frozen-scorer candidate scoring with deterministic random wrong-video IDs.
- code/select_data.py: human audit export, separated logistic calibration, selected/random/confidence arms.
- code/bootstrap.py: paired video-cluster bootstrap.
- code/test_controls.py: small meaningful checks for data selection and statistics.

## Environment

Create a separate Python 3.11 environment for this package; keep your existing thesis installation intact. Install a suitable CUDA-enabled PyTorch and torchvision build from the official PyTorch instructions, then install code/requirements.txt. Register an ipykernel and select it in JupyterLab. Confirm torch.cuda.is_available() before large extraction. Python 3.11 is a conservative compatibility choice, not a claim that your current Python cannot work.

The backbone, captioner, and generator are downloaded from their model publishers on first use. Respect their licenses and record the resolved revisions. The package intentionally targets Transformers 4.x to avoid relying on changed feature-extraction return types. After a successful local smoke run, save `pip freeze` with the run logs.

## Scope and next gates

First complete a small smoke run on copied development manifests, with 8–32 examples and 20 updates. Do not report this as validation performance. Then run the full supervised baseline before producing more synthetic data. A very low training loss is not evidence of generalization.

The score script currently uses random training-video negatives. The manuscript's stronger same-content/source-aware hard-negative policy needs a content-cluster manifest and source-group validation before making that experimental claim. Numeric video ID checks alone do not find duplicate excerpts from the same source. Add source-video grouping and duplicate audits to the manifest.

The core selector model is fit on human validity labels from actual video review. Do not fill labels automatically or infer them from scorer agreement. The audit CSV is a minimal computational interface; retain the richer support/ambiguity/type labels and annotator adjudication separately. Use independent video/source groups for fit, development, and evaluation. The notebook blocks selection until those labels exist.

Confidence and proposed arms are stratified by heuristic question type and answer-frequency bucket, with optional exact per-video matching. The actual retained fraction can differ from the requested fraction because small strata keep at least one example. The calibration report records the actual value. Random controls may overlap with selected examples by chance; that is legitimate random selection, not a disjointness bug.

For final experiments, use identical initial model weights for each synthetic arm by passing --init from a shared checkpoint. Use identical human-stage updates, evaluation schedules, and seeds. The notebook demonstrates the sequence; it does not launch all multi-seed experiments automatically. The human-only repeated-data control must receive the same total updates as the synthetic-plus-human arms. Keep the test set sealed until settings are frozen.

## Research extensions not implemented

The package does not reproduce CurEvo, EvoVid, or Video-Zero. It does not include a grounding head, NExT-QA multiple-choice adapter, curriculum sampler, SSL training, semantic hard-negative mining, robust temporal generation, or source-level perceptual deduplication. These are optional research stages. On MSRVTT-QA, create an adapter into the same canonical schema and use a separate vocabulary/cache namespace; never merge numerical video IDs from different datasets into one cache.

No paper should claim those components or publishable novelty merely because they are described in the proposed protocol. Decide the final claims after the controlled experiments.
