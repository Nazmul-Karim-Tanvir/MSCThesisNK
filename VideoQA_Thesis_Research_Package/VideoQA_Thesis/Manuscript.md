# Evidence Calibrated Synthetic Supervision for Reliable Video Question Answering

Nazmul Karim Tanvir
Department of Computer Science and Engineering, BRAC University
Supervisor: Md. Golam Rabiul Alam, PhD
Research manuscript and experimental protocol • 21 October 2026

## Abstract
Video question answering requires predictions that are both accurate and supported by the observed video. Automatically generated question–answer pairs offer additional training supervision, but they can contain unsupported details, ambiguous references, and answers predictable without visual evidence. This study proposes evidence-calibrated selection of synthetic supervision for a compact VideoQA model. A fixed candidate pool is scored using evidence-conditioned verification, question-only predictability, and answer confidence under matched wrong-video substitutions. A small, separately partitioned human audit calibrates these signals to the probability that an example is visually supported and unambiguous. Selection is evaluated against random, confidence-only, and uncalibrated filtering controls, including controls matched for question type, answer frequency, video exposure, and optimization budget. A frozen visual encoder and text encoder support efficient training of temporal and cross-attention modules. MSVD-QA provides continuity with an existing baseline, while a second QA dataset and a temporal-grounding benchmark test transfer and evidence alignment. The primary hypothesis is that calibrated selection improves the accuracy–data-efficiency trade-off without reducing performance on legitimate, visually answerable questions. A secondary hypothesis tests whether a simple fixed curriculum adds value beyond selection alone. The manuscript specifies the method, implementation, statistical analysis, and result reporting protocol; the proposed method has not yet produced benchmark results.

Keywords: video question answering; synthetic supervision; data selection; visual evidence; calibration; self-supervised representation learning.

## 1 Introduction
A VideoQA system receives a video and a natural-language question and predicts an answer. Some questions require identifying an object visible in one frame. Others require observing an action, counting repeated events, or identifying which event occurred first. A useful system should recognize this distinction: temporal sensitivity is valuable when the answer depends on order, while a stable answer is appropriate for many appearance questions.

Our starting implementation combines an R3D-18 video encoder, DistilBERT question representations, cross-attention, and a fixed answer classifier. Additional contrastive learning has been attempted on UCF101. Previously recorded MSVD-QA validation accuracy is approximately 25%, with 25.24% reported during a fine-tuning run. This is an experimental starting point, not a verified final benchmark result. The original checkpoint, preprocessing, answer vocabulary, and split definitions must be recovered before making a reproducible comparison.

The practical objective is to improve this system while producing a focused MSc contribution. Simply increasing model size or combining familiar components does not establish novelty. We instead ask which synthetic examples help a fixed downstream model, and whether their usefulness is explained by reliable visual support rather than confidence, dataset size, or answer-distribution changes.

There are two independent problems. First, a generated answer can be wrong: a caption may describe someone drinking from a glass without establishing whether the liquid is water. Second, a correct answer can be weak training material: an unusually stereotyped question may reward a model for ignoring the video. These properties are related but not identical. High question-only confidence does not prove an invalid question; low question-only confidence does not prove a correct visual label.

We therefore separate validity estimation from visual-specificity diagnostics. The proposed selector learns a small calibration model from audited training candidates and uses the result to select a fixed number of examples. Crucially, its value is assessed under matched controls and by reviewers independent of the selector. This limits circular claims in which a system defines quality using its own confidence and then announces success because its confidence increases.

## 2 Problem scope and research questions
The primary setting is supervised downstream VideoQA with additional machine-generated supervision from training videos. Benchmark training labels are permitted; validation labels select checkpoints; test data remain sealed. A pretrained captioner or generator transfers external supervision into the pipeline. Consequently, this is not an annotation-free, purely self-supervised, or zero-shot system. Self-supervision refers specifically to optional representation adaptation without action or QA labels.

RQ1 asks whether calibrated selection improves downstream accuracy compared with random and confidence-based selection from the same synthetic candidate pool. RQ2 asks whether gains persist after matching sample counts, optimizer updates, video exposure, answer frequencies, and question types. RQ3 asks whether selected data improve independently measured evidence support and reduce dependence on irrelevant video substitutions. RQ4 asks whether gains transfer to another dataset and another feature backbone. RQ5, secondary, asks whether fixed difficulty progression improves a selected dataset relative to random ordering.

The principal contribution candidate is an evidence-calibrated selection protocol for compact VideoQA, evaluated with unusually strong controls. The work does not claim to invent synthetic QA, temporal grounding, curriculum learning, confidence filtering, or counterfactual input tests. Whether the combination is sufficiently original for a particular venue remains an empirical and literature-review question.

## 3 Research analysis and related work
### 3.1 Synthetic video supervision
Just Ask generates QA from narrated video and establishes scalable automatic supervision [1]. It motivates synthetic training but also makes scale alone an unsuitable contribution for this thesis. Our contrast is a small fixed candidate pool and controlled selection, rather than an attempt to reproduce its dataset scale. A useful baseline inspired by this line of work is to train on all generated QA without the proposed selector.

FrozenBiLM shows that light trainable modules can connect visual inputs to a frozen language model [2]. This supports efficient adaptation as a credible engineering choice, while also showing that freezing a backbone is not itself novel. Its zero-shot results must not be directly ranked against our supervised answer classifier without matching the task setting and training resources.

### 3.2 Evidence and temporal reasoning
NExT-QA emphasizes causal and temporal action questions in addition to descriptive ones [3]. It is relevant because a model can improve on descriptive QA without becoming a stronger temporal reasoner. We use its question taxonomy for analysis rather than inferring reasoning categories solely from question length.

Can I Trust Your Answer introduces NExT-GQA and demonstrates why QA accuracy should be accompanied by evidence localization [4]. It directly precedes the proposed grounding audit. Our wrong-video diagnostic is weaker than annotated temporal grounding and will never be described as equivalent. If grounding becomes a central title claim, evaluation against human temporal labels is required.

Grounded Question-Answering in Long Egocentric Videos jointly addresses answering and temporal localization, with synthetic-data support [5]. The overlap is substantial at the pipeline level. Our study instead examines the marginal benefit of a calibrated selector in compact models; it does not propose a new long-video grounding architecture.

The atemporal probe study shows that image-level information can solve substantial portions of video-language benchmarks [6]. This means frame-order invariance is sometimes correct. We therefore report temporal perturbation effects on order-sensitive subsets and include a single-frame baseline; we do not filter every example for large shuffle sensitivity.

### 3.3 Self evolution and the novelty boundary
CurEvo explicitly organizes self-evolution with curricula over task difficulty, evaluation, and diversity [7]. Curriculum over generated QA is therefore established prior art. The optional schedule here is deliberately fixed and lightweight; calling it a reproduction of CurEvo would be inaccurate. A genuine CurEvo comparison would require its actual implementation and matched resources.

EvoVid already uses temporal perturbation sensitivity and segment localization rewards for Video-LLM self-evolution [8]. A claim that this thesis introduces temporal perturbation as a new training signal would be unsupported. We use interventions as selector features and evaluation controls, without proposing questioner–solver reinforcement learning.

Video-Zero centers self-evolution on localized evidence and evidence-aligned learning [9]. It further narrows the gap: evidence-centered generated supervision is already studied. Our defensible question concerns calibration, compact-model transfer, and selection effects under matched distribution and compute controls. This is a proposed distinction, not proof that an identical selector has never appeared.

### 3.4 Representation learning
VideoMAE provides masked video reconstruction as a self-supervised representation-learning route [10]. CLIP provides pretrained image–text representations [11], and DistilBERT provides a compact language encoder [12]. We treat these as alternative infrastructure. The main comparison holds the backbone fixed; adding VideoMAE, CLIP, contrastive losses, and curricula simultaneously would make attribution difficult.

### 3.5 Evidence strength of this review
The review combines primary papers, official repositories, and abstracts. CurEvo and EvoVid PDFs were retrieved and inspected for their stated mechanisms; Video-Zero was checked at abstract level. This is a targeted research refresh, not an exhaustive systematic review or verification of every reported result. Before submission, inspect the complete methods and supplements of the closest works, their released code where available, and forward citations. No borrowed benchmark numbers are included without a verified matching protocol.

| Prior line | Established capability | Thesis decision |
| --- | --- | --- |
| Just Ask | Automatically generated video QA | Use generation as infrastructure |
| FrozenBiLM | Efficient frozen-model adaptation | Include an efficient backbone control |
| NExT-GQA | QA with visual evidence evaluation | Require independent evidence checks |
| ATP | Strong image-only video baselines | Separate static and temporal questions |
| CurEvo | Curriculum guided self-evolution | Make curriculum a secondary ablation |
| EvoVid and Video-Zero | Temporal and evidence-centered evolution | Avoid broad first-method claims |
| Proposed study | Calibrated selection under matched controls | Test contribution before asserting novelty |

## 4 Data and supervision protocol
### 4.1 Dataset roles
MSVD-QA is the first downstream dataset because the local implementation and video mapping already exist. The official source describes MSVD-QA and MSRVTT-QA as QA datasets constructed from existing video descriptions [13]. Their source annotations should not be described as entirely independent human-written questions. We will compute all split counts from the downloaded files rather than copy potentially mismatched statistics.

UCF101 remains an optional representation-learning dataset, using video pixels without action-class labels. Its action distribution is narrower than general QA and it is not automatically the best source of synthetic object questions. For the first synthetic-data pilot, use MSVD training videos with QA labels hidden from the generator. This provides relevant domain content and makes the provenance clear. It is in-domain auxiliary supervision, not external unlabeled-data transfer.

MSRVTT-QA is the preferred second classification benchmark. NExT-QA and NExT-GQA provide a separate multiple-choice and evidence-localization extension [3,4]. These use a candidate scorer and a grounding adapter rather than silently reusing the fixed MSVD answer head. If only MSVD-QA is completed, narrow the paper to that dataset; do not claim broad video reasoning generalization.

| Dataset | Role | Labels permitted |
| --- | --- | --- |
| MSVD training videos | Main QA training and candidate generation | Training QA only in supervised stages |
| MSVD validation | Checkpoint and limited hyperparameter selection | Validation QA |
| MSVD test | Locked final evaluation | Evaluation only |
| UCF101 | Optional representation adaptation | Pixels only for SSL |
| MSRVTT-QA | Independent replication of the recipe | Its official training QA |
| NExT-GQA | Optional direct evidence evaluation | Specify training-label use explicitly |

### 4.2 Local data preparation
The earlier MSVD video directory was F:\MSC\thesis\dataset2\MSVD\Videos\MSVD_Videos\YouTubeClips. The annotation directory was F:\MSC\thesis\dataset2\MSVD\Annotations\MSVD-QA\MSVD-QA. Reconfirm these paths locally. Parse youtube_mapping.txt explicitly as filename-to-vidN mapping, never by filesystem order. Generate a canonical manifest with uid, video_id, video_path, question, answer, split, and question_type.

The historical train file contained 30,933 QA records, and 1,970 local video files were listed. These observations are not a substitute for a fresh manifest audit. Record missing, corrupt, or duplicate videos and a hash of every annotation file. Assert that video IDs do not cross official splits. Also group multiple clips from the same original source when creating internal audit folds. Exact hashes catch exact duplicates; perceptual checks and source identifiers are needed for overlapping excerpts.

### 4.3 Answer vocabulary and coverage
Build the answer vocabulary from normalized training answers only. Lowercase, trim, and collapse whitespace; do not introduce a new article-removal policy without comparing it to the official evaluation. Preserve the complete vocabulary initially; a top-K vocabulary is a separate experiment. Unknown validation or test answers receive no class and count as incorrect. Report answer coverage and the corresponding closed-vocabulary accuracy ceiling.

Synthetic answers are mapped to this same vocabulary for the classifier experiment. Report how many generated examples are out of vocabulary, rejected, or normalized. Never silently remove unknown benchmark answers from the evaluation denominator. For a multiple-choice extension, encode the question and each provided candidate and score candidates; report that task separately.

### 4.4 Leakage prevention
No validation or test video may enter caption generation, synthetic QA generation, SSL adaptation, selector calibration, or threshold fitting. Benchmark training videos may be used, but their role must be named. Pretrained backbones may have unknown web-data overlap; document model provenance and acknowledge that a new manifest cannot certify absence of pretraining contamination.

Partition candidate-generation training videos into three source-disjoint audit groups: calibration fit, calibration development, and independent audit evaluation. A starting allocation is 200, 100, and 100 manually reviewed candidates, expanded if labels are imbalanced or uncertainty is too large. Keep all candidates from one video in the same partition. After measuring calibration, exclude all three audit groups from the large synthetic training pool for the cleanest analysis. They may still belong to the fixed benchmark human-training split, identically across every arm.

## 5 Candidate generation and human audit
### 5.1 Evidence records
For each training video, sample 16 frames and store timestamps. The executable pilot uses a pretrained BLIP image captioner on a small number of sampled frames [14]. These captions support descriptive questions only. They do not establish event order, causality, hidden object identity, or speech content. A video-capable teacher with a documented checkpoint can later provide temporal descriptions, but that extension requires its own audit and cost accounting.

Store each caption with its frame timestamp and a source ID. A sequence-to-sequence question generator receives only the caption and is instructed to produce a short question and answer supported by that text. The packaged pilot uses FLAN-T5-base, fixed decoding, explicit parsing, and rejection logs. Generation is a source of candidates, not a correctness guarantee. Freeze the generator and prompt before main comparisons.

Every candidate record contains uid, video_id, question, answer, type, evidence_text, evidence_times, generator_id, captioner_id, and prompt_version. Keep rejected generations as well as accepted ones so yield and cost can be measured. Restrict the pilot to action or object recognition when the captions support it. Do not use a single-frame caption to manufacture before/after or why questions.

### 5.2 Verification and illustrative cases
A separate evidence-conditioned answer prompt checks whether the generated answer can be recovered from the stored caption. Exact normalized agreement provides a consistency indicator. Using the same language model for generation and verification is cheap but correlated; it is not independent verification. The main human audit must inspect the actual video and question without seeing selector scores.

A drinking example must distinguish visible action from unobservable content. If a clip shows a person raising a transparent glass to their mouth, “What is the person drinking from?” with answer “glass” can be supported. “What liquid is the person drinking?” with answer “water” is unsupported unless the liquid is explicitly identifiable. A plausible world assumption must not become a visual label. These examples are illustrative, not extracted observations from the user's videos.

Other rejection examples include a caption mentioning a bottle while the generated answer says plate; a pronoun that could refer to two people; a question asking for shoe color when shoes are invisible; and counting events from sparse still images. A high-confidence correct answer from the question alone is a shortcut-risk signal, not grounds to declare the question false.

### 5.3 Annotation protocol
Annotators record visual support, answer correctness, ambiguity, question category, and whether temporal order is necessary. Define a valid example as one with visible support, a correct answer, and an unambiguous referent. Mark uncertain examples separately instead of forcing agreement. Have two annotators independently review at least the development and evaluation audit partitions, with adjudication and agreement reporting. If only one reviewer is available, disclose that limitation and retain the original labels.

Annotators receive videos in randomized order and do not see model confidence, selection arm, or generator identity. A third adjudication step resolves disagreements using the same written rubric. Keep auditing labor and example counts in the resource ledger. The final system uses human-audited calibration, so its data curation is not annotation-free even if downstream auxiliary QA labels were generated automatically.

## 6 Evidence calibrated selection
### 6.1 Fixed scoring models
Train one question-only scorer and one multimodal scorer on the benchmark training QA. Freeze both before scoring any synthetic candidate. They are selector instruments, not the downstream student being evaluated. Their supervised training is part of the declared supervision budget. Use the same frozen scores for every synthetic-data arm, and record their checkpoints and temperatures. A training-only calibration fold may temperature-scale their probabilities if needed.

For candidate i, let pF be its proposed-answer probability under the correct video, pQ the probability under the question-only scorer, and pWj the probability under wrong video j. Wrong-video substitutions come from training videos only and must exclude the original source video. Choose J=3 videos from the same broad content cluster where possible. Freeze their IDs. If a substitution also supports the answer, it is a contaminated negative; audit a sample and report the rate.

### 6.2 Selector features
Use a compact feature vector containing pF, pQ, the mean and standard deviation of pWj, the visual-specificity margin, the evidence-conditioned answer-agreement indicator, and question-type indicators. The visual-specificity margin is defined below. A conservative additional feature can measure disagreement between two deterministic frame-sampling grids, provided both cover the full clip; sampling changes can still remove crucial evidence, so this is a diagnostic rather than guaranteed semantic invariance.

EQUATION: u_i = pF_i - max(pQ_i, mean_j pW_ij)                         (1)

A positive u suggests that the correct video contributes information relative to the tested alternatives. It does not prove faithful grounding, identify a causal effect, or guarantee label correctness. Different scorers may have different calibration, motivating an audited combination rather than a universal hand-picked threshold. Temporal-order sensitivity is excluded from the core selector because it would penalize valid static questions. A type-gated temporal feature is an optional extension only after actual temporal candidates exist.

### 6.3 Calibrated validity probability
Fit a small L2-regularized logistic regression on the calibration-fit videos. Its target is audited validity. Select regularization and a threshold on the calibration-development partition; evaluate calibration only once on the independent audit-evaluation partition. Fit standardization on the fit partition only. Report Brier score and reliability bins with counts; expected calibration error alone is unstable in a small sample. Compare to pF alone and an uncalibrated fixed filter.

EQUATION: r_i = sigmoid(b + w^T standardize(x_i))                       (2)

The r score estimates validity under the audit distribution and its annotation policy. It is not a probability of general scientific truth. If no useful held-out calibration improvement appears, remove the calibrated component from the central claim. The study should still report that a simpler confidence filter was adequate rather than hiding the negative result.

### 6.4 Selection and matched controls
First remove exact duplicate question–answer pairs within a video and malformed or out-of-vocabulary records. Keep the same base pool for all conditions. Select the highest-r examples within coarse question-type and training-answer-frequency strata at retention fractions of 25%, 50%, and 100%. At 100%, ordering is irrelevant unless score weighting is separately enabled; there should be no claimed selection gain when all examples are identically used.

For each selected subset, create a random control with the same size in each stratum. Add a stronger control with the same per-video example counts where candidate multiplicity permits. If the constraints cannot be satisfied, coarsen strata using a declared rule or report the imbalance; do not silently substitute an unmatched control. Selected and control pools share the same generator, backbone, and human QA data.

Core experiments use hard selection and ordinary cross-entropy. Reliability-weighted cross-entropy is a separate ablation so selection and weighting are not conflated. Evaluate retention, rejection causes, answer/type distributions, unique-video coverage, and human-audited precision among retained examples.

## 7 Compact VideoQA model and training
### 7.1 Representation and fusion
The main efficient configuration caches 16 frame-level CLIP ViT-B/32 features per video and masked-mean DistilBERT question embeddings. Encoders remain frozen. Project frame features and question features to dimension 256. Add learned temporal positions to frame tokens and apply two temporal Transformer encoder layers with four heads. A projected question query attends to temporal tokens through four-head cross-attention. Concatenate the attended video context and question vector, then apply a two-layer answer classifier.

EQUATION: H_v = TemporalEncoder(W_v F_v + P)                           (3)
EQUATION: h_q = W_q MaskedMean(TextEncoder(q))                          (4)
EQUATION: c = softmax((h_q W_Q)(H_v W_K)^T / sqrt(d_h)) H_v W_V         (5)
EQUATION: p(a | v,q) = softmax(MLP(concat(c,h_q)))                      (6)

The implementation uses library multihead attention, which performs the per-head projections and combination in Equation 5. Learned temporal positions preserve the possibility of order dependence, but do not guarantee that a model uses it. The question-only baseline has its own classifier and is trained without any visual input. A zero-video intervention on the multimodal model is reported separately from that independently trained baseline.

### 7.2 Training stages and objectives
Stage A reproduces a supervised baseline with the new feature cache. Stage B trains separate student copies on a synthetic subset for a fixed number of updates. Stage C fine-tunes each copy on the identical human training QA for a fixed maximum number of updates and the same validation selection rule. Compare against a human-only baseline and a human-only repeated-data control with the same total optimizer-update budget. This distinguishes auxiliary information from simply training longer.

EQUATION: L_human = -(1/B) sum_i log p(a_i | v_i,q_i)                   (7)
EQUATION: L_synth = -(1/B) sum_i log p(a_i | v_i,q_i)                   (8)
EQUATION: L_weighted = -sum_i r_i log p(a_i | v_i,q_i) / sum_i r_i      (9)

Only Equation 8 is used in the core synthetic arm. Equation 9 is optional, with detached r values and a nonzero denominator. Do not train the selector and student together in the first study, because changing candidate rankings would confound the selection comparison.

Initial development settings are AdamW, learning rate 0.0003, weight decay 0.01, dropout 0.1, gradient norm clipping at 1.0, and batch size 64 for cached features. Adjust batch size to observed memory; do not assume 64 is necessary. Select a small learning-rate grid on validation using identical budgets across arms. Log effective batches and update counts, not only epochs. Three seeds are the minimum final replication target; five are preferable when affordable.

### 7.3 Optional curriculum
The first main run has no curriculum. If selection is useful, introduce a fixed easy-to-hard schedule using a frozen difficulty score d_i = -log(max(pF_i, epsilon)) among retained examples. A growing percentile admits 50%, 75%, then 100% of examples over three equal update stages. Compare to random, reverse, and static subset schedules with equal updates. This is inspired by curriculum learning and is not an implementation of CurEvo's adaptive self-evolution.

### 7.4 Optional self supervised adaptation
The completed UCF101 contrastive experiment can be retained as a representation ablation, but a very low contrastive loss does not establish downstream quality. Check duplicate views, train/validation separation, embedding variance, positive/negative similarity, and downstream transfer. For paired views in a 2B-example batch, exclude the anchor itself from the denominator.

EQUATION: L_SSL = -(1/(2B)) sum_i log[ exp(sim(z_i,z_pos(i))/tau) / sum_(k != i) exp(sim(z_i,z_k)/tau) ]   (10)

Use multiple temporal segments per video when extracting adapted R3D features. Globally pooling the whole video before cross-attention yields only one visual token, weakening claims of temporal attention. Pretrained R3D-18 weights are supervised action pretraining; further SSL does not make their full history self-supervised [15]. The packaged CLIP baseline does not execute this optional adaptation.

## 8 Experiments and statistical analysis
### 8.1 Primary comparisons
| ID | Auxiliary stage | Selector or control | Purpose |
| --- | --- | --- | --- |
| B0 | None | Original R3D baseline | Historical continuity |
| B1 | None | Frozen CLIP baseline | Backbone improvement |
| B2 | Repeated human QA | Matched total updates | Extra optimization control |
| S0 | Synthetic QA | Random matched subset | Additional data baseline |
| S1 | Synthetic QA | Full-video confidence | Cheap selector baseline |
| S2 | Synthetic QA | Agreement and fixed thresholds | Uncalibrated baseline |
| S3 | Synthetic QA | Calibrated proposed score | Main proposed method |
| S4 | Synthetic QA | Calibrated without pQ or pW | Feature attribution |
| C1 | Same S3 subset | Fixed curriculum | Secondary ordering study |

Run B1, B2, S0, S1, S2, and S3 first with one development seed. Freeze the recipe, then repeat the essential contrasts across three seeds and both classification datasets. Run selector ablations after the basic effect is known. Keep unsuccessful arms in the report.

### 8.2 Evaluation metrics
Primary accuracy is the fraction of all benchmark questions answered correctly under the stated normalization. Also report macro accuracy by answer and question type, rare-answer accuracy with rarity defined on training data, answer-vocabulary coverage, and accuracy versus synthetic example count. Use the official evaluator whenever available and document any differences from exact normalized matching.

Report correct-video, zero-video, independently trained question-only, wrong-video, single-frame, repeated-frame, and shuffled-frame conditions. Use several fixed wrong-video mappings and several frame permutations. Video swaps must change source identity; do not shuffle questions and call it frame-order shuffling. Single-frame input retains only one original frame and pads it by repetition to keep the model shape consistent.

EQUATION: Gap_Q = Accuracy_full - Accuracy_question_only               (11)
EQUATION: Drop_wrong = Accuracy_full - Accuracy_wrong_video            (12)
EQUATION: Drop_order = Accuracy_full - Accuracy_frame_shuffled         (13)

These are diagnostics. Larger drops are not automatically improvements: a fragile model can fail under harmless shifts, and a selector trained on an intervention may overfit that diagnostic. Interpret them jointly with full accuracy, independent human labels, and direct grounding where available. Order disruption should be analyzed separately on order-sensitive and order-insensitive questions.

For NExT-GQA, add an explicit segment predictor and run the official grounding evaluator [4]. A transparent basic extension predicts start and end positions from question-conditioned frame states, enumerates valid intervals with start no later than end, and maps the chosen interval to timestamps. Train with available official training grounding labels only, declaring the additional supervision. Human temporal labels are necessary to measure temporal IoU. The packaged classification baseline does not implement this extension.

### 8.3 Statistical unit and comparisons
Questions from the same video are dependent. Use paired cluster bootstrap resampling at the video or original-source-video level to estimate confidence intervals on accuracy differences between methods using identical examples. Report mean and standard deviation over training seeds separately; a bootstrap over one seed is not evidence about training randomness. Show individual seed outcomes so an average cannot hide a failed run.

Predeclare S3 versus S0 as the main selector comparison and S3 versus S1 as the main complexity comparison. Use confidence intervals and effect sizes, and apply a multiple-comparison correction if reporting many inferential tests. Avoid picking the best seed or announcing statistical significance from question-level resampling that ignores video grouping.

### 8.4 Result tables to populate after execution
| Method | MSVD test accuracy | MSRVTT test accuracy | Wrong-video accuracy | Train updates |
| --- | --- | --- | --- | --- |
| Human-only B1 | Not run | Not run | Not run | To log |
| Matched-budget B2 | Not run | Not run | Not run | To log |
| Random synthetic S0 | Not run | Not run | Not run | To log |
| Confidence S1 | Not run | Not run | Not run | To log |
| Uncalibrated S2 | Not run | Not run | Not run | To log |
| Calibrated S3 | Not run | Not run | Not run | To log |

| Audit metric | Confidence selector | Calibrated selector |
| --- | --- | --- |
| Independent held-out validity precision | Not run | Not run |
| Brier score | Not run | Not run |
| Retained fraction | To set | To match |
| Unique videos and answer strata | To log | To log |
| Human review minutes | To log | To log |

No target value in this protocol is an achieved result. The historical approximately 25% validation result cannot occupy a final test-result row. Compare a reproduced checkpoint on a fixed evaluation set first.

## 9 Error analysis and interpretation
Divide errors into unsupported synthetic label, ambiguous question, missing evidence due to sampling, visual recognition failure, temporal reasoning failure, vocabulary limitation, and language shortcut. Inspect a fixed random sample plus a separately identified disagreement sample. Keep sampling weights if estimating overall error proportions from stratified samples.

Three outcomes have distinct meanings. If S3 improves accuracy and audit precision after distribution matching, the selector has a defensible benefit. If gains disappear after matching answer frequencies or video exposure, the evidence supports a distribution-selection explanation, not necessarily better label verification. If S3 improves validity precision but harms downstream accuracy, useful diversity may have been removed. Report the trade-off and test retention fractions rather than declaring all filtering beneficial.

If wrong-video sensitivity increases without improved independent grounding or accuracy, do not call the model more faithful. If an intervention feature is removed from the selector but the evaluation effect remains on an independently annotated set, the evidence is stronger. Repeat with a different backbone and, where feasible, a different scoring model to test whether selection is tuned to one teacher.

## 10 Reproducibility and resource accounting
Save split manifests, annotation hashes, answer vocabulary, feature-model IDs and resolved revisions, frame timestamps, candidate prompts, decoding settings, frozen scoring checkpoints, calibration partitions, selector parameters, selected IDs, wrong-video IDs, seeds, optimizer states, predictions, and error labels. The feature metadata must invalidate the cache when model, preprocessing, video, or sampling changes.

Measure peak memory and elapsed time independently for feature extraction, captioning, generation, scoring, audit, calibration, and student training. Cached FP32 CLIP features for 2,000 videos, 16 frames, and dimension 512 occupy approximately 62.5 MiB before metadata and filesystem overhead. This is arithmetic, not a measured runtime or GPU benchmark. Store intermediate features on disk and run heavy encoders sequentially so a 4GB GPU can be used productively without promising a specific throughput.

Use a clean environment rather than modifying the working thesis installation in place. Install a CUDA-compatible PyTorch build using the official installer for the local driver. The companion code uses the Transformers 4.x API range intentionally and records installed versions; it is a starting implementation, not a claim of compatibility with every current release. The environment used to author this package does not include PyTorch or the user's video datasets, so full GPU training has not been run here.

## 11 Completion plan and decision gates
Week 1 recovers the old notebook and checkpoint, validates video mapping and splits, fixes normalization, and reproduces the historical baseline. Deliver the manifest report, vocabulary coverage, and baseline validation predictions. If recovery fails, retain the old number only as history and establish a new reproducible baseline.

Week 2 caches CLIP and text features, trains B1 and B2, and overfits a tiny development subset to check the learning path. Select the backbone using validation before selection experiments. If the new baseline is poor, inspect preprocessing, class balance, label mapping, and attention shapes before generating more data.

Week 3 generates a pilot pool of approximately 2,000–5,000 candidates from training videos, measures valid parsing and vocabulary yield, and conducts the audit. Expand only if independent audit precision and diversity justify it. If a caption generator mostly produces unhelpful or generic text, improve its evidence extraction rather than generating more of the same candidates.

Week 4 fits the selector, runs S0–S3 under fixed budgets, and assesses the primary contrasts. If calibration offers no benefit over confidence, simplify the method or refine the hypothesis with the supervisor before spending on large replications.

Weeks 5 and 6 run essential three-seed experiments, replicate on MSRVTT-QA, and complete the independent evidence audit. Week 7 performs selector ablations, temporal-subset diagnostics, and the optional curriculum. Weeks 8 and 9 finish figures, results, discussion, limitations, released manifests, and supervisor review. These are planning estimates, not guarantees about data access or training duration.

## 12 Publication positioning
A strong MSc thesis can contribute a reproducible method, careful comparative evidence, and an honest account of failure modes. Higher-selectivity publication additionally requires a substantial and generalizable insight. Paper length and a large list of losses do not substitute for it. The likely strongest claim would be that a small calibrated selector improves accuracy per unit of synthetic supervision across datasets and backbones, while distribution-matched controls reveal when simpler filters are sufficient.

Use stage gates rather than promise a journal ranking: first show a reliable gain over matched random selection; then show an advantage over confidence-only selection; then show transfer and independent evidence quality; finally compare to recent competitive methods under comparable supervision. A broad claim of state of the art requires current literature and official evaluation at submission time. Venue fit, acceptance probability, and a target accuracy cannot be established before these results exist.

The paper should present the central method and decisive experiments concisely, with the full audit rubric, generation prompts, extra tables, and implementation details in supplementary material. The present expanded document is a research blueprint and manuscript scaffold; its results, discussion of observed outcomes, and final abstract must be revised after execution.

## 13 Limitations and responsible claims
Caption omissions and hallucinations constrain candidate quality. Correlated generator and verifier errors can survive consistency checks. A compact supervised scorer may reject precisely the difficult examples that a better student could learn from. Human calibration improves accountability but introduces labor, annotation policy, and distribution dependence. Wrong videos can accidentally contain the same evidence, and feature masking can create unrealistic inputs. Sparse frames can miss brief actions. Closed-vocabulary evaluation limits open-ended language claims.

The method adds no guarantee of causal reasoning or faithful explanations. Attention visualizations alone are not evidence localization ground truth. Improvements under our tested interventions should be described as those tested effects. Data licenses, pretrained model terms, redistribution limits, and identifiable content should be reviewed before releasing videos or captions; distribute IDs, scripts, and permissible derived annotations when raw redistribution is restricted.

## 14 Conclusion
This study specifies an evidence-calibrated approach to selecting synthetic VideoQA supervision and a controlled protocol for determining whether it is useful. Its main methodological choice is to isolate data selection from backbone scale, training time, candidate generation, and distribution changes. The completed thesis must establish the resulting accuracy and evidence-quality trade-offs experimentally. Positive, null, and negative findings all remain possible until the protocol is executed.

## References
[1] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just Ask: Learning to Answer Questions from Millions of Narrated Videos. ICCV, 2021. https://arxiv.org/abs/2012.00451

[2] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-Shot Video Question Answering via Frozen Bidirectional Language Models. NeurIPS, 2022. https://arxiv.org/abs/2206.08155

[3] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions. CVPR, 2021. https://arxiv.org/abs/2105.08276

[4] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can I Trust Your Answer? Visually Grounded Video Question Answering. CVPR, 2024. https://arxiv.org/abs/2309.01327 ; https://github.com/doc-doc/NExT-GQA

[5] Shangzhe Di and Weidi Xie. Grounded Question-Answering in Long Egocentric Videos. CVPR, 2024. https://arxiv.org/abs/2312.06505

[6] Shyamal Buch, Cristobal Eyzaguirre, Adrien Gaidon, Jiajun Wu, Li Fei-Fei, and Juan Carlos Niebles. Revisiting the Video in Video-Language Understanding. CVPR, 2022. https://arxiv.org/abs/2206.01720

[7] Guiyi Zeng, Junqing Yu, Yi-Ping Phoebe Chen, Chen Xu, Wei Yang, and Zikai Song. CurEvo: Curriculum-Guided Self-Evolution for Video Understanding. arXiv:2604.26707, 2026. https://arxiv.org/abs/2604.26707

[8] Shiqi Huang, Ziyue Wang, Zhongrong Zuo, Han Qiu, Qi She, and Bihan Wen. EvoVid: Temporal-Centric Self-Evolution for Video Large Language Models. arXiv:2605.21931, 2026. https://arxiv.org/abs/2605.21931

[9] Ruixu Zhang, Deyi Ji, Lanyun Zhu, Xuanyi Liu, Yuxin Meng, Ruihang Chu, and Yujiu Yang. Video-Zero: Self-Evolution Video Understanding. arXiv:2605.14733, 2026. https://arxiv.org/abs/2605.14733

[10] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. NeurIPS, 2022. https://arxiv.org/abs/2203.12602

[11] Alec Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021. https://arxiv.org/abs/2103.00020

[12] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108, 2019. https://arxiv.org/abs/1910.01108

[13] Dejing Xu et al. Video Question Answering via Gradually Refined Attention over Appearance and Motion. ACM Multimedia, 2017. Dataset and mapping source: https://github.com/xudejing/video-question-answering

[14] Hugging Face. BLIP model documentation. https://huggingface.co/docs/transformers/en/model_doc/blip ; CLIP documentation: https://huggingface.co/docs/transformers/en/model_doc/clip ; DistilBERT documentation: https://huggingface.co/docs/transformers/en/model_doc/distilbert

[15] Torchvision. R3D-18 model documentation. https://pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html . Verify the exact weights used in the recovered notebook; this documentation was not retrievable through the research browser in this session.

## Appendix A Minimum implementation sequence
1. Create the canonical data manifest and training-only answer vocabulary.
2. Extract and cache visual and question features, recording checkpoint revisions.
3. Train the question-only and full-video baseline scorers; freeze their selected checkpoints.
4. Generate captions and candidate QA from permitted training videos only.
5. Cache candidate question features and score pF, pQ, and fixed wrong-video substitutions.
6. Audit source-disjoint calibration and evaluation subsets against actual videos.
7. Fit the selector on calibration-fit, choose settings on calibration-development, and evaluate on the held-out audit.
8. Exclude audit-video groups from the synthetic pool; build selected and matched-random subsets.
9. Train student arms with fixed synthetic and human update budgets.
10. Export predictions, cluster-bootstrap main differences, audit evidence, and fill result tables.

## Appendix B What the accompanying code does
The package provides MSVD manifest creation, feature caching, a temporal cross-attention classifier, a separate question-only classifier, fixed-update training with checkpoint resume, full/zero/wrong/order/single-frame evaluation, caption and candidate generation, frozen-scorer candidate scoring, logistic selector fitting, stratified random controls, and paired video-cluster bootstrap. A notebook calls these scripts in order for a Windows Jupyter workflow. Dataset files and pretrained weights are not bundled.

The package does not implement the original CurEvo or EvoVid training algorithms, reinforcement learning, a temporal-grounding head, automatic semantic deduplication, a VideoMAE pretraining run, the optional curriculum, or cross-dataset adapters for NExT-QA. Its executable core establishes the classification and selection study. These extensions must not appear as completed implementation claims. Syntax and lightweight control logic can be checked here; GPU training and model-download compatibility must be verified in the user's environment.
