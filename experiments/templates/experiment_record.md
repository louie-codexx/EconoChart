# Experiment: `<experiment_id>`

## 1. Identity

| Field | Value |
|---|---|
| Date/time | `<UTC and Asia/Shanghai>` |
| Owner | `<name>` |
| Status | `planned / running / failed / completed / rejected / stopped` |
| Git commit | `<sha>` |
| Git dirty | `false preferred; otherwise explain` |
| Config | `<path>` |
| Parent/control run | `<experiment_id>` |

## 2. Question and hypothesis

- Observed capability gap:
- Falsifiable hypothesis:
- Why this experiment is the simplest useful test:
- Alternative not chosen and why:
- Primary metric(s):
- Guardrail metric(s):
- Accept / reject / stop condition written before training:

## 3. Single changed factor

| Item | Control | This run | Rationale |
|---|---:|---:|---|
| `<e.g. LoRA rank>` | | | |

List every other config difference. If more than one primary factor changed, explain why attribution is still possible.

## 4. Environment

| Item | Value |
|---|---|
| AutoDL image/instance | |
| GPU / count / VRAM | |
| Driver / CUDA runtime | |
| Python | |
| PyTorch / Torch CUDA | |
| Transformers / TRL / PEFT | |
| bitsandbytes / Datasets | |
| BF16 supported | |
| Optional acceleration | `none / flash-attn / other` |

Preflight report path/hash:

## 5. Data lineage

| Item | Value |
|---|---|
| EconoChart version | |
| Builder config and seed | |
| Manifest SHA-256 | |
| Train/val/test rows | |
| Train source mixture | |
| Test ID/hash frozen | |
| Public benchmark role | |
| Validator result | |

## 6. Model and optimization

| Parameter | Value | Why chosen |
|---|---:|---|
| Base model | | |
| Tuning method | | |
| Quantization | | |
| Trainable modules | | |
| Trainable params / % | | |
| LoRA r / alpha / dropout | | |
| min/max pixels | | |
| Physical / effective batch | | |
| LR / scheduler / warmup | | |
| Epochs / max steps | | |
| Optimizer / weight decay | | |
| Gradient checkpointing | | |
| GRPO generations / beta / epsilon | | |
| Reward names / weights | | |

Resolved config path/hash:

## 7. Runtime evidence

| Metric | Value |
|---|---:|
| Start/end/elapsed | |
| Completed steps/epochs | |
| Peak allocated/reserved VRAM | |
| Samples or tokens / second | |
| Final train/eval loss | |
| Best checkpoint and criterion | |
| Adapter/checkpoint size | |
| Estimated compute/cost | |

Curve summary (do not paste only the final point):

## 8. Quantitative results

| Model | Internal overall | Numeric | Trend | Format | Evidence | Risk | ChartQA | ChartQAPro | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | | | | | | | | | |
| Candidate | | | | | | | | | |
| Delta | | | | | | | | | |
| Paired 95% CI | | | | | | | | | |

Numeric recall / precision（用于区分漏答与虚构数字）：

Most improved slices:

| Slice | Rows | Metric | Delta | 95% CI | Interpretation |
|---|---:|---|---:|---|---|
| | | | | | |

Regressed/unchanged slices:

| Slice | Rows | Metric | Delta | 95% CI | Interpretation |
|---|---:|---|---:|---|---|
| | | | | | |

## 9. GRPO reward diagnostics (if applicable)

| Reward | Mean | Std | Zero-variance group rate | Train→test relationship |
|---|---:|---:|---:|---|
| Numeric | | | | |
| Trend | | | | |
| Format | | | | |
| Evidence | | | | |
| Risk | | | | |
| Length | | | | |

Reward hacking checks: repeated sections, copied prompt, unsupported numbers, verbosity, collapsed generations.

## 10. Qualitative evidence

### Representative improvement

- Sample ID:
- Control output:
- Candidate output:
- Ground truth / chart evidence:
- Why this is a real capability improvement:

### Representative regression

- Sample ID:
- Control output:
- Candidate output:
- Error category:
- Likely cause:

### Hard unresolved case

- Sample ID:
- Failure type: `visual/OCR / arithmetic / trend / unit / hallucination / risk / format / other`
- Smallest plausible next intervention:

## 11. Failures and deviations

- OOM/NaN/runtime error:
- Attempt number and exact change:
- Deviation from preregistered plan:
- Does the deviation invalidate comparison?

## 12. Decision

- Hypothesis: `supported / not supported / inconclusive`.
- Evidence in one paragraph:
- Trade-off accepted:
- Decision: `promote / rerun / change data / change parameter / stop`.
- Next experiment and why it can change the decision:

## 13. Interview-ready summary

- One-sentence result with real numbers:
- Most important design choice and evidence:
- Most important failure and what was learned:
- Limitation that must be stated proactively:

## 14. Reproduction pointers

- Command:
- Run manifest:
- Metrics and paired comparison:
- Curves/figures:
- Adapter publication location (not Git):
