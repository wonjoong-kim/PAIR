# PAIR — Probing AI for Reliability

Official implementation of **PAIR**, a turn-level reliability signal for
multi-turn agentic LLMs. PAIR couples (1) a frozen linear probe on
hidden-state activations with (2) an attention-based correction probe to
produce a per-turn `s_final ∈ (0, 1)` score, used both as an offline
contamination detector and as a dense reward signal for GRPO fine-tuning.

The repository is organised as two cleanly separable stages:

```
official_github/
├── probing/         # Stage 1: feature extraction + linear probing
│   ├── extract_features.py
│   ├── extract_multi_layer_attn.py
│   ├── train_pair.py
│   ├── train_pair_new.py
│   ├── eval_pair.py
│   └── scripts/{extract_gta,extract_toolbench,train_all}.sh
├── grpo/            # Stage 2: GRPO fine-tuning with the PAIR reward
│   ├── envs/{gta,toolbench}_env.py
│   ├── rewards/{pair,pair_new,outcome}.py
│   ├── training/{policy,grpo_loop}.py
│   └── scripts/{run_single,evaluate}.py, run_{gta,toolbench}_pair.sh
├── data/            # raw splits + extracted features + trained probes
├── requirements.txt
└── README.md
```

---

## 1. Method overview

**PAIR (canonical)** — two-stage logistic regression:

| Stage | Input | Output |
|-------|-------|--------|
| 1 | `last_token` hidden state of the assistant turn | `s_base = σ(w₁ᵀ h + b₁)` |
| 2 | `[multi_layer_attn ; s_base]` | `s_final = σ(w₂ᵀ x + b₂)` |

Both probes are trained offline with L2 regularization (`C = 0.01`).

**PAIR-NEW** — logit-space residual variant of Stage 2:

```
z_base       = w₁ᵀ h + b₁
delta_logit  = w₂ᵀ [a ; s_base] + b₂
s_final      = σ(z_base + delta_logit)
```

This keeps the single sigmoid and guarantees `s_final = s_base` when the
correction head produces zero delta — useful for clean-prefix turns
where Stage 1 should already be confident.

---

## 2. Installation

```bash
# Python 3.10+ recommended.
pip install -r requirements.txt
```

Set `PAIR_ROOT` to the project root if you keep features/models on a
shared volume:

```bash
export PAIR_ROOT=/path/to/official_github
```

By default we resolve policy models from HuggingFace:

| Alias | Default ID |
|-------|------------|
| `llama8b` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `qwen7b`  | `Qwen/Qwen2.5-7B-Instruct` |
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` |

Override with `PAIR_MODEL_QWEN7B=/local/path` if you'd rather use a
checkpoint already on disk.

---

## 3. Data

The eight dialog splits per dataset are checked into this repo under
`data/gta/` and `data/toolbench/` — clone the repo and you're ready to
run.

```
data/
├── gta/{clean,contaminated}_{train,test}.json            # GRPO rollout prompts + probe training
├── gta/matched_{clean,contaminated}_{train,test}.json    # flat per-turn records used by the probes
└── toolbench/<same eight files>
```

* `clean_*` / `contaminated_*` are episode-level JSON objects keyed by
  episode ID. They drive GRPO rollouts (`grpo/envs/*_env.py`) and
  provide the source dialogs for offline feature extraction.
* `matched_*` are flat per-turn records (`{episode_id, turn_idx,
  dialogs, label}`) used directly by the PAIR probes.

See [`data/README.md`](data/README.md) for the full schema and a table
mapping each script to the splits it reads.

Heavy artifacts (extracted features, trained probes, GRPO checkpoints)
are produced by the pipeline and intentionally gitignored — they live
under `data/features/`, `data/models/`, and `runs/` after you run the
scripts.

---

## 4. API keys (OpenAI)

A few code paths optionally call OpenAI. **The repo never embeds a key**
— each path reads from the `OPENAI_API_KEY` environment variable and
falls back to an error message when the key is missing:

```bash
export OPENAI_API_KEY="sk-..."
```

| Where | When it runs | Required for |
|-------|--------------|--------------|
| `grpo/envs/gta_env.py` — `MathOCR` GPT-4o vision fallback | Policy calls `MathOCR` on an image that isn't in the GT cache | GTA rollouts that touch math-OCR |
| `grpo/envs/gta_env.py` — `GoogleSearch` GPT-4o-mini fallback | Policy calls `GoogleSearch` on an args combo that isn't in the GT cache | GTA rollouts that touch search |
| `grpo/envs/_parsing.py` — GPT-based output parser | Only when `GPT_PARSE_PRIMARY=1` or `GPT_PARSE_FALLBACK=1` | Optional; regex parser is the default |
| `grpo/scripts/evaluate.py` — ToolBench LLM-as-judge | Only with `--llm-judge` | Recommended for ToolBench eval; string match scores ~0% on long-form answers |

Probe training, PAIR-reward GRPO updates, GTA rollouts that stay inside
the cached tool calls, and PAIR probe evaluation **do not require an
API key**.

---

## 5. Stage 1 — train PAIR probes

```bash
# (a) extract hidden + attention features for the splits that exist
python -m probing.extract_features        --model qwen7b --dataset gta
python -m probing.extract_multi_layer_attn --model qwen7b --dataset gta

# (b) train PAIR (Stage 1 + Stage 2 LR) and PAIR-NEW (logit residual)
python -m probing.train_pair      --models qwen7b --datasets gta toolbench
python -m probing.train_pair_new  --model  qwen7b --datasets gta toolbench

# (c) evaluate probes on matched_*_test splits
python -m probing.eval_pair --model qwen7b --dataset gta --all
```

Probes are saved to:

```
data/models/methods/PAIR/{model}/{dataset}/pair_{train_mode}.pkl
data/models/methods/PAIR_NEW/{model}/{dataset}/pair_new_{train_mode}.pkl
```

`train_mode ∈ {clean_only, mixed}` controls which `matched_*_train`
splits are concatenated for the probe-training set.

Reported metrics: AUROC, AUPRC, ECE (15 bins), Brier score, within-episode
pairwise ranking accuracy.

### Convenience scripts

```bash
bash probing/scripts/extract_gta.sh
bash probing/scripts/extract_toolbench.sh
bash probing/scripts/train_all.sh
```

---

## 6. Stage 2 — GRPO fine-tuning with PAIR reward

```bash
# Train (PAIR / PAIR-NEW / repair / momentum variants)
python -m grpo.scripts.run_single \
    --policy qwen7b --env gta --reward pair \
    --steps 500 --batch_size 8 --group_size 4 \
    --output_dir runs/qwen7b_gta_pair

# Or via the wrapper
bash grpo/scripts/run_gta_pair.sh         # POLICY=qwen7b REWARD=pair (defaults)
bash grpo/scripts/run_toolbench_pair.sh   # POLICY=qwen7b REWARD=pair (defaults)
```

Reward options:

| `--reward` | Description |
|------------|-------------|
| `pair`          | Canonical PAIR (`s_final` from Stage 2 LR). |
| `pair_new`      | Logit-residual variant. |
| `pair_repair`   | `pair` + logit-space repair bonus (`α · max(0, Δ_t)·(1−s_{t−1})`). |
| `pair_momentum` | `pair` + logit-space cumulative momentum (`α · (s_t − mean(s_{<t}))`). |
| `outcome`       | Sparse outcome reward at the final turn (baseline). |

Probes are loaded from
`data/models/methods/PAIR{,_NEW}/{model}/{env}/pair{,_new}_{train_mode}.pkl`,
so Stage 1 must finish before Stage 2 starts.

The trainer is a minimal custom GRPO loop (REINFORCE-style update with
group-relative advantages, optional reference-KL penalty). Metrics are
streamed to `runs/<name>/metrics.jsonl`; LoRA checkpoints are saved every
`--save_every` steps.

### Evaluation

```bash
# Single checkpoint
python -m grpo.scripts.evaluate \
    --policy qwen7b --env gta \
    --ckpt runs/qwen7b_gta_pair/ckpt_final \
    --output runs/qwen7b_gta_pair/eval.json

# Sweep every run under runs/<policy>_<env>_*/
python -m grpo.scripts.evaluate --policy qwen7b --env gta --eval_all

# ToolBench: enable LLM-as-judge (string matching ≈ 0% on long-form answers)
python -m grpo.scripts.evaluate --policy qwen7b --env toolbench \
    --ckpt runs/qwen7b_toolbench_pair/ckpt_final --llm-judge \
    --output runs/qwen7b_toolbench_pair/eval.json
```

The report is a JSON dump containing per-episode scores plus
`mean_score`, `success_rate_at_0.5`, and `n_skipped_image_compare`
(image-compare GTA episodes are skipped from aggregates by convention).

---

## 7. Notes and reproducibility

* All hard-coded paths in the original research repo have been replaced
  with `PAIR_ROOT`-relative paths so the code runs on any machine.
* PEFT 0.11 is pinned because PEFT 0.19 references
  `torch.float8_e8m0fnu` (introduced in torch ≥ 2.4); pin torch < 2.5 to
  match.
* The PAIR probes' `_temp_clip` (T = 2, ε = 0.05) softens frozen-probe
  outputs to keep the GRPO group-relative advantage signal alive when
  the policy starts saturating the probe — see comments in
  `grpo/rewards/pair.py`.
* `GPT_PARSE_PRIMARY=1` / `GPT_PARSE_FALLBACK=1` route raw model outputs
  through `gpt-4o-mini` for robust Thought/Action/Final Answer parsing
  (needs `OPENAI_API_KEY`).
* `TB_ARGS_FUZZY=1` enables same-tool fuzzy matching in the ToolBench
  replay simulator — helps rollouts keep moving when the policy's args
  drift slightly from the recorded cache key.

---

## Citation

If you use this code, please cite the PAIR paper (citation block to be
added).
