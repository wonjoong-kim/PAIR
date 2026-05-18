# PAIR: Prefix-Aware Internal Reward Model for Multi-Turn Agent Optimization

Code release for the paper *PAIR: Prefix-Aware Internal Reward Model
for Multi-Turn Agent Optimization*. PAIR is a turn-level reward for
multi-turn agentic LLMs that is dense, runs at probe-level cost, and
requires no external LLM judge, no ground-truth at inference, and no
full-trajectory rollouts.

PAIR is a two-stage probe over the agent's own internal states:

1. a frozen logistic regression on hidden states produces a
   belief-consistency score `s_bc`, and
2. a logistic regression on multi-layer attention statistics + `s_bc`
   corrects it toward grounded correctness, yielding `s_final`.

`s_final ∈ (0, 1)` is used both as an offline contamination detector and
as a dense step-level reward for GRPO fine-tuning of multi-turn agents.

The repository is organised as two cleanly separable stages:

```
official_github/
├── probing/         # Stage 1: feature extraction + linear probing
│   ├── extract_features.py
│   ├── extract_multi_layer_attn.py
│   ├── train_pair.py
│   ├── eval_pair.py
│   └── scripts/{extract_gta,extract_toolbench,train_all}.sh
├── grpo/            # Stage 2: GRPO fine-tuning with the PAIR reward
│   ├── envs/{gta,toolbench}_env.py
│   ├── rewards/{pair,outcome}.py
│   ├── training/{policy,grpo_loop}.py
│   └── scripts/{run_single,evaluate}.py, run_{gta,toolbench}_pair.sh
├── data/            # raw splits + extracted features + trained probes
├── requirements.txt
└── README.md
```

---

## 1. Method overview

PAIR is a two-stage logistic regression probe over per-turn LM internals:

| Stage | Input | Output |
|-------|-------|--------|
| 1 | `last_token` hidden state of the assistant turn | `s_bc = σ(w₁ᵀ h + b₁)` |
| 2 | `[multi_layer_attn ; s_bc]` | `s_final = σ(w₂ᵀ x + b₂)` |

Both probes are trained offline with L2 regularization (`C = 0.01`).
`s_final ∈ (0, 1)` is the per-turn reliability score, used both for
offline contamination detection and as a dense GRPO reward.

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

# (b) train PAIR (Stage 1 + Stage 2 LR)
python -m probing.train_pair --models qwen7b --datasets gta toolbench

# (c) evaluate probes on matched_*_test splits
python -m probing.eval_pair --model qwen7b --dataset gta --all
```

Probes are saved to:

```
data/models/methods/PAIR/{model}/{dataset}/pair_{train_mode}.pkl
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
# Train (paper's headline setting: PAIR + momentum, defaults match Table 7)
python -m grpo.scripts.run_single \
    --policy qwen7b --env gta --reward pair_momentum \
    --output_dir runs/qwen7b_gta_pair_momentum

# Or via the wrapper
bash grpo/scripts/run_gta_pair.sh         # POLICY=qwen7b REWARD=pair_momentum (defaults)
bash grpo/scripts/run_toolbench_pair.sh   # POLICY=qwen7b REWARD=pair_momentum (defaults)
```

Reward options:

| `--reward` | Description |
|------------|-------------|
| `pair`          | Vanilla PAIR — `s̃_final` from Stage 2 LR with temperature clip (paper §4.2 "PAIR w/o momentum"). |
| `pair_momentum` | **Headline.** Adds a logit-space momentum bonus (`α · (s̃_final,t − mean(s̃_<t))`, α = 5; paper Eq. 7). |
| `outcome`       | Sparse outcome reward at the final turn (baseline). |

Probes are loaded from
`data/models/methods/PAIR/{model}/{env}/pair_{train_mode}.pkl`,
so Stage 1 must finish before Stage 2 starts.

The trainer is a minimal in-process GRPO loop: per-prompt group-relative
advantage normalization plus a REINFORCE-style LoRA update with a
reference-KL anchor (`kl_beta = 0.01`). Metrics stream to
`runs/<name>/metrics.jsonl`; LoRA checkpoints save every `--save_every`
steps (default 50; the paper recommends selecting the best checkpoint
across the 50-step grid rather than always using `ckpt_final`).

### Evaluation

```bash
# Single checkpoint
python -m grpo.scripts.evaluate \
    --policy qwen7b --env gta \
    --ckpt runs/qwen7b_gta_pair/ckpt_final \
    --output runs/qwen7b_gta_pair/eval.json

# Sweep every run under runs/<policy>_<env>_*/
python -m grpo.scripts.evaluate --policy qwen7b --env gta --eval_all

# ToolBench: use the partial-credit LLM judge (paper Appendix H).
# Binary scoring is ~0% on long-form gold answers; partial credit
# resolves the population into a usable 0.34–0.41 range.
python -m grpo.scripts.evaluate --policy qwen7b --env toolbench \
    --ckpt runs/qwen7b_toolbench_pair_momentum/ckpt_final --llm-judge \
    --output runs/qwen7b_toolbench_pair_momentum/eval.json
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
  the policy starts saturating the probe (paper Eq. 5) — see
  `grpo/rewards/pair.py`.
* Paper Table 7 defaults are now wired into `run_single.py` and
  `GRPOConfig` (α = 5, β = 0.01, lr = 3e-7, batch = 1, group = 4,
  max_new_tokens = 1024, save_every = 50). Override at the CLI as
  needed.
* The paper trains on `train_mode = "mixed"` (matched_clean_train +
  matched_contaminated_train). `clean_only` is included for
  ablations.
* `GPT_PARSE_PRIMARY=1` / `GPT_PARSE_FALLBACK=1` route raw model outputs
  through `gpt-4o-mini` for robust Thought/Action/Final Answer parsing
  (needs `OPENAI_API_KEY`).
* `TB_ARGS_FUZZY=1` enables same-tool fuzzy matching in the ToolBench
  replay simulator — helps rollouts keep moving when the policy's args
  drift slightly from the recorded cache key.

---

## Citation

Citation will be added after the review process.
