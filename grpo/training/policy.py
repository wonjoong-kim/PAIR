"""LoRA policy wrapper with hidden-state / attention capture.

Wraps a HuggingFace causal LM with PEFT LoRA adapters. Exposes a
`chat(system_prompt, user_prompt, capture_internal=...)` method that:

  1. Builds a chat-formatted prompt via the tokenizer's chat template.
  2. Generates a response with `model.generate()`.
  3. Optionally re-runs a forward pass over the full sequence with
     `output_hidden_states / output_attentions` to capture internal
     state for the PAIR reward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as Fnn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..paths import model_path
from ..rewards.base import PolicyOutput

logger = logging.getLogger("policy")


DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class PolicyConfig:
    name: str                     # llama8b / qwen7b / mistral7b (or explicit HF path)
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_context_len: int = 4096
    dtype: str = "bfloat16"


class LoRAPolicy:
    """HF causal LM + LoRA adapter with multi-turn chat and internal capture."""

    def __init__(self, cfg: PolicyConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device

        path = model_path(cfg.name)
        logger.info(f"Loading tokenizer for {cfg.name} from {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[cfg.dtype]

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpus >= 2:
            max_mem = {
                i: f"{int(torch.cuda.get_device_properties(i).total_memory / 1e9) - 8}GiB"
                for i in range(n_gpus)
            }
            base = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                device_map="auto",
                max_memory=max_mem,
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to("cuda" if torch.cuda.is_available() else "cpu")

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=DEFAULT_LORA_TARGETS,
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.print_trainable_parameters()
        self.model.eval()

    def chat(self, system_prompt: str, user_prompt: str,
             capture_internal: bool = False) -> PolicyOutput:
        """Generate a response, optionally capturing hidden states/attentions."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = f"{system_prompt}\n{user_prompt}"

        enc = self.tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=self.cfg.max_context_len,
        )
        input_ids = enc.input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            gen = self.model.generate(
                input_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=max(1e-4, self.cfg.temperature),
                top_p=self.cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        full_ids = gen[0]
        gen_ids = full_ids[prompt_len:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        po = PolicyOutput(
            text=gen_text,
            turn_start=prompt_len,
            turn_end=full_ids.shape[0],
            num_generated_tokens=int(gen_ids.shape[0]),
            full_token_ids=full_ids.detach().cpu().numpy().astype(np.int64),
        )

        if capture_internal:
            hs, att, token_lp = self._capture_internal(full_ids, prompt_len)
            po.hidden_states = hs
            po.attentions = att
            po.token_logprobs = token_lp
        return po

    @torch.no_grad()
    def _capture_internal(self, full_ids: torch.Tensor, prompt_len: int):
        if full_ids.shape[0] > self.cfg.max_context_len:
            full_ids = full_ids[-self.cfg.max_context_len:]

        out = self.model(
            full_ids.unsqueeze(0).to(self.device),
            output_hidden_states=True,
            output_attentions=True,
        )
        hs = [h[0].float().cpu().numpy() for h in out.hidden_states]
        att = [a[0].float().cpu().numpy() for a in out.attentions]

        logits = out.logits[0]
        log_probs = Fnn.log_softmax(logits.float(), dim=-1)
        if full_ids.shape[0] > prompt_len:
            gen_lps = log_probs[prompt_len - 1: -1].gather(
                1, full_ids[prompt_len:].unsqueeze(-1)
            ).squeeze(-1).cpu().numpy()
        else:
            gen_lps = np.zeros(0, dtype=np.float32)
        return hs, att, gen_lps

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    def save_lora(self, path: str) -> None:
        self.model.save_pretrained(path)
        logger.info(f"Saved LoRA to {path}")

    def load_lora(self, path: str) -> None:
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model.base_model, path)
        self.model.to(self.device)
