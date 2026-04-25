"""Phase B — GRPO training driver.

Thin wrapper around TRL's GRPOTrainer. The heavy lifting (tokenization,
LoRA config, reward function) lives in the Colab notebook so judges can
reproduce it with zero local setup. This file is the CLI shortcut.

Expected env vars / args:
  --model          base model HF id
  --rollouts       path to SFT-adapted checkpoint directory
  --out            output directory for the GRPO-adapted LoRA
  --group-size     rollouts per prompt (default 8)
  --steps          optimizer steps (default 200)

Dependencies (installed in the Colab):
  unsloth, trl >= 0.8, peft, accelerate, bitsandbytes
"""

import argparse
import json
import os
from typing import List, Dict, Any


def run(args):
    try:
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ImportError:
        print("[WARN] unsloth/trl not available — see notebooks/PROTOCOL_ARENA_Colab.ipynb")
        return

    from ..server.arena_env import ProtocolArenaEnvironment
    from ..models import OrchestratorAction
    from ..tasks import ALL_TASKS

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048,
        dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # Run each completion as a single action against a fresh env snapshot.
        rewards = []
        for p, c in zip(prompts, completions):
            try:
                decision = json.loads(c)
            except json.JSONDecodeError:
                rewards.append(-0.5); continue
            tid = kwargs.get("task_id", list(ALL_TASKS.keys())[0])
            env = ProtocolArenaEnvironment()
            env.reset(task_id=tid, seed=0)
            try:
                act = OrchestratorAction(**{k: v for k, v in decision.items()
                                           if k in {"kind", "rationale", "mcp_call",
                                                    "a2a_call", "dag_delta", "kg_op", "final"}})
                obs = env.step(act)
                rewards.append(obs.reward)
            except Exception:
                rewards.append(-0.2)
        return rewards

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=5e-6,
        num_generations=args.group_size,
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        beta=0.04,
    )

    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, reward_funcs=[reward_fn], args=cfg,
        train_dataset=_load_prompts(),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)


def _load_prompts():
    from datasets import Dataset
    from ..tasks import ALL_TASKS
    rows = [{"prompt": t["spec"], "task_id": tid} for tid, t in ALL_TASKS.items()]
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="outputs/grpo_lora")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
