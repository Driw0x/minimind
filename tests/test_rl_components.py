from __future__ import annotations

from types import SimpleNamespace

import torch

from model.model_minimind import MiniMindForCausalLM
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps


class DummyRewardModel:
    """Tiny stand-in for InternLM2: no download and no 1.8B model in CI."""

    def get_score(self, messages, answer):
        return 0.25


def test_torch_rollout_engine_on_directml(tokenizer, tiny_config, directml_device):
    model = MiniMindForCausalLM(tiny_config).to(directml_device).eval()
    engine = create_rollout_engine(
        engine_type="torch",
        policy_model=model,
        tokenizer=tokenizer,
        device=directml_device,
    )

    enc = tokenizer(
        "Hello",
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(directml_device)
    attention_mask = enc["attention_mask"].to(directml_device)

    result = engine.rollout(
        prompt_ids=input_ids,
        attention_mask=attention_mask,
        num_generations=2,
        max_new_tokens=4,
        temperature=0.8,
    )

    assert result.completion_ids.shape[0] == 2
    assert result.per_token_logps.shape == result.completion_ids.shape
    assert len(result.completions) == 2


def test_compute_per_token_logps_on_directml(tokenizer, tiny_config, directml_device):
    model = MiniMindForCausalLM(tiny_config).to(directml_device).eval()
    ids = tokenizer("DirectML smoke test", return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids = ids.to(directml_device)
    mask = torch.ones_like(ids)

    n_keep = min(2, max(ids.size(1) - 1, 1))
    values = compute_per_token_logps(model, ids, n_keep=n_keep, attention_mask=mask)

    assert values.shape == (1, n_keep)
    assert torch.isfinite(values).all().item()


def test_grpo_reward_function_with_dummy_reward(monkeypatch, directml_device):
    import trainer.train_grpo as grpo

    grpo.args = SimpleNamespace(
        device=directml_device,
        num_generations=2,
    )
    prompts = [
        "<|im_start|>user\nSay hello<|im_end|>"
    ]
    responses = [
        "Hello, nice to meet you!",
        "Hello there!",
    ]

    rewards = grpo.calculate_rewards(prompts, responses, DummyRewardModel())

    assert rewards.shape == (2,)
    assert torch.isfinite(rewards).all().item()


def test_agent_reward_function_without_external_rm(directml_device):
    import trainer.train_agent as agent

    prompts = ["<|im_start|>user\nWhat is 2+2?<|im_end|>"]
    completions = ["4", "The answer is 4."]
    gt_batch = [["4"]]
    tools_batch = [[]]

    rewards = agent.calculate_rewards(
        prompts=prompts,
        completions=completions,
        gt_batch=gt_batch,
        tools_batch=tools_batch,
        num_gen=2,
        reward_model=None,
        device=directml_device,
        turn_outputs_batch=[["4"], ["The answer is 4."]],
        unfinished_batch=[False, False],
    )

    assert rewards.shape == (2,)
    assert torch.isfinite(rewards).all().item()
