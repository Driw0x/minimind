[English](directml_audit.md) | [Français](directml_audit.fr.md) | [简体中文](directml_audit.zh-CN.md)

[项目 README](../../README.zh-CN.md)

<a id="directml-compatibility-audit"></a>
# DirectML 兼容性审计

> **历史 DirectML 归档 — 更新日期：2026-09-24**
>
> DirectML 已不再作为 MiniMind 模型训练的主后端。M5 最终验证发现了一条特定于测试的 DirectML FP16 路径的数值偏差。历史结果仍被保留以供可重现性和工程参考；ROCm 开发情况另作记录。

由 `scripts/audit_directml.py` 自动生成。

> **历史审计备注**：训练器精度处理、受限的 `--max_steps` 执行以及共享的 DirectML FP16 工具在本快照之后已发生变化。本报告被保留以记录 DirectML 相关工作，但 DirectML 已不再作为主训练后端。此静态 M2 时代审计早于后来的 M5 数值诊断。它被独立于运行时基准测试，因为它记录的是代码层面的兼容性发现，而非实际运行行为。

> 本报告识别出潜在的 CUDA 特定代码。每次出现都必须在修改前手动审查。

<a id="summary"></a>
## 摘要

总发现项：**86**

  类别             数量
  ------------------ -------
  AMP                     29 CUDA 参考          15 CUDA 工具             1 设备放置         2 分布式             39

<a id="findings"></a>
## 发现项

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  文件                                           行号 类型          行动        代码
  --------------------------------- ----------------- ------------- ------------- -------------------------------------------------------------------------------------------------------------------------------------------------------
  `trainer/rollout_engine.py`                      13 分布式   调查   `import torch.distributed as dist`

  `trainer/rollout_engine.py`                      66 AMP           调查   `def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "auto", autocast_ctx=None):`

  `trainer/rollout_engine.py`                      70 AMP           调查   `self.autocast_ctx = autocast_ctx`

  `trainer/rollout_engine.py`                      74 AMP           调查   `ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()`

  `trainer/rollout_engine.py`                     215 AMP           调查   `autocast_ctx = None,`

  `trainer/rollout_engine.py`                     221 AMP           调查   `return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)`

  `trainer/train_agent.py`                         18 分布式   调查   `import torch.distributed as dist`

  `trainer/train_agent.py`                        276 AMP           调查   `with autocast_ctx:`

  `trainer/train_agent.py`                        416 分布式   调查   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_agent.py`                        417 分布式   调查   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_agent.py`                        427 AMP           调查   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_agent.py`                        449 AMP           调查   `autocast_ctx=autocast_ctx,`

  `trainer/train_agent.py`                        478 分布式   调查   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_agent.py`                        486 CUDA          审查        `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"), collate_fn=collate_fn)` 参考

  `trainer/train_distillation.py`                  13 分布式   调查   `import torch.distributed as dist`

  `trainer/train_distillation.py`                  73 AMP           调查   `with autocast_ctx:`

  `trainer/train_distillation.py`                 197 分布式   调查   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_distillation.py`                 198 分布式   调查   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_distillation.py`                 210 AMP           调查   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_distillation.py`                 230 AMP           调查   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_distillation.py`                 249 分布式   调查   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_distillation.py`                 257 CUDA       审查       `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` 参考

  `trainer/train_dpo.py`                           13 分布式   调查   `import torch.distributed as dist`

  `trainer/train_dpo.py`                           73 AMP       调查   `with autocast_ctx:`

  `trainer/train_dpo.py`                          160 分布式   调查   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_dpo.py`                          161 分布式   调查   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_dpo.py`                          172 AMP       调查   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_dpo.py`                          194 AMP       调查   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_dpo.py`                          213 分布式   调查   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_dpo.py`                          221 CUDA       审查       `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` 参考

  `trainer/train_full_sft.py`                      12 分布式   调查   `import torch.distributed as dist`

  `trainer/train_full_sft.py`                      35 AMP       调查   `with autocast_ctx:`

  `trainer/train_full_sft.py`                     112 分布式   调查   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_full_sft.py`                     113 分布式   调查   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_full_sft.py`                     124 AMP       调查   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_full_sft.py`                     139 AMP       调查   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_full_sft.py`                     158 分布式   调查   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_full_sft.py`                     166 CUDA       审查       `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` 参考

  `trainer/train_grpo.py`                          15 分布式   调查   `import torch.distributed as dist`

  `trainer/train_grpo.py`                          98 AMP       调查   `with autocast_ctx:`

  `trainer/train_grpo.py`                         248 分布式   调查   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_grpo.py`                         249 分布式   调查   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_grpo.py`                         261 AMP       调查   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_grpo.py`                         287 AMP       调查   `autocast_ctx=autocast_ctx,`

  `trainer/train_grpo.py`                         318 分布式   调查   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_grpo.py`                         327 CUDA       审查       `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` 参考

  `trainer/train_lora.py`                          12 分布式   调查   `import torch.distributed as dist`

  `trainer/train_lora.py`                          36 AMP       调查   `with autocast_ctx:`

  `trainer/train_lora.py`                         106 分布式   调查   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_lora.py`                         107 分布式   调查   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_lora.py`                         118 AMP       调查   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_lora.py`                         152 AMP       调查   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_lora.py`                         169 分布式   调查   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_lora.py`                         177 CUDA       审查       `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` 参考

  `trainer/train_ppo.py`                           13 分布式   调查   `import torch.distributed as dist`

  `trainer/train_ppo.py`                          173 AMP           调查   `with autocast_ctx:`

  `trainer/train_ppo.py`                          357 分布式   调查   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_ppo.py`                          358 分布式   调查   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_ppo.py`                          369 AMP           调查   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_ppo.py`                          399 AMP           调查   `autocast_ctx=autocast_ctx,`

  `trainer/train_ppo.py`                          434 分布式   调查   `actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])`

  `trainer/train_ppo.py`                          435 分布式   调查   `critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])`

  `trainer/train_ppo.py`                          444 CUDA          审查   `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` 参考

  `trainer/train_pretrain.py`                      12 分布式   调查   `import torch.distributed as dist`

  `trainer/train_pretrain.py`                      35 AMP           调查   `with autocast_ctx:`

  `trainer/train_pretrain.py`                     115 分布式   调查   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_pretrain.py`                     116 分布式   调查   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_pretrain.py`                     127 AMP           调查   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_pretrain.py`                     142 AMP           调查   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_pretrain.py`                     161 分布式   调查   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_pretrain.py`                     169 CUDA          审查   `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` 参考

  `trainer/trainer_utils.py`                       12 分布式   调查   `import torch.distributed as dist`

  `trainer/trainer_utils.py`                       36 CUDA          审查   `if torch.cuda.is_available():` 参考

  `trainer/trainer_utils.py`                       37 设备        适配   `return torch.device("cuda")` 安装位置

  `trainer/trainer_utils.py`                       68 CUDA          审查   `if device.type == "cuda":` 参考

  `trainer/trainer_utils.py`                       69 CUDA 工具  调查   `torch.cuda.empty_cache()`

  `trainer/trainer_utils.py`                      107 CUDA          审查   `if device.type != "cuda":` 参考

  `trainer/trainer_utils.py`                      108 CUDA          审查   `raise RuntimeError(f"Distributed training is currently supported only with CUDA, got device: {device}.")` 参考

  `trainer/trainer_utils.py`                      110 分布式   调查   `dist.init_process_group(backend="nccl")`

  `trainer/trainer_utils.py`                      111 分布式   调查   `local_rank = int(os.environ["LOCAL_RANK"])`

  `trainer/trainer_utils.py`                      112 分布式   调查   `torch.cuda.set_device(local_rank)`

  `trainer/trainer_utils.py`                      113 分布式   调查   `return local_rank`

  `trainer/trainer_utils.py`                      120 CUDA          审查   `if torch.cuda.is_available():` 参考

  `trainer/trainer_utils.py`                      121 CUDA          审查   `torch.cuda.manual_seed(seed)` 参考

  `trainer/trainer_utils.py`                      122 CUDA          审查   `torch.cuda.manual_seed_all(seed)` 参考

  `trainer/trainer_utils.py`                      225 设备        适配   `def __init__(self, model_path, device="cuda", dtype=torch.float16):` 安装位置
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<a id="action-meanings"></a>
## 操作含义

-   **适配** --- 预期需要一个支持 DirectML 的实现。
-   **调查** --- 必须在更改之前测试其行为或 DirectML 支持。
-   **审查** --- 手动检查；该出现可能无害。

<a id="recommended-review-order"></a>
## 推荐的审查顺序

1.  设备放置
2.  CUDA 工具
3.  AMP / 混合精度
4.  分布式训练
5.  精度选项
6.  通用 CUDA 引用
