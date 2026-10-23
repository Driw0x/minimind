[English](directml_audit.md) | [Français](directml_audit.fr.md) | [简体中文](directml_audit.zh-CN.md)

[README du projet](../../README.fr.md)

<a id="directml-compatibility-audit"></a>
# Audit de compatibilité DirectML

> **Archive historique DirectML — mis à jour le 2026-09-24**
>
> DirectML n'est plus le backend actif pour le training de MiniMind. La validation finale M5 a isolé une divergence numérique spécifique à la chaîne DirectML FP16 testée. Les résultats historiques ci-dessous sont conservés pour la reproductibilité et comme référence technique ; le développement ROCm est documenté séparément.

Généré automatiquement par `scripts/audit_directml.py`.

> **Note d'audit historique :** le traitement de la précision du traineur, l'exécution bornée de `--max_steps` et les utilitaires partagés DirectML FP16 ont été modifiés après ce point d'arrêt. Le rapport est conservé pour documenter le travail effectué sur DirectML, mais DirectML n'est plus le backend actif de training. Ce audit statique de l'époque M2 précède le diagnostic numérique ultérieur de l'analyse M5. Il est gardé séparé des benchmarks en temps réel car il enregistre des découvertes de compatibilité au niveau du code plutôt que le comportement mesuré.

> Ce rapport identifie du code spécifique à CUDA. Chaque occurrence doit encore être examinée manuellement avant toute modification.

<a id="summary"></a>
## Résumé

Nombre total de découvertes : **86**

  Catégorie             Comptage
  ------------------ -------
  AMP                     29 CUDA de référence          15 CUDA utilitaire             1 Placement du dispositif         2 Distribué             39

<a id="findings"></a>
## Découvertes

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Fichier                                           Ligne Type          Action        Code
  --------------------------------- ----------------- ------------- ------------- -------------------------------------------------------------------------------------------------------------------------------------------------------
  `trainer/rollout_engine.py`                      13 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/rollout_engine.py`                      66 AMP           Étudier   `def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "auto", autocast_ctx=None):`

  `trainer/rollout_engine.py`                      70 AMP           Étudier   `self.autocast_ctx = autocast_ctx`

  `trainer/rollout_engine.py`                      74 AMP           Étudier   `ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()`

  `trainer/rollout_engine.py`                     215 AMP           Étudier   `autocast_ctx = None,`

  `trainer/rollout_engine.py`                     221 AMP           Étudier   `return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)`

  `trainer/train_agent.py`                         18 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/train_agent.py`                        276 AMP           Étudier   `with autocast_ctx:`

  `trainer/train_agent.py`                        416 Distribué   Étudier   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_agent.py`                        417 Distribué   Étudier   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_agent.py`                        427 AMP           Étudier   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_agent.py`                        449 AMP           Étudier   `autocast_ctx=autocast_ctx,`

  `trainer/train_agent.py`                        478 Distribué   Étudier   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_agent.py`                        486 CUDA          Examiné        `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"), collate_fn=collate_fn)` de référence

  `trainer/train_distillation.py`                  13 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/train_distillation.py`                  73 AMP           Étudier   `with autocast_ctx:`

  `trainer/train_distillation.py`                 197 Distribué   Étudier   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_distillation.py`                 198 Distribué   Étudier   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_distillation.py`                 210 AMP           Étudier   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_distillation.py`                 230 AMP           Étudier   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_distillation.py`                 249 Distribué   Étudier   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_distillation.py`                 257 CUDA          Revue        `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` référence

  `trainer/train_dpo.py`                           13 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/train_dpo.py`                           73 AMP           Étudier   `with autocast_ctx:`

  `trainer/train_dpo.py`                          160 Distribué   Étudier   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_dpo.py`                          161 Distribué   Étudier   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_dpo.py`                          172 AMP           Étudier   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_dpo.py`                          194 AMP           Étudier   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_dpo.py`                          213 Distribué   Étudier   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_dpo.py`                          221 CUDA          Revue        `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` référence

  `trainer/train_full_sft.py`                      12 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/train_full_sft.py`                      35 AMP           Étudier   `with autocast_ctx:`

  `trainer/train_full_sft.py`                     112 Distribué   Étudier   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_full_sft.py`                     113 Distribué   Étudier   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_full_sft.py`                     124 AMP           Étudier   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_full_sft.py`                     139 AMP           Étudier   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_full_sft.py`                     158 Distribué   Étudier   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_full_sft.py`                     166 CUDA          Revue        `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` référence

  `trainer/train_grpo.py`                          15 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/train_grpo.py`                          98 AMP           Étudier   `with autocast_ctx:`

  `trainer/train_grpo.py`                         248 Distribué   Étudier   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_grpo.py`                         249 Distribué   Étudier   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_grpo.py`                         261 AMP           Étudier   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_grpo.py`                         287 AMP           Étudier   `autocast_ctx=autocast_ctx,`

  `trainer/train_grpo.py`                         318 Distribué   Étudier   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_grpo.py`                         327 CUDA          Revue        `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` référence

  `trainer/train_lora.py`                          12 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/train_lora.py`                          36 AMP           Étudier   `with autocast_ctx:`

  `trainer/train_lora.py`                         106 Distribué   Étudier   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_lora.py`                         107 Distribué   Étudier   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_lora.py`                         118 AMP           Étudier   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_lora.py`                         152 AMP           Étudier   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_lora.py`                         169 Distribué   Étudier   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_lora.py`                         177 CUDA          Revue        `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` référence

  `trainer/train_ppo.py`                           13 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/train_ppo.py`                          173 AMP           Étudier   `with autocast_ctx:`

  `trainer/train_ppo.py`                          357 Distribué   Étudier   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_ppo.py`                          358 Distribué   Étudier   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_ppo.py`                          369 AMP           Étudier   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_ppo.py`                          399 AMP           Étudier   `autocast_ctx=autocast_ctx,`

  `trainer/train_ppo.py`                          434 Distribué   Étudier   `actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])`

  `trainer/train_ppo.py`                          435 Distribué   Étudier   `critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])`

  `trainer/train_ppo.py`                          444 CUDA          Revue        `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` référence

  `trainer/train_pretrain.py`                      12 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/train_pretrain.py`                      35 AMP           Étudier   `with autocast_ctx:`

  `trainer/train_pretrain.py`                     115 Distribué   Étudier   `local_rank = init_distributed_mode(args.device)`

  `trainer/train_pretrain.py`                     116 Distribué   Étudier   `if dist.is_initialized(): args.device = get_device(f"cuda:{local_rank}")`

  `trainer/train_pretrain.py`                     127 AMP           Étudier   `autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device_type == "cuda" else nullcontext()`

  `trainer/train_pretrain.py`                     142 AMP           Étudier   `scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == 'float16'))`

  `trainer/train_pretrain.py`                     161 Distribué   Étudier   `model = DistributedDataParallel(model, device_ids=[local_rank])`

  `trainer/train_pretrain.py`                     169 CUDA          Revue        `loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=(device_type == "cuda"))` référence

  `trainer/trainer_utils.py`                       12 Distribué   Étudier   `import torch.distributed as dist`

  `trainer/trainer_utils.py`                       36 CUDA          Revue        `if torch.cuda.is_available():` référence

  `trainer/trainer_utils.py`                       37 Appareil       Adapter         `return torch.device("cuda")` placement

  `trainer/trainer_utils.py`                       68 CUDA          Revue        `if device.type == "cuda":` référence

  `trainer/trainer_utils.py`                       69 Outil CUDA   Étudier   `torch.cuda.empty_cache()`

  `trainer/trainer_utils.py`                      107 CUDA          Revue        `if device.type != "cuda":` référence

  `trainer/trainer_utils.py`                      108 CUDA          Revue        `raise RuntimeError(f"Distributed training is currently supported only with CUDA, got device: {device}.")` référence

  `trainer/trainer_utils.py`                      110 Distribué   Étudier   `dist.init_process_group(backend="nccl")`

  `trainer/trainer_utils.py`                      111 Distribué   Étudier   `local_rank = int(os.environ["LOCAL_RANK"])`

  `trainer/trainer_utils.py`                      112 Distribué   Étudier   `torch.cuda.set_device(local_rank)`

  `trainer/trainer_utils.py`                      113 Distribué   Étudier   `return local_rank`

  `trainer/trainer_utils.py`                      120 CUDA          Revue        `if torch.cuda.is_available():` référence

  `trainer/trainer_utils.py`                      121 CUDA          Revue        `torch.cuda.manual_seed(seed)` référence

  `trainer/trainer_utils.py`                      122 CUDA          Revue        `torch.cuda.manual_seed_all(seed)` référence

  `trainer/trainer_utils.py`                      225 Appareil       Adapter         `def __init__(self, model_path, device="cuda", dtype=torch.float16):` placement
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<a id="action-meanings"></a>
## Sens des actions

-   **Adapter** --- attendu qu'une implémentation compatible avec DirectML soit requise.
-   **Étudier** --- le comportement ou la prise en charge de DirectML doit être testé avant de le modifier.
-   **Revue** --- inspection manuelle ; l'occurrence peut être sans danger.

<a id="recommended-review-order"></a>
## Ordre recommandé de revue

1.  Position du dispositif
2.  Outils CUDA
3.  AMP / précision mixte
4.  Formation distribuée
5.  Options de précision
6.  Références générales CUDA
