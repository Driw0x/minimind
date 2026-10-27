[English](roadmap.md) | [Français](roadmap.fr.md) | [简体中文](roadmap.zh-CN.md)

[README du projet](../README.fr.md)

<a id="minimind-rocm-roadmap"></a>
# MiniMind ROCm Roadmap

Ce calendrier suit la validation du pipeline de formation de MiniMind en amont sur Windows avec ROCm.

Pour les commandes de formation reproductibles et la syntaxe de reprise de point d'arrêt, consultez [`training_commands.md`](training_commands.fr.md).

---

<a id="training-rule"></a>
## Règle de formation

Les exécutions courtes valident uniquement la compatibilité avec ROCm. Les points d'arrêt finaux doivent provenir des exécutions complètes prévues.

Ne pas poursuivre une exécution complète à partir d'un point d'arrêt de validation courte. La reprise est utilisée uniquement pour continuer la même exécution interrompue.

---

<a id="m1--dense-pretrain--evaluation-"></a>
## M1 — Pré-formation dense + évaluation ✅

Valider la pré-formation dense sur ROCm et produire le point d'arrêt de base pour la formation complète SFT.

<a id="training"></a>
### Formation

- [x] Valider l'exécution de ROCm avec une exécution courte
- [x] Compléter la pré-formation dense
- [x] Confirmer une perte finie et décroissante
- [x] Enregistrer le point d'arrêt final de pré-formation

<a id="evaluation"></a>
### Évaluation

- [x] Charger le point d'arrêt avec `python eval_llm.py --weight pretrain`
- [x] Exécuter le mode d'auto-évaluation amont (`[0] 自动测试`)
- [x] Confirmer une génération cohérente sur les prompts intégrés

**Sortie :** point d'arrêt de pré-formation dense conservé, formé sur l'ensemble de données de pré-formation, destiné à la formation SFT complète.

---

<a id="m2--dense-full-sft--evaluation-"></a>
## M2 — Formation dense complète SFT + évaluation ✅

Valider la formation SFT complète sur ROCm à partir du point d'arrêt conservé de M1.

<a id="training-1"></a>
### Formation

- [x] Valider l'exécution de ROCm avec une exécution courte de SFT complète
- [x] Confirmer une perte finie et décroissante
- [x] Enregistrer et recharger le point d'arrêt de validation
- [x] Compléter la formation SFT complète
- [x] Enregistrer le point d'arrêt final de SFT complète

<a id="evaluation-1"></a>
### Évaluation

- [x] Charger le point d'arrêt final avec `eval_llm.py`
- [x] Exécuter le mode d'auto-évaluation amont (`[0] 自动测试`)
- [x] Confirmer des réponses cohérentes et stables sur les prompts intégrés
- [x] Évaluer la précision factuelle et la capacité de raisonnement
- [x] Évaluer la cohérence dans l'obéissance aux instructions
- [x] Comparer avec le point d'arrêt de pré-formation M1 sur les mêmes prompts

**Limites connues :** la précision factuelle, le raisonnement arithmétique, la correction de code et l'obéissance stricte aux instructions restent peu fiables.

**Sortie :** point d'arrêt de formation SFT dense conservé, formé sur l'ensemble de données de SFT, destiné aux étapes suivantes.

---

<a id="m3--remaining-upstream-pipeline-"></a>
## M3 — Étapes restantes du pipeline amont ⏳

Valider les étapes restantes du pipeline de formation de MiniMind sur ROCm sans rédesigner le pipeline amont.

<a id="pipeline"></a>
### Pipeline

```text
Dense Full SFT
    ├── LoRA
    ├── DPO
    ├── GRPO
    ├── PPO
    └── Agent RL

MoE Pretrain
    ↓
MoE Full SFT

Dense Full SFT + MoE Full SFT
    ↓
Distillation
```

<a id="dense-stages"></a>
### Étapes dense

- [ ] Valider LoRA
- [ ] Valider DPO
- [ ] Valider GRPO
- [ ] Valider PPO
- [ ] Valider l'apprentissage par agent RL

<a id="moe"></a>
### MoE

- [x] Valider et compléter la pré-formation MoE
- [x] Valider et compléter la formation SFT complète MoE
- [x] Évaluer le point d'arrêt final MoE

Points de contrôle conservés : `pretrain_768_moe.pth` et `full_sft_768_moe.pth`, entraînés sur `pretrain_t2t.jsonl` puis `sft_t2t.jsonl`, avec Full SFT initialisé à partir de MoE Pretrain. L'évaluation initiale sur ROCm est terminée : pertes sur les références Mini et huit prompts externes déterministes. Les résultats et les limites méthodologiques figurent dans [`generation_quality.md`](generation_quality.fr.md).

<a id="distillation"></a>
### Distillation

- [ ] Valider la distillation avec un étudiant dense et un enseignant MoE
- [ ] Compléter la distillation
- [ ] Évaluer le point d'arrêt distillé

<a id="final-validation"></a>
### Validation finale

- [ ] Vérifier la compatibilité du point d'arrêt entre les étapes
- [ ] Enregistrer l'environnement final ROCm et les commandes de formation
- [ ] Conserver uniquement les modifications spécifiques à ROCm nécessaires provenant de la branche principale

---

<a id="validation-rule"></a>
## Règle de validation

Une étape est validée lorsque les vérifications pertinentes sont passées :

```text
ROCm execution
    +
finite and stable training
    +
checkpoint save/load
    +
upstream evaluation
```

Les modifications spécifiques à ROCm doivent être introduites uniquement pour des problèmes de compatibilité vérifiés.
