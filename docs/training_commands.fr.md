[English](training_commands.md) | [Français](training_commands.fr.md) | [简体中文](training_commands.zh-CN.md)

<a id="minimind-rocm--training-commands"></a>
# MiniMind ROCm — Commandes de formation

L'environnement de configuration ainsi que les sémantiques ROCm/PyTorch pour les dispositifs sont documentés dans le dossier racine [`README.md`](../README.fr.md).

L'ordre de formation, les règles mini-vs-plein et les jalons de validation sont documentés dans [`roadmap.md`](roadmap.fr.md).

Exécuter les commandes du formateur à partir de :

```powershell
cd trainer
```

<a id="gpu-selection-with-integrated-graphics"></a>
### Sélection du GPU avec graphique intégré

Sur les systèmes où le CPU fournit également un GPU AMD intégré, ROCm/PyTorch peut exposer le GPU intégré comme `cuda:0` et le GPU dédié Radeon comme un autre dispositif.

Vérifier les dispositifs détectés avant de commencer la formation :

```powershell
python -c "import torch; print('device_count:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

Exemple :

```text
0 AMD Radeon(TM) Graphics
1 AMD Radeon RX 7800 XT
```

Si le GPU dédié est le dispositif `1`, le sélectionner avant de lancer MiniMind :

```powershell
$env:HIP_VISIBLE_DEVICES="1"
```

Puis vérifier le dispositif visible :

```powershell
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Résultat attendu :

```text
1
AMD Radeon RX 7800 XT
```

Après que `HIP_VISIBLE_DEVICES` filtre les dispositifs, le GPU dédié sélectionné devient `cuda:0` dans le processus Python. Les commandes du formateur devraient donc continuer à utiliser `--device cuda:0` lorsque cette variable d'environnement est active.

`HIP_VISIBLE_DEVICES` s'applique à la session PowerShell actuelle. La réinitialiser lors de l'ouverture d'une nouvelle console si le GPU intégré serait autrement sélectionné.

---

<a id="upstream-datasets"></a>
## Ensembles de données upstream

Les ensembles de données officiels de formation de MiniMind sont fournis par le projet upstream :

- [ModelScope — minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- [Hugging Face — minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

Seuls les fichiers nécessaires doivent être téléchargés et placés dans :

```text
dataset/
```

Ensembles de données principaux :

```text
pretrain_t2t_mini.jsonl   # Mini Pretrain
pretrain_t2t.jsonl        # Full Pretrain

sft_t2t_mini.jsonl        # Mini SFT
sft_t2t.jsonl             # Full SFT

dpo.jsonl                 # DPO
rlaif.jsonl               # GRPO / PPO
agent_rl.jsonl            # Agent RL
agent_rl_math.jsonl       # Agent RL math
```

Pour une validation rapide, utilisez les ensembles de données `*_mini.jsonl`.

Pour une formation complète, utilisez `pretrain_t2t.jsonl` et `sft_t2t.jsonl`.

---

<a id="1-dense-pretrain"></a>
## 1. Pré-formation dense

Ensemble de données par défaut :

```text
../dataset/pretrain_t2t_mini.jsonl
```

<a id="mini-validation"></a>
### Validation mini

```powershell
python train_pretrain.py --save_weight pretrain_mini
```

<a id="full-training"></a>
### Formation complète

```powershell
python train_pretrain.py --data_path ../dataset/pretrain_t2t.jsonl
```

Résultat final :

```text
out/pretrain_mini_768.pth
out/pretrain_768.pth
```

---

<a id="2-dense-full-sft"></a>
## 2. Formation complète dense (SFT)

Nécessite :

```text
out/pretrain_768.pth
```

Ensemble de données par défaut :

```text
../dataset/sft_t2t_mini.jsonl
```

<a id="mini-validation-1"></a>
### Validation mini

```powershell
python train_full_sft.py --save_weight full_sft_mini
```

<a id="full-training-1"></a>
### Formation complète

```powershell
python train_full_sft.py --data_path ../dataset/sft_t2t.jsonl
```

Résultat final :

```text
out/full_sft_mini_768.pth
out/full_sft_768.pth
```

---

<a id="3-moe-pretrain"></a>
## 3. Pré-formation MoE

Utilise les mêmes paramètres par défaut de pré-formation, avec MoE activé.

```powershell
python train_pretrain.py --use_moe 1 --data_path ../dataset/pretrain_t2t.jsonl
```

Résultat final :

```text
out/pretrain_768_moe.pth
```

Si la mémoire VRAM est insuffisante, réduisez `batch_size` et augmentez `accumulation_steps` en conséquence.

---

<a id="4-moe-full-sft"></a>
## 4. Formation complète MoE (SFT)

Nécessite :

```text
out/pretrain_768_moe.pth
```

```powershell
python train_full_sft.py --use_moe 1 --data_path ../dataset/sft_t2t.jsonl
```

Résultat final :

```text
out/full_sft_768_moe.pth
```

---

<a id="5-lora"></a>
## 5. LoRA

Nécessite une formation dense complète.

Configuration par défaut utilise déjà :

```text
base weight: full_sft
dataset: lora_medical.jsonl
Dense model
```

<a id="training"></a>
### Formation

```powershell
python train_lora.py
```

Résultat final :

```text
out/lora_medical_768.pth
```

<a id="evaluation"></a>
### Évaluation

LoRA est évalué conjointement avec son modèle de base.

La valeur `--weight` doit correspondre au modèle de base utilisé lors de la formation LoRA.

```powershell
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

Exemple avec un autre adaptateur LoRA :

```powershell
python eval_llm.py --weight full_sft --lora_weight lora_identity
```

Cela conserve les capacités générales du modèle de base tout en appliquant le comportement spécifique au domaine appris par l'adaptateur LoRA.

<a id="merge"></a>
### Fusion

Un adaptateur LoRA peut être optionnellement fusionné dans son modèle de base pour produire un point de contrôle complet autonome.

Utiliser :

```text
scripts/convert_model.py
```

avec :

```text
convert_merge_base_lora
```

<a id="full-fine-tuning-alternative"></a>
### Alternative de formation complète

Avec une quantité suffisante de données de domaine, une formation complète par paramètre peut être utilisée au lieu de LoRA.

Les données de domaine doivent être mélangées avec soin aux données générales afin de réduire le risque d'ajustement excessif au domaine spécialisé et de dégrader les capacités générales du modèle.

---

<a id="6-dpo"></a>
## 6. DPO

Nécessite un SFT dense.

Configuration par défaut :

```text
base weight: full_sft
dataset: dpo.jsonl
beta: 0.15
```

```powershell
python train_dpo.py
```

Sortie finale :

```text
out/dpo_768.pth
```

---

<a id="7-distillation"></a>
## 7. Distillation

Élève dense ← Enseignant MoE.

Nécessite :

```text
out/full_sft_768.pth
out/full_sft_768_moe.pth
```

Configuration par défaut :

```text
student: Dense Full SFT
teacher: MoE Full SFT
student hidden size: 768
teacher hidden size: 768
alpha: 0.5
temperature: 1.5
```

```powershell
python train_distillation.py --data_path ../dataset/sft_t2t.jsonl
```

Sortie finale :

```text
out/full_dist_768.pth
```

---

<a id="8-grpo"></a>
## 8. GRPO

Nécessite un SFT dense.

Configuration par défaut :

```text
base weight: full_sft
dataset: rlaif.jsonl
loss: cispo
num_generations: 6
beta: 0.1
rollout engine: torch
```

```powershell
python train_grpo.py
```

Sortie finale :

```text
out/grpo_768.pth
```

---

<a id="9-ppo"></a>
## 9. PPO

Nécessite un SFT dense.

Configuration par défaut :

```text
base weight: full_sft
dataset: rlaif.jsonl
actor learning rate: 3e-7
critic learning rate: 5e-7
mini batch size: 2
rollout engine: torch
```

```powershell
python train_ppo.py
```

Sortie finale :

```text
out/ppo_actor_768.pth
```

---

<a id="10-agent-rl"></a>
## 10. RL d'agent

Nécessite un SFT dense.

Configuration par défaut :

```text
base weight: full_sft
dataset: agent_rl.jsonl
loss: cispo
num_generations: 4
beta: 0.1
max sequence length: 1024
max generation length: 768
max total length: 2500
rollout engine: torch
```

```powershell
python train_agent.py
```

Sortie finale :

```text
out/agent_768.pth
```

---

<a id="11-resume"></a>
## 11. Reprise de l’entraînement

Réprendre uniquement une exécution interrompue de la **même étape et même jeu de données**.

Utiliser la même commande et ajouter :

```text
--from_resume 1
```

Exemples :

```powershell
python train_pretrain.py --from_resume 1
```

```powershell
python train_pretrain.py --data_path ../dataset/pretrain_t2t.jsonl --from_resume 1
```

```powershell
python train_full_sft.py --data_path ../dataset/sft_t2t.jsonl --from_resume 1
```

Reste inchangé :

```text
dataset
hidden size
layer count
Dense / MoE
training stage
checkpoint name
```

Points de contrôle de réprise :

|Formateur|Point de contrôle|
| --- | --- |
|Pré-entraînement dense| `checkpoints/pretrain_768_resume.pth` |
|SFT dense| `checkpoints/full_sft_768_resume.pth` |
|Pré-entraînement MoE| `checkpoints/pretrain_768_moe_resume.pth` |
|SFT complet MoE| `checkpoints/full_sft_768_moe_resume.pth` |
|LoRA| `checkpoints/lora_medical_768_resume.pth` |
|Distillation| `checkpoints/full_dist_768_resume.pth` |
|GRPO| `checkpoints/grpo_768_resume.pth` |
|PPO| `checkpoints/ppo_actor_768_resume.pth` |
|RL d'agent| `checkpoints/agent_768_resume.pth` |

Vérifier les points de contrôle disponibles avec :

```powershell
Get-ChildItem ..\checkpoints\*_resume.pth
```

---
