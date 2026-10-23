[English](training_commands.md) | [Français](training_commands.fr.md) | [简体中文](training_commands.zh-CN.md)

[README du projet](../../README.fr.md)

<a id="minimind-directml-----historical-training-commands-and-checkpoint-resume"></a>
# MiniMind DirectML --- Commandes historiques de formation et reprise de point d'arrêt

> **Archive DirectML historique — mise à jour le 2026-09-24**
>
> DirectML n'est plus le backend actif de la formation de MiniMind. La validation finale M5 a isolé une divergence numérique spécifique à la chaîne de traitement DirectML FP16. Les résultats historiques ci-dessous sont conservés pour la reproductibilité et comme référence technique ; le développement ROCm est documenté séparément.

Ce document centralise les commandes de formation DirectML utilisées lors de l'étude de faisabilité et explique comment les points d'arrêt automatiques et la reprise de formation fonctionnaient sur cette branche. Ces commandes sont conservées uniquement pour la reproduction et ne sont **pas les commandes recommandées pour une nouvelle formation complète**.

<a id="current-recommendation"></a>
## Recommandation actuelle

Ne lancez pas une nouvelle formation complète de MiniMind à l'aide des commandes DirectML ci-dessous. L'évaluation finale M5 a confirmé une grande lacune en qualité : sur les mêmes tokens `204,327`, le point d'arrêt formé avec DirectML a obtenu une perte `6.880233` et une perplexité `972.8529`, contre `1.846587` et `6.3382` pour le point d'arrêt formé avec ROCm.

Ces commandes sont conservées uniquement pour la reproduction historique. Le parcours DirectML est terminé à M5 ; toute commande ROCm appartient au document de référence actif ROCm, et non à cette archive.

<a id="trainer-configurations"></a>
## Configuration du formateur

Les valeurs ci-dessous sont les **paramètres par défaut proposés par les formateurs MiniMind de référence**. Ils ont été utilisés comme configurations cibles dans le workflow séquentiel historique DirectML.

|Formateur|Taille du batch|Longueur max de séquence|Accumulation|
| --- | ---: | ---: | ---: |
|Pré-formation dense| 32 | 340 | 8 |
|Formation complète dense (SFT)| 16 | 768 | 1 |
|LoRA| 32 | 340 | 1 |
|DPO| 4 | 1024 | 1 |
|Distillation| 32 | 340 | 1 |
|GRPO| 2 | 768 | 1 |
|PPO| 2 | 768 | 1 |
|Apprentissage par agents (RL)| 2 | 1024 | 1 |

Les paramètres numériques spécifiques à DirectML restent :

``` text
Device: directml:1
DType: float16
DirectML static loss scale: 1024
torch.compile: disabled
```

Pour la formation DirectML FP16, le projet utilise également :

- des poids maîtres en FP32 pour les mises à jour de l'optimiseur ;
- une perte par token (`reduction="none"`) suivie d'une moyenne en FP32 sur les tokens valides.

Ces adaptations évitent l'instabilité de l'optimiseur en FP16 et les réductions incorrectes de la perte DirectML identifiées lors de la validation prolongée.

La pré-formation dense utilise également des poids maîtres en FP32 avec AdamW `eps = 1e-8`. Les autres chemins de précision spécifiques aux formateurs restent inchangés sauf s'ils sont explicitement documentés.

<a id="historical-m4-directml-baseline"></a>
### Base historique M4 DirectML

La base de validation M4 soutenue était :

``` text
Trainer: Dense Pretrain
batch_size = 8
max_seq_len = 340
accumulation_steps = 8
compute dtype = float16
loss_scale = 1024
AdamW eps = 1e-4
```

Cette configuration reste utile comme preuve historique de validation M4.

<a id="current-m5-dense-reference"></a>
### Référence actuelle M5 Dense

Le chemin de pré-formation dense conservé utilise :

``` text
batch_size = 32
max_seq_len = 340
accumulation_steps = 8
compute dtype = float16
optimizer weights = float32 master weights
AdamW eps = 1e-8
loss = per-token cross-entropy → FP32 valid-token mean
```

Ce chemin corrigé a réussi à compléter les deux époques de pré-formation complète.

Les valeurs plus élevées dans le tableau restent des cibles de formation à valider individuellement sur DirectML.

<a id="moe-directml-override"></a>
### Surcharge DirectML (MoE)

Les pré-formation et la formation complète (Full SFT) exposent les mêmes paramètres par défaut quel que soit le mode Dense/MoE. Pour cette branche DirectML, le mode MoE conserve intentionnellement un batch physique conservateur car le chemin de compatibilité sans dispersion évalue tous les experts :

|Formateur MoE|Batch DirectML|Longueur max de séquence|Accumulation|
| --- | ---: | ---: | ---: |
|MoE Pretrain| 1 | 340 | 64 |
|MoE Full SFT| 1 | 768 | 64 |

Ces valeurs MoE sont une surcharge de sécurité DirectML, et non des paramètres par défaut en amont.

Le pré-entraînement et le Full SFT en amont actuels exposent `--seed 42`. La branche DirectML actuelle utilisée dans ce projet n'expose peut-être pas encore cette option CLI, donc les commandes ci-dessous ne forcent pas `--seed` ; conservez le comportement actuel de la graine de la branche jusqu'à ce que l'argumentaire local soit synchronisé.

Exécutez les commandes individuelles du traineur à partir de :

``` powershell
cd trainer
```

La pipeline complète séquentielle est lancée depuis le répertoire racine avec :

``` powershell
.\scripts\train_all.ps1
```

<a id="checkpoint-behavior"></a>
## Comportement du point d'arrêt

Chaque commande ci-dessous utilise la valeur par défaut actuelle de `save_interval` de l'entraîneur en amont, sauf si une exécution de validation DirectML courte l'override intentionnellement.

L'entraîneur écrit un point d'arrêt de reprise à l'intervalle configuré et à la fin normale d'une époque.

Deux types de fichiers sont produits :

``` text
out/
    # Model weights intended for later stages / inference.

checkpoints/
    # Complete resume state used to continue interrupted training.
```

Par exemple, le pré-entraînement Dense produit :

``` text
out/pretrain_768.pth
checkpoints/pretrain_768.pth
checkpoints/pretrain_768_resume.pth
```

Le fichier `_resume.pth` est le fichier important pour reprendre l'entraînement. Il contient l'état du modèle, l'état de l'optimiseur, l'époque, l'étape, la taille du monde et l'état du scalateur si applicable. Les entraîneurs avec des planificateurs stockent également leur état de planificateur. PPO stocke en plus l'état du critique et les états de l'optimiseur/schéma du critique.

L'implémentation actuelle du point d'arrêt conserve le **dernier point d'arrêt** pour chaque étape d'entraînement. Un nouveau sauvegarde remplace le point d'arrêt précédent pour le même nom de poids. Par conséquent, la distance de récupération est approximativement bornée par la valeur configurée de `save_interval` ; l'implémentation actuelle ne crée pas une archive permanente de chaque point d'arrêt.

<a id="resume-an-interrupted-training"></a>
## Reprendre un entraînement interrompu

Pour reprendre, exécutez à nouveau la **même commande d'entraînement** et ajoutez :

``` text
--from_resume 1
```

Ne pas modifier les paramètres de l'architecture lors de la reprise. En particulier, gardez la même taille cachée, le nombre de couches, la configuration Dense/MoE et la phase d'entraînement.

L'entraîneur recherche automatiquement le fichier correspondant dans `checkpoints/`, restaure l'état sauvegardé, saute les lots déjà effectués de l'époque actuelle et reprend à l'étape suivante.

Exemple :

``` text
Checkpoint saved at step 1200
Training interrupted at step 1267

Resume:
    --from_resume 1

The trainer reloads step 1200 and continues from step 1201.
The work from steps 1201–1267 must be recomputed.
```

------------------------------------------------------------------------

<a id="1-dense-pretrain"></a>
# 1. Pré-entraînement Dense

Entraînement complet :

``` powershell
python train_pretrain.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 32 `
  --learning_rate 5e-4 `
  --num_workers 8 `
  --accumulation_steps 8 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 340 `
  --use_moe 0 `
  --directml_loss_scale 1024 `
  --from_weight none `
  --use_compile 0
```

Reprise :

``` powershell
python train_pretrain.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 32 `
  --learning_rate 5e-4 `
  --num_workers 8 `
  --accumulation_steps 8 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 340 `
  --use_moe 0 `
  --directml_loss_scale 1024 `
  --from_weight none `
  --from_resume 1 `
  --use_compile 0
```

Point d'arrêt de reprise :

``` text
checkpoints/pretrain_768_resume.pth
```

------------------------------------------------------------------------

<a id="2-dense-full-sft"></a>
# 2. Full SFT Dense

Nécessite `out/pretrain_768.pth`.

Entraînement complet :

``` powershell
python train_full_sft.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 16 `
  --learning_rate 1e-5 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 768 `
  --use_moe 0 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight pretrain `
  --use_compile 0
```

Reprise : utilisez la même commande et ajoutez :

``` text
--from_resume 1
```

Point d'arrêt de reprise :

``` text
checkpoints/full_sft_768_resume.pth
```

------------------------------------------------------------------------

<a id="3-moe-pretrain"></a>
# 3. Pré-entraînement MoE

La branche MoE utilise la configuration micro-batch conservatrice utilisée par le script d'entraînement séquentiel.

``` powershell
python train_pretrain.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 1 `
  --learning_rate 5e-4 `
  --num_workers 8 `
  --accumulation_steps 64 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 340 `
  --use_moe 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight none `
  --use_compile 0
```

Reprise : ajoutez :

``` text
--from_resume 1
```

Point d'arrêt de reprise :

``` text
checkpoints/pretrain_768_moe_resume.pth
```

------------------------------------------------------------------------

<a id="4-moe-full-sft"></a>
# 4. Full SFT MoE

Nécessite `out/pretrain_768_moe.pth`.

``` powershell
python train_full_sft.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 1 `
  --learning_rate 1e-5 `
  --num_workers 8 `
  --accumulation_steps 64 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 768 `
  --use_moe 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight pretrain `
  --use_compile 0
```

Reprise : ajoutez :

``` text
--from_resume 1
```

Point d'arrêt de reprise :

``` text
checkpoints/full_sft_768_moe_resume.pth
```

------------------------------------------------------------------------

<a id="5-lora"></a>
# 5. LoRA

Utilise par défaut le modèle Dense Full SFT.

``` powershell
python train_lora.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 10 `
  --batch_size 32 `
  --learning_rate 1e-4 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 10 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 340 `
  --use_moe 0 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight full_sft `
  --use_compile 0
```

Reprise : ajoutez :

``` text
--from_resume 1
```

Avec la valeur par défaut `--lora_name lora_medical`, le point d'arrêt de reprise est :

``` text
checkpoints/lora_medical_768_resume.pth
```

Si `--lora_name` est modifié, le nom du point d'arrêt change en conséquence.

------------------------------------------------------------------------

<a id="6-dpo"></a>
# 6. DPO

La pipeline séquentielle conserve DPO sur la branche Dense Full SFT.

``` powershell
python train_dpo.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 1 `
  --batch_size 4 `
  --learning_rate 4e-8 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 100 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 1024 `
  --use_moe 0 `
  --beta 0.15 `
  --from_weight full_sft `
  --use_compile 0
```

Reprise : ajoutez :

``` text
--from_resume 1
```

**Important** : DPO se trouve en dehors du dernier 9/9 DirectML trainer smoke runner. Le traineur actuel doit être vérifié avant d'être utilisé pour une longue exécution DirectML FP16.

------------------------------------------------------------------------

<a id="7-distillation-----dense-student--moe-teacher"></a>
# 7. Distillation --- Élève dense ← Enseignant MoE

Nécessite :

``` text
out/full_sft_768.pth
out/full_sft_768_moe.pth
```

Commande :

``` powershell
python train_distillation.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 6 `
  --batch_size 32 `
  --learning_rate 5e-6 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 100 `
  --max_seq_len 340 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --student_hidden_size 768 `
  --student_num_layers 8 `
  --student_use_moe 0 `
  --from_student_weight full_sft `
  --teacher_hidden_size 768 `
  --teacher_num_layers 8 `
  --teacher_use_moe 1 `
  --from_teacher_weight full_sft `
  --use_compile 0
```

Reprise : ajouter :

``` text
--from_resume 1
```

Avec la configuration par défaut `--save_weight full_dist`, le point de reprise est :

``` text
checkpoints/full_dist_768_resume.pth
```

------------------------------------------------------------------------

<a id="8-grpo"></a>
# 8. GRPO

``` powershell
python train_grpo.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 1 `
  --batch_size 2 `
  --learning_rate 3e-7 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 1 `
  --save_interval 10 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 768 `
  --max_gen_len 1024 `
  --use_moe 0 `
  --num_generations 6 `
  --beta 0.1 `
  --loss_type cispo `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight full_sft `
  --use_compile 0
```

Reprise : ajouter :

``` text
--from_resume 1
```

Point de reprise :

``` text
checkpoints/grpo_768_resume.pth
```

L'état de reprise contient également l'état du planificateur.

------------------------------------------------------------------------

<a id="9-ppo"></a>
# 9. PPO

``` powershell
python train_ppo.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 1 `
  --batch_size 2 `
  --learning_rate 3e-7 `
  --critic_learning_rate 5e-7 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 1 `
  --save_interval 10 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 768 `
  --max_gen_len 1024 `
  --use_moe 0 `
  --mini_batch_size 2 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight full_sft `
  --use_compile 0
```

Reprise : ajouter :

``` text
--from_resume 1
```

Point de reprise :

``` text
checkpoints/ppo_actor_768_resume.pth
```

PPO restaure à la fois l'état de formation de l'acteur et de l'expert.

------------------------------------------------------------------------

<a id="10-agent-rl"></a>
# 10. Agent RL

``` powershell
python train_agent.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 1 `
  --batch_size 2 `
  --learning_rate 3e-7 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 1 `
  --save_interval 10 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 1024 `
  --max_gen_len 768 `
  --max_total_len 2500 `
  --use_moe 0 `
  --num_generations 4 `
  --beta 0.1 `
  --loss_type cispo `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight full_sft `
  --use_compile 0
```

Reprise : ajouter :

``` text
--from_resume 1
```

Point de reprise :

``` text
checkpoints/agent_768_resume.pth
```

L'état du planificateur est restauré comme partie du point de reprise.

------------------------------------------------------------------------

<a id="full-sequential-training"></a>
# Formation séquentielle

À partir de la racine du dépôt :

``` powershell
.\scripts\train_all.ps1
```

Le script séquentiel doit préserver l'intervalle de sauvegarde prévu par chaque traineur, sauf si une exécution de validation DirectML surcharge intentionnellement cet intervalle.

Si une étape est interrompue, ne redémarrez pas immédiatement l'ensemble du pipeline. Reprenez directement l'entraîneur interrompu avec la commande ci-dessus et `--from_resume 1`. Une fois que cette étape est terminée, les étapes restantes peuvent être lancées individuellement ou le script séquentiel peut être adapté pour démarrer à partir de l'étape suivante.

<a id="quick-resume-rule"></a>
## Règle de reprise rapide

``` text
Normal start:
    --from_resume 0   (default)

After interruption:
    same command
    + --from_resume 1
```

Avant de reprendre, vérifiez que le fichier attendu `_resume.pth` existe dans `checkpoints/`.

Exemple :

``` powershell
Get-ChildItem ..\checkpoints\*_resume.pth
```

lorsqu'exécuté depuis `trainer/`, ou :

``` powershell
Get-ChildItem .\checkpoints\*_resume.pth
```

lorsqu'exécuté depuis la racine du dépôt.

<a id="important-notes"></a>
## Remarques importantes

-   `out/*.pth` contient les poids du modèle ; il n'est pas suffisant pour récupérer l'avancement de l'optimiseur.
-   `checkpoints/*_resume.pth` est le fichier utilisé par `--from_resume 1`.
-   Conservez `out/` et `checkpoints/` lorsque vous souhaitez reprendre.
-   Ne nettoyez pas le dossier de sauvegarde après une formation interrompue.
-   Reprenez avec la même architecture et configuration d'étape.
-   Le point de sauvegarde est remplacé à chaque sauvegarde ; l'implémentation actuelle ne conserve pas une histoire numérotée de chaque sauvegarde toutes les 100 étapes.
