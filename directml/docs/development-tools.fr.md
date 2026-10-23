[English](development-tools.md) | [Français](development-tools.fr.md) | [简体中文](development-tools.zh-CN.md)

[README du projet](../../README.fr.md)

<a id="development-tools"></a>
# Outils de développement

> **Archive historique de DirectML — mis à jour le 2026-09-24**
>
> DirectML n'est plus le backend actif pour le training de MiniMind. La validation finale M5 a isolé une divergence numérique spécifique à la chaîne de traitement DirectML FP16 testée. Les résultats historiques ci-dessous sont conservés pour la reproductibilité et comme référence technique ; le développement ROCm est documenté séparément.

Ce document décrit les outils de développement ajoutés pour soutenir et valider l'adaptation historique de MiniMind à DirectML. Ils restent utiles pour les tests de régression, le diagnostic et la documentation de l'étude de faisabilité de DirectML, mais les travaux de training actuels ciblent désormais ROCm.

<a id="directml-compatibility-audit"></a>
## Audit de compatibilité DirectML

L'audit DirectML scanne le projet en recherchant du code qui pourrait dépendre de comportements spécifiques à CUDA.

Lancer depuis le répertoire racine du projet :

``` bash
python scripts/audit_directml.py
```

Le rapport généré est écrit à :

``` text
docs/directml_audit.md
```

L'audit détecte des préoccupations de compatibilité telles que :

-   références spécifiques à CUDA ;
-   hypothèses sur la localisation des dispositifs ;
-   outils CUDA ;
-   précision mixte automatique (AMP) ;
-   code de formation distribuée.

Ce rapport est conçu comme un outil d'analyse statique. Une occurrence signalée ne représente pas nécessairement une incompatibilité et doit être examinée manuellement.

------------------------------------------------------------------------

<a id="test-fixture-generation"></a>
## Génération de fixtures de test

Des ensembles de données petits et déterministes sont utilisés par le cadre de test automatisé pour valider les étapes de training de MiniMind sans nécessiter les données de training complètes.

Générer les fixtures avec :

``` bash
python scripts/generate_test_fixtures.py
```

Les ensembles de données générés sont stockés dans :

``` text
tests/fixtures/
```

Ils fournissent des entrées légères pour tester les flux de training tels que :

-   pré-formation ;
-   fine-tuning supervisé ;
-   DPO ;
-   RL agent ;
-   autres étapes de training prises en charge.

Ces fixtures sont destinés à des tests de compatibilité et de vérification de base, pas à l'entraînement réel des modèles.

------------------------------------------------------------------------

<a id="sequential-training"></a>
## Formation séquentielle

Le script PowerShell `train_all.ps1` exécute les étapes principales de training de MiniMind de manière séquentielle sur Windows.

Lancer depuis le répertoire racine du projet :

``` powershell
.\scripts\train_all.ps1
```

Le script exécute chaque étape de training une après l'autre. Une nouvelle étape commence uniquement après que l'étape précédente ait été terminée avec succès.

Cela évite d'exécuter plusieurs processus de training DirectML en parallèle, ce qui pourrait augmenter l'utilisation de la mémoire GPU et réduire la stabilité.

Si une étape de training échoue, le script s'arrête immédiatement au lieu de continuer avec les étapes suivantes.

Le script conserve la structure de sortie standard de MiniMind :

``` text
out/            # Generated model weights
checkpoints/    # Training resume checkpoints
```

Cet outil était conçu pour simplifier la validation et l'exécution séquentielle du pipeline de training DirectML sur Windows. Il est maintenant conservé uniquement pour la reproduction historique et les tests de régression ; il n'est pas la voie recommandée pour un training complet.

`train_all.ps1` utilise des paramètres spécifiques aux entraîneurs, tels que la taille des lots, la longueur des séquences et l'accumulation des gradients, au lieu d'appliquer une seule configuration partagée à chaque étape. Les intervalles de sauvegarde sont spécifiques aux entraîneurs et sont documentés dans `training_commands.md` ; des exécutions de validation courtes peuvent intentionnellement les surcharger.

Voir [`training_commands.md`](training_commands.fr.md) pour le tableau complet de configuration par entraîneur et les commandes de récupération `--from_resume 1`.

------------------------------------------------------------------------

<a id="real-training-benchmark"></a>
## Benchmark d'entraînement réel

L'outil `benchmark_training.py` mesure la performance d'entraînement préalable de MiniMind en utilisant le jeu de données réel de préentraînement au lieu d'un travail de compatibilité synthétique.

Lancer depuis le répertoire racine du projet :

``` bash
python scripts/benchmark_training.py
```

Un lancement typique de référence DirectML peut être démarré avec :

``` powershell
python scripts/benchmark_training.py `
    --device directml:1 `
    --batch_size 8 `
    --max_seq_len 340 `
    --accumulation_steps 8 `
    --warmup_steps 8 `
    --steps 100 `
    --model_dtype float16
```

Le benchmark effectue les opérations principales d'une étape d'entraînement préalable réelle :

-   chargement du jeu de données ;
-   passage avant ;
-   calcul de la perte ;
-   passage inverse;
-   accumulation du gradient;
-   clipping du gradient;
-   étapes de l'optimiseur AdamW.

Les itérations de préchauffage sont exclues des résultats mesurés afin que les frais d'initialisation n'altèrent pas la performance d'état stable.

Le rapport de l'outil indique des métriques incluant :

``` text
Average iteration time
Samples / second
Effective tokens / second
Padded tokens / second
Estimated steps / epoch
Estimated epoch duration
```

Ce benchmark vise à évaluer la capacité pratique de formation et le comportement de mise en œuvre durable de DirectML. Il complète le benchmark de compatibilité plus court : une configuration qui passe un test de compatibilité court n'est pas nécessairement stable lors d'une formation à long terme avec des données réelles.

------------------------------------------------------------------------

<a id="log-sync-trainer-benchmark"></a>
## Benchmark du formateur synchronisé en journal

L'outil `benchmark_log_sync.ps1` compare le formateur Dense stable avec la variante synchronisée en journal sur le même charge de travail bornée.

Exécutez-le à partir du répertoire du projet :

``` powershell
.\scripts\benchmark_log_sync.ps1
```

La comparaison de 1000 étapes actuelle a mesuré :

``` text
Stable:    2.0038 s / step
Log-Sync:  1.9307 s / step
Speedup:   1.038×
Reduction: 3.65%
```

Le benchmark valide l'amélioration du débit. Une validation de qualité de point d'arrêt reste nécessaire avant que Log-Sync ne devienne la référence finale de formation.

------------------------------------------------------------------------

<a id="bounded-training-runs"></a>
## Exécutions limitées de formation

Les entraîneurs de modèles compatibles DirectML permettent des exécutions limitées grâce à l'argument `--max_steps`.

Cela permet aux workflows d'entraînement d'être testés sur un nombre fixe d'itérations sans nécessiter une époque complète. Le comportement partagé d'exécution limitée est utilisé par l'entraînement préalable, le Full SFT, LoRA, la distillation, GRPO, l'RL agent et PPO.

Exemple :

``` powershell
python trainer/train_pretrain.py `
    --device directml:1 `
    --batch_size 8 `
    --max_seq_len 340 `
    --accumulation_steps 8 `
    --max_steps 100
```

Les exécutions limitées sont utiles pour :

-   la validation de la stabilité de l'entraînement sur des données réelles ;
-   comparer les configurations de taille de batch et de longueur de séquence ;
-   reproduire les échecs de mémoire de DirectML ;
-   étudier le comportement de retour à l'CPU ;
-   exécuter des vérifications de stabilité plus longues sans s'engager dans une époque complète de formation.

Pour le travail de performance de DirectML, des exécutions avec données réelles limitées doivent être utilisées en conjonction avec `benchmark_training.py` : le formateur valide le flux réel de formation, tandis que le benchmark fournit des mesures de débit plus ciblées.

<a id="directml-loss-validation-tests"></a>
## Tests de validation de perte DirectML

Les outils de régression ciblés pour l'investigation de la croisement de DirectML sont stockés dans `tests/` :

``` text
test_directml_cross_entropy.py
test_directml_loss_reductions.py
test_batch_loss_consistency.py
test_train_diagnostic_loss.py
```

Ils valident le comportement de réduction de DirectML `cross_entropy`, y compris le chemin de moyenne par token `reduction="none"` → FP32 des tokens valides, et comparant la perte de formation avec la perte diagnostic normalisée par token. Ces tests sont gardés séparés de `scripts/`, tandis que `scripts/diagnose_pretrain.py` reste l'outil d'analyse des points d'arrêt.

------------------------------------------------------------------------

<a id="trainer-smoke-test-runner"></a>
## Exécuteur de test de base du formateur

L'outil `tests/test_all_trainers.py` exécute les tests de fumée du traineur DirectML en mode FP16 de manière séquentielle à partir d'un point d'entrée unique en Python.

Lancer explicitement depuis le répertoire racine du projet :

``` bash
python tests/test_all_trainers.py
```

Le lanceur valide actuellement :

``` text
Dense Pretrain
Dense Full SFT
MoE Pretrain
MoE Full SFT
LoRA
Distillation
GRPO
Agent RL
PPO
```

Chaque traineur utilise une configuration bornée via `--max_steps`. Le lanceur de tests démarre chaque traineur à partir du répertoire de travail `trainer/` afin que les chemins relatifs MiniMind restent valides.

Le jeu de tests s'arrête à la première défaillance et indique le nombre de traineurs effectués. L'exécution complète de la validation M4 a passé pour tous les traineurs inclus dans ce jeu de tests :

``` text
Passed: 9/9
```

Ce lanceur est intentionnellement exécuté séparément de la suite normale `pytest -q` car il démarre des charges de formation réelles DirectML et est considérablement plus lourd que les tests unitaires et les tests de régression légers.

------------------------------------------------------------------------

<a id="m5-backend-comparison-utilities"></a>
## Outils de comparaison du backend M5

Les outils de validation finale M5 DirectML ont utilisé trois petits outils de comparaison entre backends :

```text
scripts/compare_pretrain_logs.py
    -> plots training loss, learning rate, ETA and relative loss reduction

scripts/compare_checkpoint_loss.py
    -> evaluates multiple checkpoints on the same fixed dataset subset

scripts/compare_single_batch_backend.py
    -> compares the same checkpoint and fixed batch across DirectML/ROCm
```

Ces outils séparent le comportement des journaux de formation du niveau de qualité des points d'arrêt et du comportement numérique du backend. Ils ont été utilisés uniquement pour clore l'analyse M5 DirectML et sont préférés à l'évaluation de la correction du backend à partir de la perte de formation seule.
