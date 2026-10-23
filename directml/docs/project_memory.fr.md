[English](project_memory.md) | [Français](project_memory.fr.md) | [简体中文](project_memory.zh-CN.md)

[README du projet](../../README.fr.md)

<a id="minimind-----project-memory"></a>
# MiniMind --- Mémoire du projet

> **Archive historique de DirectML — mis à jour le 24 septembre 2026**
>
> DirectML n'est plus le backend actif pour le training de MiniMind. La validation finale M5 a isolé une divergence numérique spécifique au chemin FP16 testé de DirectML. Les résultats historiques ci-dessous sont conservés pour la reproductibilité et comme référence technique ; le développement de ROCm est documenté séparément.

Ce document fournit une mémoire technique concise de l'adaptation de MiniMind à DirectML et de la décision finale de cesser l'utilisation de DirectML pour le training et de passer à ROCm.

Il enregistre les principales décisions architecturales, les leçons techniques et les références de validation qui devraient rester utiles tout au long du projet.

Les problèmes techniques détaillés sont documentés dans [`directml_issues.md`](directml_issues.fr.md).

Les résultats de benchmark et les expériences de performance sont documentés dans [`directml_benchmarks.md`](directml_benchmarks.fr.md).

Les limitations actuelles et leurs contournements sont résumés dans [`directml_issues.md`](directml_issues.fr.md).

La progression du développement est suivie séparément dans [`update_log.md`](update_log.fr.md).

------------------------------------------------------------------------

<a id="architecture-decisions"></a>
# Décisions architecturales

<a id="keep-directml-changes-close-to-upstream"></a>
## Conserver les modifications DirectML proches de l'origine

Le support de DirectML doit modifier autant que possible le logiciel d'entraînement original de MiniMind.

Le pipeline d'exécution existant de MiniMind doit être réutilisé autant que possible :

``` text
Model
  ↓
Standard PyTorch operations
  ↓
Resolved execution device
  ↓
CPU / CUDA / DirectML
```

Les copies spécifiques à DirectML de pipelines d'entraînement existants doivent être évitées.

<a id="decision"></a>
### Décision

Des outils de compatibilité partagés sont préférables à des implémentations séparées de DirectML.

------------------------------------------------------------------------

<a id="separate-backend-compatibility-from-training-logic"></a>
## Séparer la compatibilité du backend de la logique d'entraînement

Les algorithmes d'entraînement doivent rester indépendants du backend autant que possible.

Les responsabilités spécifiques au backend incluent :

-   la résolution du dispositif ;
-   l'initialisation de DirectML ;
-   la mise en place du modèle ;
-   les exigences spécifiques au dispositif des composants ;
-   les règles de compatibilité du backend.

Ces responsabilités doivent être placées dans une infrastructure partagée de formateur plutôt que dans les algorithmes d'entraînement individuels.

<a id="decision-1"></a>
### Décision

Les formateurs individuels doivent se concentrer sur leur algorithme d'entraînement.

La compatibilité avec le backend doit être centralisée dans les outils partagés de formateur.

------------------------------------------------------------------------

<a id="allow-component-specific-device-placement"></a>
## Autoriser une mise en place spécifique au dispositif des composants

Tous les composants d'un flux d'entraînement n'ont pas besoin d'exécuter sur le même dispositif.

Certains modèles ou opérations peuvent nécessiter une exécution sur CPU lorsque DirectML ne supporte pas leur chemin complet d'inférence.

La configuration actuelle de l'entraînement d'alignement peut donc utiliser :

``` text
Trainable model → DirectML
Reward model    → CPU
```

Les données sont transférées entre dispositifs lorsqu'elles le nécessitent.

<a id="decision-2"></a>
### Décision

La correction et la stabilité prennent priorité sur l'obligation de placer chaque composant sur DirectML.

------------------------------------------------------------------------

<a id="preserve-minimind-checkpoint-semantics"></a>
## Préserver les sémantiques des points d'arrêt de MiniMind

DirectML est un backend d'exécution et ne doit pas introduire un format de point d'arrêt distinct.

La compatibilité des points d'arrêt dépend de facteurs tels que :

``` text
Model architecture
Training stage
State dictionary structure
```

et non pas de la manière dont le modèle a été exécuté sur CPU, CUDA ou DirectML.

<a id="decision-3"></a>
### Décision

Les sémantiques existantes des points d'arrêt de MiniMind doivent rester inchangées.

La compatibilité avec le backend et la compatibilité avec les points d'arrêt doivent être validées de manière indépendante.

------------------------------------------------------------------------

<a id="important-technical-lessons"></a>
# Leçons techniques importantes

<a id="directml-device-representation"></a>
## Représentation du dispositif DirectML

`torch-directml` intègre DirectML via le backend `PrivateUse1` de PyTorch.

Un dispositif DirectML apparaît donc internement sous la forme :

    privateuseone:<index>

L'indice de l'adaptateur est conservé entre le dispositif DirectML visible par l'utilisateur et la représentation interne de PyTorch.

Par exemple :

    directml:0 → privateuseone:0
    directml:1 → privateuseone:1

Ceci est un comportement attendu.

Les options de dispositif MiniMind visible par l'utilisateur restent :

    directml
    directml:<index>

<a id="lesson"></a>
### Leçon

Ne considérez pas `privateuseone` comme un backend de dispositif distinct.

C'est la représentation interne de PyTorch du dispositif DirectML sélectionné via `torch-directml`.

Lorsque plusieurs GPU sont disponibles, l'indice du dispositif est significatif et doit être conservé lors de la documentation ou de la comparaison des exécutions.

------------------------------------------------------------------------

<a id="cpu-fallbacks-are-acceptable-when-necessary"></a>
## Les retours à CPU sont acceptables lorsqu'ils sont nécessaires

DirectML ne prend pas en charge toutes les opérations utilisées par MiniMind et ses dépendances dans PyTorch.

Certaines opérations non prises en charge peuvent passer automatiquement en CPU.

D'autres composants peuvent nécessiter de rester explicitement sur CPU.

Des exemples actuels incluent :

``` text
AdamW unsupported operation → automatic CPU fallback
Reward model                → CPU execution
```

<a id="lesson-1"></a>
### Leçon

Un retour à CPU n'est pas automatiquement une échec de compatibilité.

Les retours à CPU sont acceptables lorsque la correction est préservée, mais leur impact sur les performances doit être évalué séparément.

Les retours actuels sont suivis dans [`directml_issues.md`](directml_issues.fr.md).

------------------------------------------------------------------------

<a id="physical-gpu-selection-must-be-explicit"></a>
## Le choix du GPU physique doit être explicite

L'ordinateur de développement expose plusieurs adaptateurs graphiques, y compris un GPU intégré et un GPU dédié.

Les deux peuvent être visibles pour DirectML.

Par conséquent :

``` text
CPU
 ≠
Integrated GPU
 ≠
Dedicated GPU
```

Le choix implicite du dispositif DirectML peut rendre les résultats de benchmark difficiles à interpréter car plusieurs exécutions peuvent s'exécuter sur des GPU physiques différents.

<a id="lesson-2"></a>
### Leçon

Le dispositif DirectML physique souhaité doit être sélectionné explicitement avant l'entraînement ou le benchmarking.

Les mesures de performance de référence doivent uniquement être comparées lorsqu'elles utilisent le même GPU physique.

------------------------------------------------------------------------

<a id="benchmark-compatibility-is-not-full-training-stability"></a>
## La compatibilité de benchmark n'est pas la stabilité complète de l'entraînement

Les benchmarks de compatibilité courts sont utiles pour identifier les configurations qui échouent immédiatement.

Cependant :

``` text
Benchmark PASS
    ≠
Guaranteed full-training stability
```

Une exécution complète d'entraînement dure beaucoup plus longtemps et peut rencontrer des conditions différentes en mémoire, en performance ou dans le backend.

<a id="lesson-3"></a>
### Leçon

Les résultats de benchmark doivent être considérés comme des indicateurs de compatibilité plutôt que comme des recommandations de performance.

Les configurations finales doivent être validées par des exécutions réelles de données.

L'entraînement préalable réel a confirmé qu'une configuration pouvait s'exécuter correctement sur DirectML tout en étant impraticable pour un entraînement à grande échelle en raison de contraintes de performance ou de mémoire.

La validation DirectML doit donc utiliser des séries de formation avec données réelles bornées et `--max_steps` lorsque l'exécution sur l'ensemble complet des données n'est pas pratique.

Des mesures détaillées doivent être incluses dans [`directml_benchmarks.md`](directml_benchmarks.fr.md).

------------------------------------------------------------------------

<a id="training-measurement-can-introduce-major-synchronization-overhead"></a>
## La mesure de formation peut introduire une surcharge importante de synchronisation

Des mesures initiales de formation réelle ont été rapportées environ :

``` text
5.280 s / iteration
1.52 samples / second
304.81 effective tokens / second
```

Cela a initialement suggéré une durée estimée de pré-formation d'environ :

``` text
9.70 days / epoch
```

L'analyse de la performance M4 a montré que le chemin de mesure lui-même introduisait une surcharge importante de synchronisation entre DirectML et le CPU.

Après suppression de la synchronisation inutile dans le chemin critique de formation et de mesure, le même travail de référence `8 × 340` avec accumulation des gradients `8` a terminé un nouveau benchmark de 100 étapes avec :

``` text
Average iteration time:      0.482 s
Samples / second:            16.61
Effective tokens / second:   3341.37
Padded tokens / second:      5647.46
Estimated epoch duration:    21.24 h
Estimated epoch duration:    0.89 days
```

L'amélioration du temps d'itération était d'environ 10,95× sans réduire l'architecture du modèle, la taille physique du batch, la longueur de séquence ou la configuration d'accumulation des gradients.

<a id="lesson-4"></a>
### Leçon

L'instrumentation de performance peut fortement distordre les mesures DirectML lorsqu'elle force une synchronisation entre dispositif et CPU au sein de la boucle critique de formation.

Les études de performance doivent isoler :

``` text
Model computation
        +
Required synchronization
        +
Optimizer overhead
        +
Logging / measurement overhead
```

avant d'attribuer un temps d'exécution inattendument faible au backend DirectML lui-même.

Les valeurs de performance de référence doivent être conservées uniquement après avoir validé le chemin de mesure afin de s'assurer qu'il ne introduit pas une surcharge importante de synchronisation.

------------------------------------------------------------------------

<a id="regression-configuration"></a>
# Configuration de régression

Un modèle MiniMind léger est utilisé pour une validation rapide de la compatibilité DirectML :

``` text
hidden_size = 128
num_hidden_layers = 2
```

Cette configuration a été utilisée pour valider l'ensemble du flux :

``` text
Training
   ↓
Checkpoint
   ↓
Checkpoint loading
   ↓
Evaluation
   ↓
Text generation
```

<a id="decision-4"></a>
### Décision

La configuration `128 / 2-layer` est conservée comme configuration de régression légère pour DirectML.

Elle est destinée à la validation de compatibilité plutôt qu'à des tests de performance représentatifs du modèle.

------------------------------------------------------------------------

<a id="validation-principles"></a>
# Principes de validation

<a id="test-backend-compatibility-independently"></a>
## Valider la compatibilité du backend de manière indépendante

La validation de l'exécution DirectML doit être effectuée de manière indépendante des problèmes liés à d'autres modèles ou points de sauvegarde.

Une panne ne doit pas être automatiquement attribuée à DirectML.

Des vérifications pertinentes incluent :

``` text
Device placement
Model architecture
Checkpoint compatibility
Dataset validity
Backend operation support
```

------------------------------------------------------------------------

<a id="prefer-real-training-validation"></a>
## Privilégier la validation avec formation réelle

Des tests synthétiques et de courte durée sont utiles pour le développement et les tests de régression.

Ils ne remplacent pas la validation avec formation réelle.

La stratégie de validation doit donc combiner :

``` text
Unit tests
    +
Smoke tests
    +
Compatibility benchmarks
    +
Real training runs
```

Chaque couche répond à une question différente de compatibilité.

------------------------------------------------------------------------

<a id="documentation-responsibilities"></a>
# Responsabilités de la documentation

La documentation du projet est intentionnellement séparée par objectif :

``` text
roadmap.md
    → planned milestones and remaining work

update_log.md
    → chronological development history

project_memory.md
    → durable architectural decisions and lessons

directml_issues.md
    → current limitations plus problems, causes, solutions, and decisions

directml_benchmarks.md
    → experimental compatibility, performance, and final M5 validation results

development-tools.md
    → development and validation utilities

directml_audit.md
    → automatically generated compatibility audit
```

Cette séparation doit être conservée au fur et à mesure que le projet évolue.

<a id="m4-final-decisions"></a>
# Décisions finales M4

<a id="directml-fp16-requires-explicit-numerical-handling"></a>
## DirectML FP16 nécessite un traitement numérique explicite

Le chemin de formation final M4 utilise DirectML FP16 avec :

``` text
Static loss scale = 1024
AdamW epsilon     = 1e-4
```

La valeur par défaut de l'épsilon AdamW `1e-8` a provoqué une formation non finie dès la première mise à jour de l'optimiseur. L'inspection des gradients a montré des gradients finis avant l'étape de l'optimiseur, donc l'échec a été isolé à la phase de mise à jour.

<a id="lesson-5"></a>
### Leçon

Les paramètres de l'optimiseur FP16 DirectML doivent être validés indépendamment des hypothèses de précision mixte CUDA.

L'échelle de perte statique et un epsilon AdamW sécurisé pour DirectML sont centralisés dans des outils communs de formation plutôt que dupliqués dans chaque algorithme de formation.

------------------------------------------------------------------------

<a id="sustained-validation-is-required-after-the-first-optimizer-step"></a>
## Une validation durable est requise après la première étape d'optimiseur.

Un run de compatibilité borné doit traverser une frontière d'actualisation d'optimiseur pour fournir une validation FP16 utile.

La matrice de compatibilité finale a donc utilisé un run borné à 9 étapes avec accumulation des gradients `8`.

Pour une viabilité pratique, cela a été complété par un run de pré-formation réel `1000`.

Le run de référence durable s'est terminé avec :

``` text
batch_size = 8
max_seq_len = 340
accumulation_steps = 8
dtype = float16
loss_scale = 1024
AdamW eps = 1e-4
```

<a id="lesson-6"></a>
### Leçon

La validation doit distinguer entre :

``` text
Immediate execution
        ↓
First optimizer update
        ↓
Post-update execution
        ↓
Sustained training
```

Le simple passage de la première étape est insuffisant.

------------------------------------------------------------------------

<a id="fp16-improves-both-throughput-and-memory-headroom"></a>
## L'FP16 améliore à la fois la vitesse de traitement et la marge mémoire.

La mesure finale du benchmark FP16 de référence a été d'environ :

``` text
0.482 s / iteration
16.61 samples / second
3341.37 effective tokens / second
```

La surveillance de la mémoire dédiée à la GPU a mesuré un pic d'environ :

``` text
7,769.62 MB
```

par rapport à environ `11,623.12 MB` dans la mesure précédente en FP32, une réduction d'environ `33.15%`.

<a id="decision-5"></a>
### Décision

La base pratique DirectML pour le matériel testé est FP16 plutôt que FP32.

------------------------------------------------------------------------

<a id="use-one-explicit-cross-trainer-smoke-runner"></a>
## Utiliser un run de fumée unique entre traiteurs

Les principaux traiteurs DirectML FP16 utilisent une exécution bornée `--max_steps` et peuvent être validés séquentiellement par :

``` text
tests/test_all_trainers.py
```

Le run de fumée M4 a été validé pour :

``` text
Pretrain
Full SFT
LoRA
Distillation
GRPO
Agent RL
PPO
```

DPO reste partie de la chaîne de validation précédente mais n'est pas inclus dans le résultat spécifique du run de fumée M4 FP16.

<a id="decision-6"></a>
### Décision

Une validation de fumée lourde DirectML est exécutée explicitement avec Python et est maintenue séparée du workflow léger normal `pytest -q`.

------------------------------------------------------------------------

<a id="m4-practical-viability-conclusion--superseded"></a>
## Conclusion de viabilité pratique M4 — Supprimée

À la fin de M4, la configuration DirectML MiniMind testée était considérée comme pratiquement viable du point de vue de l'exécution, de la vitesse, de la mémoire et de la couverture des tests de fumée. Une validation complète du modèle de formation ultérieure a supprimé cette conclusion.

Cette conclusion repose sur la combinaison d'une vitesse corrigée, d'une utilisation réduite de la mémoire FP16, d'une stabilité durable sur 1000 étapes et d'une validation réussie de la fumée entre traiteurs.

La conclusion historique de M4 était spécifique au matériel et à la configuration validés. Elle ne doit pas être interprétée comme preuve que le chemin de formation DirectML final a produit un modèle acceptable.


------------------------------------------------------------------------

<a id="directml-moe-requires-a-compatibility-routing-path"></a>
## DirectML MoE nécessite un chemin de routage de compatibilité

Le chemin sparse MoE en amont repose sur des opérations de routage qui déclenchent un comportement de dispersion non pris en charge sur le backend DirectML testé.

L'architecture conservée est :

``` text
CPU / CUDA → original sparse MoE routing
DirectML   → scatter-free compatibility routing
```

Le chemin DirectML calcule tous les experts et les combine à l'aide des poids de routage sélectionnés.

<a id="lesson-7"></a>
### Leçon

La compatibilité du backend peut nécessiter une mise en œuvre locale en cas de non-support d'une opération sparse en amont.

Cette mise en œuvre doit rester isolée au niveau DirectML afin de préserver le comportement des CPU et CUDA en amont.

------------------------------------------------------------------------

<a id="agent-rl-requires-windows-and-fp16-specific-compatibility-handling"></a>
## L'apprentissage par renforcement d'agent nécessite un traitement spécifique de compatibilité Windows et FP16

Les workers DataLoader sous Windows exigent que l'Agent `collate_fn` soit défini au niveau du module.

La validation FP16 DirectML a également montré que le comportement de rollout peut rester fini alors que le recalcul complet de la politique/du référentiel masqué devient non fini.

Pour les lots d'agents droite-paddés validés, le masque de séquence complète problématique est omis lors du recalcul de la politique/du référentiel, tandis que les calculs numériquement sensibles de log-probabilité et de ratio d'apprentissage par renforcement utilisent FP32.

<a id="lesson-8"></a>
### Leçon

La stabilité lors de la génération ne garantit pas que le recalcul de la séquence complète suive un chemin de fond aussi stable.

Les deux chemins doivent être validés de manière indépendante.

------------------------------------------------------------------------

<a id="final-cross-trainer-m4-validation"></a>
## Validation finale du Cross-Trainer M4

Le runner de fumée final a été complété :

``` text
All trainer smoke tests passed
Passed: 9/9
```

Le runner couvre :

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

DPO reste validé précédemment mais en dehors de ce runner spécifique.

<a id="decision-7"></a>
### Décision

Le résultat `9/9` est la référence finale du cross-trainer M4.

------------------------------------------------------------------------

<a id="m5-training-execution"></a>
# Exécution de la formation M5

Le workflow complet maintient les configurations du formateur en amont séparées de la base de validation DirectML durable.

La base de pré-formation validée M4 reste :

``` text
batch_size = 8
max_seq_len = 340
accumulation_steps = 8
```

Pour la formation complète, `train_all.ps1` utilise les valeurs cibles spécifiques au formateur documentées dans [`training_commands.md`](training_commands.fr.md). Les étapes MoE DirectML conservent une taille de batch physique conservatrice.

Les étapes de formation longue actualisent leur point de restauration le plus récent selon les valeurs `--save_interval` spécifiques au formateur documentées dans [`training_commands.md`](training_commands.fr.md).

Une étape interrompue peut être reprise à partir de ce point de restauration avec :

``` text
--from_resume 1
```

<a id="decision-8"></a>
### Décision

Les paramètres par défaut du formateur en amont ne doivent pas être présentés comme des résultats de validation DirectML durables. Des points de restauration périodiques sont conservés pour les exécutions de formation longues.

------------------------------------------------------------------------

<a id="dense-directml-pretraining-uses-fp32-master-weights"></a>
## La pré-formation Dense DirectML utilise des poids maîtres en FP32

Les diagnostics des points de restauration à long terme ont montré que le chemin d'optimiseur pur FP16 DirectML initial pouvait rester fini tout en produisant un modèle dégénéré par copie auto.

La pré-formation Dense conserve maintenant le calcul du modèle en FP16 mais effectue les mises à jour de l'optimiseur via des poids maîtres en FP32 :

``` text
FP16 compute
    ↓
FP32 gradient unscale
    ↓
AdamW FP32 update
    ↓
Copy master weights back to FP16 model
```

Une étude ultérieure sur le cross-entropy DirectML a montré que la réduction par défaut avec `ignore_index=-100` normalisait incorrectement les lots paddés. Le premier remède explicite `reduction="sum" / valid_tokens` a été ensuite démontré incorrect sur le chemin DirectML FP16 testé.

L'implémentation conservée calcule la perte par token `reduction="none"` et effectue la moyenne des tokens valides en FP32. Sur le test de référence avec un batch de 32, la perte du modèle `6.30274916` correspondait exactement à la calcul par token DirectML et restait approximativement dans la marge `0.003` par rapport à la référence CPU FP32 `6.30530691`.

Les exécutions précédentes à l'étape 100 et à l'étape 1000 restent une preuve utile de validation FP32-maître, mais elles datent de avant la correction finale de la réduction cross-entropy.

<a id="decision-9"></a>
### Décision

Des pertes finies ne sont pas suffisantes pour prouver une formation à précision mixte durable.

La pré-formation Dense DirectML conserve les poids maîtres en FP32 dans les points de restauration afin de préserver la précision de l'optimiseur lors des interruptions. Les anciens points de restauration sans poids maîtres ne doivent pas être utilisés avec le nouveau chemin.

La perte causale du modèle DirectML doit utiliser une entropie par token suivie d'une moyenne en FP32 sur les tokens valides non remplies. Ni la réduction par défaut du backend ni l'approche testée en FP16 `sum / valid_tokens` ne sont conservées.



------------------------------------------------------------------------

<a id="intermediate-dense-pretraining-completed-epoch-1"></a>
## Pré-entraînement dense intermédiaire terminé, Épisode 1

Un parcours intermédiaire Dense DirectML a été redémarré à partir de zéro, en utilisant les paramètres d'entraînement supérieurs, y compris le physique `batch_size = 32`, les poids maîtres en FP32 et l'approche initiale de perte `reduction="sum" / valid_tokens`.

L'année complète a été terminée avec succès.

Un test de cohérence des points d'arrêt sur le modèle de l'année 1 a confirmé :

``` text
Model loss:                  6.48799419
Manual token loss:           6.48799419
Difference:                  0.0000000000
Samples 0–255 global loss:   6.1017
Top-1 accuracy:              15.07%
Top-1 repeat rate:            0.83%
Mean entropy:                 5.3160
```

Par rapport au point d'arrêt corrigé à l'étape 1000 (`loss 6.7785`, Top-1 `6.76%`, répétition `0.76%`), l'année complète montre une amélioration significative de la précision sur le prochain token sans retour de la crise historique de copie auto.

Un test ultérieur spécifique à la réduction a montré que cette exécution utilisait encore une perte DirectML incorrecte `reduction="sum" / valid_tokens`. Ce point d'arrêt est donc conservé uniquement pour une investigation et n'est pas la base d'entraînement finale.

<a id="decision-10"></a>
### Décision

La référence de qualité de l'entraînement dense final doit être produite à partir de zéro, avec des poids maîtres en FP32 et la perte conservée `reduction="none"` → moyenne en FP32 sur les tokens valides.

L'année intermédiaire terminée montre encore que le physique `batch_size = 32` est exécutable à travers une année complète sur le chemin mémoire actuel ; les observations antérieures de surcharge de mémoire restent des résultats historiques d'un parcours d'entraînement antérieur.

------------------------------------------------------------------------

<a id="final-directml-loss-path-completed-epoch-1"></a>
## Parcours DirectML de perte dense terminé, Épisode 1

Le parcours intermédiaire DirectML d'entraînement dense a terminé son premier épisode complet à partir de zéro, avec le calcul du modèle en FP16, les poids maîtres en FP32, l'entropie par token (`reduction="none"`) et la moyenne en FP32 sur les tokens valides.

Deux régions diagnostiques indépendantes de 256 échantillons ont produit des pertes d'entraînement/manuelles de `6.0612 / 6.0637` et `6.0177 / 6.0201`, avec une précision Top-1 de `13.71%` et `13.80%`, des taux de répétition de `0.81%` et `0.71%`, et une entropie moyenne de `5.3487` et `5.3436`.

Les pertes du modèle et les pertes indépendantes des tokens ont été alignées dans une marge d'environ `0.0025` après un épisode complet, et la crise historique de copie auto n'est pas revenue.

<a id="decision-11"></a>
### Décision

Le parcours d'entraînement dense complet en FP16, avec des poids maîtres en FP32 et une moyenne par token en FP32 valide, est désormais le parcours validé d'entraînement dense complet DirectML.

La même exécution corrigée a ensuite terminé l'épisode 2. Une évaluation plus large de 4 096 échantillons sur quatre régions de données a maintenu les pertes d'entraînement/manuelles en alignement étroit, avec des taux de répétition Top-1 entre `0.56%` et `0.96%`. Un contrôle qualitatif de génération a également montré qu'aucune récurrence de la crise historique de répétition de token n'était apparue.

<a id="final-decision"></a>
### Décision finale

Le point d'arrêt terminé de l'épisode 2 n'est pas conservé comme base d'entraînement dense finale pour le Full SFT. Bien que les diagnostics numériques soient raisonnables, la génération qualitative reste incohérente. Le parcours d'entraînement DirectML est donc abandonné et le prochain travail d'entraînement passe à ROCm.
------------------------------------------------------------------------

<a id="cuda-mixed-precision-and-directml-use-different-mechanisms"></a>
## Les mécanismes d'entraînement CUDA et DirectML sont différents

Le parcours d'entraînement CUDA supérieur conserve les paramètres du modèle en FP32 tout en utilisant l'auto-cast BF16 pour les opérations de calcul compatibles.

Conceptuellement :

``` text
Upstream CUDA:
FP32 model parameters
        ↓
BF16 autocast compute
        ↓
AdamW on FP32 parameters
```

La pile actuelle de DirectML ne fournit pas un chemin équivalent validé d'automatisation BF16. Une expérience directe avec un tenseur BF16 a échoué avec `Invalid or unsupported data type BFloat16`.

L'approximation conservée de DirectML est donc :

``` text
DirectML:
FP16 model compute
        ↓
FP32 master parameters
        ↓
AdamW FP32 update
        ↓
copy FP32 master → FP16 model
```

<a id="lesson-9"></a>
### Leçon

Les poids maîtres en FP32 ne sont pas une copie arbitraire. Ils fournissent la précision FP32 de l'état/paramètre de l'optimiseur que CUDA obtient naturellement en gardant les paramètres réels du modèle en FP32 tout en effectuant l'automatisation du calcul.

------------------------------------------------------------------------

<a id="fast-direct-fp16-re-test-confirms-master-weights-are-required"></a>
## Test rapide de Direct FP16 confirme que les poids maîtres sont nécessaires

Une nouvelle expérience de type upstream a supprimé les poids maîtres en FP32 et a appliqué AdamW directement aux paramètres du modèle en FP16.

Les diagnostics des points d'arrêt sur les échantillons `0–255` ont évolué comme suit :

``` text
step 1000 → loss 9.2573, Top-1 0.31%, repeat 0.29%
step 2000 → loss 8.3102, Top-1 0.92%, repeat 1.20%
step 5000 → loss 7.3196, Top-1 1.80%, repeat 39.24%
```

Le taux de répétition des données réelles a resté approximativement `0.29%`.

La diminution de la perte a donc masqué une dégradation forte du comportement de prédiction. Cela est cohérent avec les preuves précédentes sur la longue durée indiquant que les mises à jour directes en FP16 d'AdamW ne sont pas fiables pour une pré-entraînement prolongé dans DirectML.

<a id="decision-12"></a>
### Décision

Le chemin de précision Dense conservé de DirectML reste :

``` text
FP16 forward / backward
        ↓
FP32 gradient conversion and unscale
        ↓
FP32 gradient clipping
        ↓
AdamW on FP32 master weights
        ↓
FP32 master → FP16 model synchronization
```

Les travaux sur les performances doivent préserver cette architecture sauf si une alternative future est validée de manière indépendante avec des diagnostics des points d'arrêt.

------------------------------------------------------------------------

<a id="limit-diagnostic-cpu-synchronization-to-logging-intervals"></a>
## Limitez la synchronisation CPU pour les diagnostics à des intervalles de journalisation.

Les poids maîtres en FP32 et la synchronisation maître-modèle font partie du chemin requis de l'optimiseur et restent sur le dispositif DirectML.

En revanche, les lectures scalaires de diagnostic telles que :

``` text
loss.cpu().item()
grad_norm.cpu().item()
```

ne sont pas nécessaires à chaque étape d'entraînement et imposent une synchronisation entre l'exécution asynchrone de DirectML et le CPU.

Une variante optimisée du traineur effectue donc ces lectures scalaires uniquement aux frontières `log_interval` tout en conservant l'ajustement des gradients et les mises à jour des poids maîtres en FP32 inchangés à chaque étape de l'optimiseur.

<a id="decision-13"></a>
### Décision

Utilisez `log_interval` comme fréquence normale pour les vérifications de valeurs finies côté CPU et le journalisation des scalaires.

Une comparaison d'étapes `1000` a mesuré :

``` text
Stable trainer:     2003.79 s total, 2.0038 s / step
Log-Sync trainer:   1930.75 s total, 1.9307 s / step
Speedup:             1.038×
Time reduction:      3.65%
```

Le bénéfice en termes de débit est donc validé pour la configuration testée. Une validation des points d'arrêt est encore requise avant que le traineur synchronisé par journalisation ne devienne la référence finale.

------------------------------------------------------------------------

<a id="m5-final-diagnosis-and-backend-decision--2026-09-24"></a>
# M5 — Diagnostic final et décision sur le backend — 2026-09-24

La preuve finale sépare la qualité des points d'arrêt du fonctionnement du backend.

Sur les mêmes échantillons d'évaluation `1,000` (`204,327` tokens valides), sous ROCm :

```text
DirectML-trained checkpoint: loss 6.880233, perplexity 972.8529
ROCm-trained checkpoint:     loss 1.846587, perplexity   6.3382
```

Le même point d'arrêt entraîné par DirectML a également révélé une différence spécifique au niveau de la précision du backend :

```text
same batch, 873 valid tokens

DirectML FP16: 5.733902
ROCm FP16:     6.230469

DirectML FP32: 6.231627
ROCm FP32:     6.231628
```

DirectML en FP16 sans SDPA a produit des logits NaN. En FP32, DirectML et ROCm sont en accord à environ `1e-6`.

<a id="final-lesson"></a>
### Dernière leçon

Une perte d'entraînement finie, un succès dans le chargement/stockage des points d'arrêt et même des calculs de perte internes cohérents ne sont pas suffisants pour valider le backend. Le point d'arrêt doit être évalué de manière indépendante, et l'équivalence du backend en précision mixte doit être testée directement lorsque le comportement numérique est suspect.

<a id="final-decision-1"></a>
### Dernière décision

- Arrêtez l'utilisation de DirectML pour l'entraînement de MiniMind.
- Attribuez le blocage restant à la chaîne numérique FP16 de DirectML testée, et non à la charge des points d'arrêt, au tokenizer, à l'ensemble de données ou à l'inference générale.
- Conserver le code DirectML et les preuves comme une étude de faisabilité terminée.
- Fermer l'étude de formation DirectML ; les travaux futurs sur ROCm sont suivis séparément.
