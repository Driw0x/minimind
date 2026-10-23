[English](directml_benchmarks.md) | [Français](directml_benchmarks.fr.md) | [简体中文](directml_benchmarks.zh-CN.md)

[README du projet](../../README.fr.md)

<a id="minimind-----directml-benchmarks"></a>
# MiniMind --- Benchmark DirectML

> **Archive historique DirectML — mis à jour le 2026-09-24**
>
> DirectML n'est plus le backend actif pour le training de MiniMind. La validation finale M5 a isolé une divergence numérique spécifique à la chaîne de traitement DirectML FP16. Les résultats historiques ci-dessous sont conservés pour la reproductibilité et comme référence technique ; le développement ROCm est documenté séparément.

Ce document enregistre des tests d'compatibilité et de performance expérimentaux pour MiniMind.

L'objectif est d'identifier des configurations pratiques de training pour le matériel DirectML cible.

Ces résultats décrivent l'environnement testé et ne doivent pas être interprétés comme des limites universelles de DirectML.

------------------------------------------------------------------------

<a id="batch-size-and-sequence-length"></a>
# Taille du batch et longueur de séquence

<a id="motivation"></a>
## Motivation

Les identifiants `batch_size` et `max_seq_len` influencent directement les besoins en mémoire de training et le temps d'exécution.

Augmenter `batch_size` permet de traiter simultanément plus de séquences.

Augmenter `max_seq_len` augmente le nombre de tokens traités par séquence et entraîne une augmentation significative du coût d'attention des transformateurs.

Conceptuellement :

``` text
batch_size ↑
     +
max_seq_len ↑
     ↓
Higher memory usage
Higher computation cost
Longer training steps
```

Les paramètres par défaut de MiniMind en amont ne peuvent donc pas être automatiquement considérés comme pratiques sur le matériel DirectML.

------------------------------------------------------------------------

<a id="methodology"></a>
# Méthodologie

Un benchmark d'compatibilité a été introduit pour tester plusieurs combinaisons de :

``` text
batch_size
max_seq_len
```

Chaque configuration effectue une courte session de training réel plutôt qu'une session complète de training.

Les résultats sont classés comme suit :

``` text
PASS
FAIL
TIMEOUT
```

Cela permet d'identifier avant même de lancer la pipeline de training complet des configurations instables ou extrêmement lentes.

------------------------------------------------------------------------

<a id="initial-pretraining-benchmark"></a>
# Benchmark initial de pré-entraînement

Le benchmark initial de `train_pretrain.py` a produit :

    Batch size   Sequence length Result
  ------------ ----------------- ---------------------
             1                64 PASS
             1               128 PASS
             1               256 PASS
             1               340 FAIL (`0xC0000409`)
             2               128 PASS
             2               256 PASS
             2               340 PASS
             4               128 PASS
             4               256 TIMEOUT
             8               128 TIMEOUT

`2 × 256` a été initialement sélectionné comme configuration conservative pour valider le reste de la pipeline de training.

Ces résultats s'appliquent spécifiquement à la configuration de pré-entraînement testée et ne doivent pas être généralisés automatiquement à tous les entraîneurs de MiniMind.

------------------------------------------------------------------------

<a id="unexpected-result"></a>
# Résultat inattendu

Le benchmark initial a produit une observation apparemment contradictoire :

``` text
1 × 340 → FAIL
2 × 340 → PASS
```

Au début, cela semblait indiquer que la stabilité de DirectML n'était pas monotone en fonction de la taille du batch et de la longueur de séquence.

Cependant, l'ordinateur a été découvert exposer à la fois :

``` text
Integrated GPU
Dedicated GPU
```

et le choix explicite de GPU physique n'avait pas encore été pleinement contrôlé.

L'observation ne peut donc pas être interprétée de manière sûre comme un comportement inhérent de DirectML en fonction de la taille du batch ou de la longueur de séquence.

Elle reste enregistrée car elle a aidé à identifier le problème de sélection des dispositifs multi-GPU.

------------------------------------------------------------------------

<a id="multi-gpu-impact-on-benchmarking"></a>
# Impact des GPU multiples sur le benchmarking

Les résultats de benchmark ne sont comparables que si l'environnement d'exécution reste constant.

En particulier :

``` text
Same model
    +
Same trainer
    +
Same DirectML adapter
    +
Same benchmark conditions
    ↓
Comparable results
```

L'adaptateur DirectML physique doit donc être sélectionné explicitement avant de définir les limites de performance de référence.

Les résultats obtenus sur la GPU intégrée et la GPU dédiée ne doivent pas être combinés dans une seule table de compatibilité.

------------------------------------------------------------------------

<a id="reference-benchmark"></a>
# Benchmark de référence

Après avoir sélectionné explicitement la GPU dédiée, le benchmark de compatibilité a été relancé sur :

``` text
Device: directml:1
```

La matrice de compatibilité résultante était :

    Batch size   Sequence length Result
  ------------ ----------------- --------
             1                64 PASS
             1               128 PASS
             1               256 PASS
             1               340 PASS
             2               128 PASS
             2               256 PASS
             2               340 PASS
             4               128 PASS
             4               256 PASS
             4               340 PASS
             8               128 PASS
             8               256 PASS
             8               340 PASS
            16               128 PASS
            16               256 PASS
            16               340 PASS
            32               128 PASS
            32               256 FAIL

Le benchmark a cessé de tester des longueurs de séquence plus grandes pour une taille de batch `32` après avoir échoué à `32 × 256`.

Cela établit une plage de compatibilité validée significativement plus large sur la GPU dédiée sélectionnée explicitement que celle suggérée par le benchmark initial. En particulier, toutes les configurations testées jusqu'à `16 × 340` ont réussi, tandis que `32 × 128` a réussi et `32 × 256` n'a pas réussi.

Le benchmark initial reste utile comme données historiques de développement car il documente le comportement observé avant que la sélection physique de la GPU ne soit contrôlée.

------------------------------------------------------------------------

<a id="benchmark-vs-real-training"></a>
# Benchmark vs Formation réelle

Une configuration passant le benchmark de compatibilité n'est pas garantie de rester stable pendant une formation complète.

Le benchmark ne déclenche qu'un travail de formation court. Son objectif est d'identifier les configurations qui échouent immédiatement ou sont clairement incompatibles avec l'environnement DirectML testé.

Une formation complète peut se comporter différemment car elle dure beaucoup plus longtemps et peut rencontrer des conditions mémoire et d'exécution différentes.

Par conséquent :

``` text
Benchmark PASS
    ≠
Guaranteed full-training stability
```

Les résultats du benchmark doivent être considérés comme des indicateurs de compatibilité plutôt que comme des configurations garanties sans risque de formation.

Les derniers `batch_size` et `max_seq_len` doivent toujours être validés par une formation réelle.

------------------------------------------------------------------------

<a id="configuration-selection"></a>
# Sélection de la configuration

La configuration finale de DirectML doit équilibrer :

``` text
Stability
    +
Training speed
    +
Memory usage
```

plutôt que de simplement sélectionner la configuration la plus grande qui démarre avec succès.

Le benchmark doit être répété lorsque des conditions d'exécution pertinentes changent, y compris :

-   GPU physique ;
-   taille du modèle ;
-   entraîneur ;
-   environnement DirectML/PyTorch ;
-   autres modifications ayant un impact significatif sur les besoins en mémoire ou en calcul.

------------------------------------------------------------------------

<a id="real-training-performance"></a>
# Performance de la formation réelle

Le fait de passer le benchmark de compatibilité ne signifie pas qu'une configuration est stable ou efficace pour une formation réelle.

Des benchmarks de pré-formation réels ont donc été effectués à l'aide du jeu de données de pré-formation MiniMind réel et du modèle de pré-formation de taille complète.

L'environnement testé utilisait :

-   Appareil : `directml:1`
-   Représentation de l'appareil dans PyTorch : `privateuseone:1`
-   Données : `pretrain_t2t_mini.jsonl`
-   Échantillons du jeu de données : `1,270,238`
-   Taille cachée : `768`
-   Couche cachée : `8`
-   Paramètres du modèle : environ `63.91M`
-   Longueur maximale de séquence : `340`
-   Étapes d'accumulation des gradients : `8`

Le benchmark effectue des passes avant, des passes arrière, des coupes des gradients et des étapes de l'optimiseur AdamW.

------------------------------------------------------------------------

<a id="real-training-stability"></a>
## Stabilité de l'entraînement réel

L'entraînement sur données réelles a révélé des limites de stabilité invisibles dans le benchmark de compatibilité court.

<a id="batch-size-32"></a>
### Taille du batch 32

Configuration :

    batch_size = 32
    max_seq_len = 340

L'entraînement a commencé avec succès mais a échoué lors de la deuxième passe arrière en raison de mémoire GPU insuffisante.

Résultat :

    32 × 340 → OOM

Cette configuration n'est donc pas considérée stable pour l'entraînement réel sur les matériel testé.

<a id="batch-size-16"></a>
### Taille du batch 16

Configuration :

    batch_size = 16
    max_seq_len = 340
    accumulation_steps = 8

Le benchmark a terminé le premier cycle d'accumulation des gradients et l'étape de l'optimiseur.

L'entraînement a ensuite échoué lors d'une passe arrière ultérieure en raison de mémoire GPU insuffisante.

Résultat :

    16 × 340 → OOM after first accumulation cycle

Cela démontre que la réussite de plusieurs étapes d'entraînement ne garantit pas une stabilité mémoire durable.

<a id="batch-size-8"></a>
### Taille du batch 8

Configuration :

    batch_size = 8
    max_seq_len = 340
    accumulation_steps = 8

Un benchmark d'entraînement réel limité à 100 étapes a été effectué avec succès sans erreur de mémoire GPU.

Les premiers 8 pas ont été utilisés comme phase de préchauffage, laissant 92 étapes mesurées.

Résultat :

    8 × 340 → PASS (100-step bounded run)

Ceci est actuellement la configuration la plus grande testée qui a réussi la validation d'entraînement réel limitée.

------------------------------------------------------------------------

<a id="initial-real-training-performance-measurement"></a>
# Mesure initiale de la performance d'entraînement réel

La mesure initiale d'entraînement réel de 100 pas a utilisé :

    Device:                   directml:1
    PyTorch device:           privateuseone:1
    Dataset samples:          1,270,238
    Batch size:               8
    Max sequence length:      340
    Gradient accumulation:    8
    Warmup steps:             8
    Total benchmark steps:    100
    Measured steps:           92

Performance mesurée :

    Average iteration time:   5.280 s
    Samples / second:         1.52
    Effective tokens / sec:   304.81
    Padded tokens / sec:      515.17
    Steps / epoch:            158,780
    Estimated epoch duration: 232.87 h
    Estimated epoch duration: 9.70 days
    Peak dedicated GPU memory: 11,623.12 MB

La mémoire dédiée GPU a été mesurée en dehors du cadre, en utilisant les compteurs de mémoire des processus GPU de Windows pendant le benchmark de référence.

La mémoire dédiée GPU maximale a atteint environ 11,62 Go.

L'entraînement a resté proche de 11,6 Go de mémoire dédiée GPU pendant une grande partie de l'exécution, indiquant que la configuration de référence fonctionne relativement près de la limite de VRAM disponible.

La mesure initiale suggérait que la pré-formation réelle de MiniMind était techniquement exécutable sur les matériel DirectML testé, mais extrêmement lente.

À la vitesse mesurée :

    1 epoch  ≈ 9.70 days
    2 epochs ≈ 19.4 days

Ces valeurs étaient des estimations initiales basées sur le premier benchmark borné. Elles ont été ultérieurement trouvées fortement affectées par les surcoûts de synchronisation dans le chemin de mesure et sont conservées ici comme résultats historiques M4.

------------------------------------------------------------------------

<a id="corrected-real-training-benchmark"></a>
# Benchmark de formation corrigé

Une investigation supplémentaire de M4 a montré que le chemin initial de mesure introduisait un surcoût significatif de synchronisation entre DirectML et le CPU.

Après suppression des synchronisations inutiles dans le chemin critique de formation et de mesure, le même charge de référence a été réexécuté sur 100 étapes.

Configuration de référence :

    Device:                   directml:1
    PyTorch device:           privateuseone:1
    Model dtype:              float16
    Loss scale:               1024.0
    AdamW epsilon:            0.0001
    Dataset samples:          1,270,238
    Batch size:               8
    Max sequence length:      340
    Gradient accumulation:    8
    Warmup steps:             8
    Total benchmark steps:    100
    Measured steps:           92

Performance mesurée corrigée :

    Average iteration time:   0.482 s
    Samples / second:         16.61
    Effective tokens / sec:   3341.37
    Padded tokens / sec:      5647.46
    Steps / epoch:            158,780
    Estimated epoch duration: 21.24 h
    Estimated epoch duration: 0.89 days

Le benchmark corrigé a réussi à compléter toutes les 100 étapes.

Comparé à la mesure initiale :

    Average iteration time: 5.280 s → 0.482 s
    Samples / second:       1.52    → 16.61
    Effective tokens / sec: 304.81  → 3341.37
    Estimated epoch:        9.70 d  → 0.89 d

L'amélioration du temps d'itération est d'environ 10,95×.

Les paramètres de l'architecture du modèle, du jeu de données, de la taille physique du batch, de la longueur de séquence et de la configuration d'accumulation des gradients n'ont pas été réduits pour obtenir cette amélioration.

Ce résultat démontre que l'estimation initiale de performance était dominée par les surcoûts de synchronisation d'exécution et de mesure, et non par le coût réel d'entraînement en état stable de DirectML seul.

L'adaptation CPU AdamW `aten::lerp.Scalar_out` reste présente, mais elle ne empêche pas le benchmark corrigé de maintenir des itérations sous seconde pour la configuration testée.

------------------------------------------------------------------------

<a id="compatibility-benchmark-vs-real-training"></a>
# Benchmark de compatibilité vs Formation réelle

Le benchmark de compatibilité et le benchmark de formation réelle servent à des fins différentes.

Le benchmark de compatibilité a montré :

    8 × 340  → PASS
    16 × 340 → PASS
    32 × 128 → PASS
    32 × 256 → FAIL

Cependant, la formation sur données réelles a montré :

    32 × 340 → OOM
    16 × 340 → OOM after first accumulation cycle
     8 × 340 → PASS for 100 bounded steps

La configuration `32 × 340` n'était pas incluse dans la matrice de compatibilité validée car les tests pour la taille de batch `32` se sont arrêtés après l'échec de `32 × 256`.

La différence importante est donc le comportement de `16 × 340`.

Il a passé le benchmark de compatibilité court mais a échoué lors de l'exécution réelle prolongée.

Cela confirme :

    Compatibility PASS
            ≠
    Sustained training stability

Les essais de compatibilité courts restent utiles pour détecter des pannes immédiates du backend et de la mémoire, mais les configurations finales doivent être validées à l'aide de données réelles et de plusieurs cycles d'accumulation de gradients.

------------------------------------------------------------------------

<a id="throughput-and-batch-size"></a>
# Débit et taille du batch

Les expériences de formation réelle montrent que la configuration la plus grande acceptée par un test de compatibilité court n'est pas nécessairement la configuration la plus pratique.

Les résultats de stabilité sur données réelles restent :

    32 × 340 → OOM
    16 × 340 → OOM after first accumulation cycle
     8 × 340 → PASS for 100 bounded steps

À ce stade de l'analyse M4, la configuration de référence `batch_size = 8` avec `max_seq_len = 340` et l'accumulation de gradients `8` était la préférée pour les expériences de formation réelle.

Après correction des surcoûts de synchronisation dans le chemin de mesure, cette configuration atteint approximativement :

    0.482 s / iteration
    16.61 samples / second
    3341.37 effective tokens / second

Ce choix repose sur la stabilité sur données réelles et sur le débit mesuré corrigé, et non sur la configuration maximale acceptée par le test de compatibilité.

------------------------------------------------------------------------

<a id="directml-optimizer-limitation"></a>
# Limitation de l'Optimiseur DirectML

Lors de la formation réelle, PyTorch signale un recours à DirectML pour :

    aten::lerp.Scalar_out

Cette opération est actuellement non prise en charge par le backend DirectML et recourt à l'exécution sur CPU lors de l'étape de l'optimiseur AdamW.

Ce recours ne bloque pas la formation, mais il peut introduire un surcoût de synchronisation et de performance.

Le benchmark de référence inclut ce recours et représente donc le comportement global observé des opérations de formation testées, et non une exécution pure sur GPU DirectML.

Le benchmark corrigé inclut toujours ce recours tout en maintenant une performance moyenne de `0.482 s/iteration`.

Ce recours reste donc une limitation connue de DirectML, mais le grand résultat initial de `5.280 s/iteration` ne doit pas être attribué uniquement à ce recours.

------------------------------------------------------------------------

<a id="intermediate-m4-performance-conclusion"></a>
# Conclusion intermédiaire sur les performances M4

L'analyse de formation réelle M4 établit plusieurs observations importantes.

Premièrement, les tests de compatibilité ne sont pas suffisants pour déterminer la stabilité de la formation prolongée.

Deuxièmement, la validation sur données réelles a confirmé que `8 × 340` avec l'accumulation de gradients `8` est stable pour la course de référence limitée à 100 étapes, tandis que des lots physiques plus grands entraînaient des échecs de mémoire GPU.

Troisièmement, la mesure initiale de `5.280 s/iteration` n'était pas représentative du coût réel de la formation en état stable. Le chemin de mesure a introduit un surcoût significatif de synchronisation entre DirectML et CPU.

Après correction de ce problème, la même configuration de référence a terminé un nouveau benchmark de 100 étapes avec :

    16.61 samples/s
    3341.37 effective tokens/s
    0.482 s/iteration

et une durée estimée d'un tour complet d'environ :

    21.24 hours
    0.89 days

Le résultat corrigé est environ 10,95 fois plus rapide en temps d'itération que la mesure initiale.

La formation à l'échelle complète est donc considérablement plus pratique que ce que les premières mesures M4 ont suggéré, bien que le benchmark de 100 étapes ne soit pas encore suffisant pour prouver une stabilité durable à l'échelle d'un époque complète.

À ce stade, la validation supplémentaire M4 a porté sur :

-   une stabilité de formation durable au-delà de la course bornée de 100 étapes ;
-   la validation du débit corrigé sur des courtes durées ;
-   l'observation continue du comportement de la mémoire GPU ;
-   l'évaluation finale de la faisabilité pratique de DirectML ;
-   la consolidation des limitations de performance restantes.

------------------------------------------------------------------------

<a id="final-m4-fp16-validation"></a>
# Validation finale M4 FP16

M4 s'est terminé par une chaîne de formation FP16 spécifique à DirectML conçue pour éviter l'instabilité numérique observée avec l'épsilon par défaut d'AdamW.

Les paramètres FP16 validés de DirectML sont :

``` text
Model dtype:              float16
Static loss scale:        1024
AdamW epsilon:            1e-4
Reference device:         directml:1
PyTorch representation:   privateuseone:1
```

L'utilisation de l'épsilon par défaut d'AdamW de `1e-8` a produit des valeurs non finies dès la première mise à jour de l'optimiseur. Un échelonnement statique de perte n'a pas résolu le problème. L'utilisation de `1e-4` pour l'AdamW de DirectML FP16 a stabilisé le chemin de l'optimiseur.

Le tableau de compatibilité DirectML FP16 a ensuite été relancé avec le modèle complet `768 / 8-layer`, l'accumulation de gradients `8` et une course bornée de 9 étapes afin que chaque configuration traverse la première frontière d'actualisation de l'optimiseur.

    Batch size   Sequence length Result
  ------------ ----------------- --------
             1                64 PASS
             1               128 PASS
             1               256 PASS
             1               340 PASS
             2               128 PASS
             2               256 PASS
             2               340 PASS
             4               128 PASS
             4               256 PASS
             4               340 PASS
             8               128 PASS
             8               256 PASS
             8               340 PASS
            16               128 PASS
            16               256 PASS
            16               340 PASS
            32               128 PASS
            32               256 PASS
            32               340 PASS

Ce tableau valide une exécution courte et bornée au-delà de la première frontière d'actualisation de l'optimiseur. Il ne signifie pas que chaque configuration large est adaptée à une formation durable. Les tests réels antérieurs restent pertinents : des lots physiques importants peuvent encore épuiser la mémoire VRAM lors d'une exécution prolongée.

------------------------------------------------------------------------

<a id="sustained-pretraining-validation"></a>
# Validation de la pré-formation durable

La validation finale durable M4 a utilisé :

``` text
Device:                 directml:1
Model dtype:            float16
Batch size:             8
Max sequence length:    340
Gradient accumulation:  8
Static loss scale:      1024
AdamW epsilon:          1e-4
Maximum steps:          1000
```

L'exécution a réussi tous les pas de formation `1000`.

Comme `1000` est divisible par le facteur d'accumulation `8`, l'exécution a effectué `125` mises à jour de l'optimiseur.

Aucune perte NaN ou Inf n'a été observée pendant l'exécution durable. Les pertes enregistrées ont resté finies jusqu'au pas `1000`.

L'estimation finale de la durée de formation était d'environ :

``` text
1360 minutes
22.67 hours / epoch
```

Ceci est cohérent avec l'estimation corrigée du benchmark de 100 étapes d'environ `21.24 hours / epoch`.

L'exécution durable confirme donc que le débit corrigé est représentatif au-delà du court benchmark de 100 étapes et que la configuration de référence FP16 DirectML sélectionnée reste numériquement stable à travers de nombreuses mises à jour de l'optimiseur.

------------------------------------------------------------------------

<a id="final-fp16-gpu-memory-measurement"></a>
# Mesure finale de la mémoire GPU FP16

L'observation de la mémoire GPU externe pendant l'exécution de référence FP16 finale a enregistré :

``` text
Peak dedicated VRAM:     7,769.62 MB
Median active VRAM:      7,751.02 MB
Average active VRAM:     7,218.87 MB
Typical active plateau:  ~7,751 MB
```

Comparé à la pointe de mémoire GPU de `11,623.12 MB` observée précédemment en FP32, le FP16 a réduit la pointe de mémoire dédiée GPU d'environ :

``` text
3,853.50 MB
33.15%
```

L'ensemble de charge de référence FP16 final utilise donc environ `7.59 GiB` de VRAM dédiée en tant que point culminant, au lieu d'environ `11.35 GiB` utilisé dans la mesure précédente en FP32.

------------------------------------------------------------------------

<a id="cross-trainer-smoke-validation"></a>
# Validation du cross-trainer Smoke

Le chemin FP16 DirectML et l'exécution bornée `--max_steps` ont été propagés dans les principaux flux d'entraînement utilisés par le lanceur de fumée M4 :

``` text
Pretrain
Full SFT
LoRA
Distillation
GRPO
Agent RL
PPO
```

Le lanceur consolidé `tests/test_all_trainers.py` a réussi à exécuter tous les tests de fumée inclus dans les entraîneurs.

DPO reste intégré à la chaîne d'entraînement précédemment validée MiniMind, mais il n'est pas inclus dans le résultat de ce lanceur de fumée M4 FP16.

------------------------------------------------------------------------

<a id="final-m4-conclusion--historical-superseded"></a>
# Conclusion finale M4 — Historique, Superséduite

À la fin de M4, les preuves disponibles sur l'exécution, la vitesse, la mémoire et les tests de fumée suggéraient que l'entraînement DirectML était pratiquement viable pour la configuration testée de MiniMind sur le matériel de référence. **Cette conclusion a été ultérieurement remplacée par une validation complète de la qualité du modèle d'entraînement.**

La base recommandée d'entraînement réel est :

``` text
Device:                 directml:1
Model dtype:            float16
Batch size:             8
Max sequence length:    340
Gradient accumulation:  8
Static loss scale:      1024
AdamW epsilon:          1e-4
```

Les preuves finales combinent :

``` text
Short compatibility tests
        +
100-step performance benchmark
        +
1000-step sustained pretraining validation
        +
GPU memory monitoring
        +
Cross-trainer smoke validation
```

À la phase M4, les limitations restantes du backend et les retours en mode CPU n'ont pas empêché l'exécution du cycle d'entraînement. Des vérifications ultérieures sur la qualité des points d'arrêt et la génération qualitative ont montré que l'exécution réussie et des métriques de perte raisonnables n'étaient pas suffisantes pour établir un comportement correct d'entraînement complet.


------------------------------------------------------------------------

<a id="moe-directml-validation"></a>
# Validation DirectML MoE

Le flux de fumée final M4 a été étendu pour couvrir la configuration MoE de MiniMind en plus du modèle dense.

Le modèle MoE `768 / 8-layer` testé rapporte environ :

``` text
Model Params: 198.42M-A63.94M
```

DirectML a révélé un problème de compatibilité dans la chaîne de routage des experts dispersés. Les opérations de routage concernées nécessitaient un comportement de dispersion non supporté de manière fiable par le chemin directeur/retour de DirectML.

Un chemin de routage sans dispersion spécifique à DirectML a donc été introduit. CPU et CUDA conservent le comportement de routage dispersé original.

Le chemin de compatibilité DirectML calcule tous les experts et combine leurs sorties en utilisant les poids de routage sélectionnés. Il s'agit d'un recours fonctionnel de compatibilité et ne doit pas être interprété comme une performance native de MoE dispersée.

Les workflows suivants ont été exécutés avec succès :

``` text
MoE Pretrain
MoE Full SFT
Dense student ← MoE teacher Distillation
```

Un autre entraînement préalable borné de 20 étapes MoE a également été limité.

Une comparaison rapide entre un modèle dense et un modèle MoE, sous des conditions de validation identiques, a produit un rapport d'exécution préliminaire d'environ :

``` text
MoE / Dense ≈ 1.158×
```

ou environ `15.8%` de temps supplémentaire dans cette expérience.

Cette valeur ne représente qu'une observation relative courte et n'est pas considérée comme une évaluation complète de la performance MoE.

------------------------------------------------------------------------

<a id="final-cross-trainer-smoke-result"></a>
# Résultat final du cross-trainer Smoke

Après l'ajout de la couverture dense et MoE et la résolution des problèmes de compatibilité restants liés à l'agent, le lanceur final de fumée DirectML a réussi à exécuter :

``` text
All trainer smoke tests passed
Passed: 9/9
```

Le lanceur final valide :

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

DPO reste intégré à la chaîne d'entraînement précédemment validée, mais n'est pas inclus dans ce lanceur spécifique de 9 tests.

------------------------------------------------------------------------

<a id="fp32-master-weight-and-cross-entropy-validation"></a>
# Validation du poids maître FP32 et de la croisée de l'entropie

Une investigation ultérieure sur la qualité des points d'arrêt a montré que le chemin d'optimiseur pur FP16 DirectML précédent pouvait rester fini tout en convergant vers un modèle dégénéré de copie auto-identique.

La pré-formation dense a été donc modifiée pour utiliser le calcul en FP16 avec des poids principaux en FP32 et des mises à jour en FP32 avec AdamW.

Une investigation séparée sur la perte a révélé deux problèmes de réduction DirectML sur le chemin FP16 testé :

1. `reduction="mean"` avec `ignore_index=-100` n'a pas normalisé uniquement sur les tokens valides ;
2. l'approche initiale `reduction="sum" / valid_tokens` a également échoué à reproduire la référence en FP32 sur CPU.

La comparaison décisive sur la réduction des logits a produit :

``` text
Batch 8:
sum / valid:             14.1078
none → FP32 valid mean:   6.6688
CPU FP32 reference:       6.6715

Batch 32:
sum / valid:             10.7772
none → FP32 valid mean:   6.3027
CPU FP32 reference:       6.3053
```

L'implémentation finale de la perte utilise donc :

``` python
token_loss = F.cross_entropy(
    x,
    y,
    ignore_index=-100,
    reduction="none",
)
valid = y != -100
loss = token_loss[valid].float().mean()
```

Après ce changement, la perte du modèle en lot de 32 correspondait exactement à la moyenne par token en FP32 de DirectML et restait dans une marge d'environ `0.003` par rapport à la référence en FP32 sur CPU :

``` text
Model outputs.loss:        6.30274916
DirectML none/FP32 mean:   6.30274916
CPU FP32 reference:        6.30530691
```

Les étapes précédentes à 100 et à 1000 ont validé la direction de l'optimiseur en FP32-master, mais elles ont été effectuées avant la correction finale de la réduction de la croisée de croisement et sont conservées comme résultats d'investigation intermédiaires plutôt que comme références finales de qualité de formation.

------------------------------------------------------------------------

<a id="intermediate-full-epoch-dense-pretraining-validation"></a>
# Validation intermédiaire de la pré-formation dense complète

Après le changement des poids principaux en FP32 et l'approche initiale `reduction="sum" / valid_tokens` de croisée de croisement, la pré-formation dense a été redémarrée à partir de zéro en utilisant les paramètres de pré-formation upstream, avec uniquement le chemin de précision/backend DirectML adapté :

``` text
Device:                 directml:1
Model dtype:            float16
Epochs:                 2
Batch size:             32
Learning rate:          5e-4
Max sequence length:    340
Gradient accumulation:  8
Hidden size:            768
Hidden layers:          8
Dense / use_moe:        0
```

L'année complète a été terminée avec succès.

Ceci est important car l'observation précédente `32 × 340 → OOM` a été obtenue avant l'établissement du chemin de formation DirectML final. Elle est conservée ci-dessus comme preuve historique, mais elle ne décrit plus l'implémentation actuelle corrigée.

La validation des points d'arrêt sur les échantillons `0–255` a produit :

``` text
Valid tokens:                 51,596
Mean batch loss:              6.0922
Token-weighted global loss:   6.1017
Minimum batch loss:           5.6996
Maximum batch loss:           6.4880
Top-1 accuracy:               15.07%
Top-1 repeat rate:             0.83%
Mean entropy:                  5.3160
```

Une comparaison directe entre la perte de formation du modèle et une perte normalisée manuelle par token sur le même lot a produit :

``` text
Model loss:   6.48799419
Manual loss:  6.48799419
Difference:   0.0000000000
MATCH
```

L'arrêt a montré une apprentissage utile sur le prochain token et un taux de répétition faible, mais la comparaison ultérieure de réduction a prouvé que cette exécution utilisait encore une perte incorrecte DirectML `reduction="sum" / valid_tokens` lors de la propagation en arrière.

Il est donc conservé comme preuve que le physique `batch_size = 32` peut exécuter une époque complète sur le chemin mémoire actuel, mais ce n'est pas une référence finale de qualité de formation et ne doit pas être utilisé comme point d'arrêt de base pour la chaîne de formation SFT.

La pré-formation dense finale doit être redémarrée à partir de zéro avec l'implémentation conservée `reduction="none"` → moyenne par token valide en FP32.

------------------------------------------------------------------------

<a id="final-loss-path--1200-step-validation"></a>
# Chemin final de perte — validation sur 1200 étapes

Le chemin final de perte DirectML a été validé jusqu'à l'étape `1200` en utilisant le calcul en FP16, des poids principaux en FP32, une croisée de croisement par token suivie d'une moyenne par token valide en FP32.

Deux régions indépendantes de données de 256 échantillons ont produit :

|Échantillons|Perte de formation|Perte manuelle|Différence|Top-1|Répétition|
|---|---:|---:|---:|---:|---:|
|0–255| 6.3752 | 6.3779 | 0.00264 | 8.66% | 1.96% |
|100000–100255| 6.3568 | 6.3595 | 0.00263 | 8.91% | 2.08% |

Les pertes de formation sur le chemin de formation et la perte calculée indépendamment par token restent alignées dans une marge d'environ `0.003`.

Comparée à l'étape `200`, la perte a diminué de l'ordre de `7.18` à `6.36`, tandis que la précision Top-1 a augmenté de l'ordre de `5.1%` à `8.8%`.

Le taux de répétition Top-1 restait bas par rapport à l'ancien échec de copie auto, et le taux de correction et de répétition Top-1 restait approximativement `0.02%` sur les deux régions d'évaluation.

Aucune récurrence de l'ancien échec de copie auto n'a été observée.

Ce résultat valide la dernière chaîne de calcul FP16 + poids principaux FP32 + perte moyenne FP32 par token au-delà des premiers tests courts sur un point d'arrêt initial. Une validation complète d'époque a été ensuite réalisée avec succès.

------------------------------------------------------------------------

<a id="final-loss-path--full-epoch-1-validation"></a>
# Chaîne finale de perte — Validation complète de l'époque 1

La dernière chaîne de pré-entraînement Dense DirectML a terminé son premier cycle complet en utilisant un calcul FP16, des poids principaux FP32, et une perte de croisement par token suivie d'une moyenne FP32 des tokens valides.

|Échantillons|Perte d'entraînement|Perte manuelle|Différence|Top-1|Répétition|Entropie|
|---|---:|---:|---:|---:|---:|---:|
|0–255| 6.0612 | 6.0637 | 0.00250 | 13.71% | 0.81% | 5.3487 |
|100000–100255| 6.0177 | 6.0201 | 0.00246 | 13.80% | 0.71% | 5.3436 |

Les pertes d'entraînement du chemin d'entraînement et calculées de manière indépendante restaient alignées à environ `0.0025` après un cycle complet.

Par rapport à l'étape `1200`, la perte d'entraînement a diminué de l'ordre de `6.36` à `6.02–6.06`, tandis que la précision Top-1 a augmenté de l'ordre de `8.8%` à `13.7–13.8%`.

Le taux de correction et de répétition Top-1 restait très bas (`0.06%` et `0.03%`), et l'ancien échec de copie auto n'a pas réapparu.

C'est la première validation complète d'un cycle d'entraînement réussi pour le chemin de pré-entraînement Dense DirectML conservé.

<a id="final-epoch-2-validation"></a>
## Validation finale de l'époque 2

La même exécution corrigée a terminé l'époque 2 avec succès. Une évaluation plus large a couvert 4 096 échantillons sur quatre régions de données :

|Région d'échantillon|Tokens valides|Perte d'entraînement|Perte manuelle|Top-1|Répétition|
|---|---:|---:|---:|---:|---:|
|0–1023| 205,439 | 6.3317 | 6.3343 | 12.23% | 0.69% |
|300000–301023| 116,790 | 6.3619 | 6.3645 | 10.57% | 0.56% |
|600000–601023| 92,088 | 5.9203 | 5.9288 | 13.22% | 0.96% |
|1000000–1001023| 261,529 | 6.0706 | 6.0731 | 12.94% | 0.73% |

Dans ces régions, les pertes d'entraînement et les pertes manuelles indépendantes restaient étroitement alignées et le taux de répétition restait en dessous de `1%`.

Un contrôle qualitatif final de génération a produit des sorties variées plutôt que le précédent échec de répétition des tokens. Le texte restait faible et fragmentaire, comme prévu avant la fine-tuning supervisé.

L'exécution corrigée de deux cycles d'entraînement Dense a été terminée numériquement, mais le modèle localement entraîné n'a pas produit une génération cohérente acceptable. Il est donc **ne pas conservé comme base d'entraînement finale pour le Full SFT**. Le chemin d'entraînement DirectML est fermé et le projet passe à ROCm.
------------------------------------------------------------------------

<a id="fast-directml-fp16-re-test"></a>
# Test rapide DirectML FP16

Un chemin d'entraînement préliminaire expérimental a été testé afin de déterminer si la mécanique des poids principaux FP32 pouvait être supprimée pour améliorer le débit DirectML.

Le chemin expérimental a conservé :

``` text
FP16 model parameters
FP16 forward / backward
Static loss scale = 1024
AdamW directly on FP16 parameters
```

et a intentionnellement supprimé les paramètres principaux FP32 et la copie des poids principaux vers le modèle.

Les diagnostics sur la même région d'évaluation `0–255` ont montré :

|Étape|Perte du chemin d'entraînement|Perte manuelle|Top-1|Répétition|Entropie|
|---:|---:|---:|---:|---:|---:|
| 1000 | 9.2573 | 9.2609 | 0.31% | 0.29% | 7.0694 |
| 2000 | 8.3102 | 8.3136 | 0.92% | 1.20% | 7.2550 |
| 5000 | 7.3196 | 7.3225 | 1.80% | 39.24% | 7.3674 |

Le taux de répétition des données réelles a resté approximativement `0.29%`.

Bien que la perte diagnostique ait amélioré entre les étapes `1000` et `5000`, le taux de répétition Top-1 a augmenté brusquement jusqu'à `39.24%`. À l'étape `5000`, environ `39.10%` des prédictions étaient des erreurs exclusivement de répétition.

Ceci reproduit le mode de défaillance à long terme déjà observé dans le chemin précédent à optimiseur pur FP16 : une perte finie et décroissante n'est pas une preuve suffisante que les mises à jour directes des paramètres en FP16 apprennent un modèle utilisable.

La trajectoire exacte de défaillance n'est pas identique à la précédente chute de copie auto-identique `93.48%`, mais les deux expériences montrent que les mises à jour directes d'AdamW sur les paramètres de modèle en FP16 ne sont pas conservées comme une voie fiable de pré-entraînement.

Le chemin FP32-master reste donc l'architecture de référence :

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

------------------------------------------------------------------------

<a id="stable-master-weight-synchronization-optimization"></a>
# Optimisation de synchronisation des poids maîtres stables

L'optimisation de performance se concentre désormais sur la préservation de l'architecture de poids maître FP32 validée tout en réduisant le surcoût de synchronisation diagnostique.

Le précédent entraîneur stable produisait des valeurs scalaires générées par DirectML sur le CPU plus fréquemment que nécessaire à l'entraînement, y compris des vérifications de perte finie sur chaque lot et des vérifications de norme de gradient sur les mises à jour de l'optimiseur.

Une variante optimisée de l'entraîneur maintient maintenant le chemin d'optimisation inchangé, mais effectue uniquement des lectures DirectML vers CPU pour les bornes `log_interval`.

Conceptuellement :

``` text
Between log points:
    FP16 forward / backward
    FP32 master update
    gradient clipping
    master → model synchronization
    no diagnostic .cpu().item()

At log_interval:
    read loss
    read latest gradient norm
    finite-value checks
    logging
```

Ce changement vise à réduire le surcoût de synchronisation sans supprimer les poids maîtres en FP32.

Une comparaison directe `1000` a ensuite été effectuée avec la même configuration d'entraînement pour les deux entraîneurs :

``` text
Steps:                    1000

Stable total:             2003.79 s
Log-Sync total:           1930.75 s

Stable time / step:       2.0038 s
Log-Sync time / step:     1.9307 s

Stable steps / second:    0.4991
Log-Sync steps / second:  0.5179

Speedup:                  1.038×
Time reduction:           3.65%
```

L'entraîneur optimisé a réduit le temps total d'environ `73.04 s` sur `1000` étapes tout en conservant le chemin d'optimisation des poids maîtres en FP32.

L'amélioration de la performance est donc validée pour la configuration testée. La validation de la qualité des points d'arrêt reste en attente avant que l'entraîneur synchronisé par journal ne remplace l'entraîneur stable comme référence finale d'entraînement.

------------------------------------------------------------------------

<a id="m5-final-directml-cross-backend-validation--2026-09-24"></a>
# Validation croisée DirectML entre backends — M5 Final — 2026-09-24

La comparaison finale a dépassé la perte d'entraînement dans le journal et a évalué les points d'arrêt résultants sur les mêmes données et backend.

Sur un sous-ensemble fixe de `1,000` échantillons (`204,327` tokens valides), les deux points d'arrêt ont été évalués sous ROCm :

|Point d'arrêt|Perte d'évaluation|Perplexité|Génération|
|---|---:|---:|---|
|Formé par DirectML| 6.880233 | 972.8529 |incohérent|
|Formé par ROCm| 1.846587 | 6.3382 |cohérent|

Un test sur un backend avec le même point d'arrêt a ensuite utilisé le point d'arrêt formé par DirectML sur un lot fixe (`873` tokens valides) :

|Chemine d'exécution|Perte|Résultat|
|---|---:|---|
|DirectML FP16| 5.733902 |fini mais divergent|
|ROCm FP16| 6.230469 |comportement de référence FP16|
|DirectML FP32| 6.231627 |correspond au FP16 de ROCm|
|ROCm FP32| 6.231628 |correspond à DirectML|
|DirectML FP16 sans SDPA|NaN|instable numériquement|
|ROCm FP16 sans SDPA| 6.230469 |stable|

La perte du modèle DirectML FP16 a également été comparée à une réévaluation de la croisement entropique en FP32 sur une CPU à partir de ses propres logits (`5.733902` vs `5.736305`). Les valeurs analogues ROCm étaient `6.230469` vs `6.231570`. Ainsi, la réduction de l'entropie des tokens valides conservés n'est pas la source finale de l'incidence : les logits produits par le parcours forward DirectML FP16 diffèrent de manière significative.

En FP32, DirectML et ROCm sont en accord à environ `1e-6`, y compris les statistiques de perte et de logits. Cela isole le comportement bloquant final à la chaîne numérique testée de DirectML FP16 plutôt qu'au format de checkpoint, au tokenizer, à l'ensemble de données ou à l'inférence générale de DirectML.

**Décision** : cette validation M5 ferme l'étude de faisabilité de DirectML. Le travail futur sur ROCm est documenté séparément.
