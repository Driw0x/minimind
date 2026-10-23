[English](roadmap.md) | [Français](roadmap.fr.md) | [简体中文](roadmap.zh-CN.md)

[README du projet](../../README.fr.md)

<a id="minimind-backend-roadmap"></a>
# Roadmap du backend MiniMind

> **Archive historique DirectML — mis à jour le 2026-09-24**
>
> DirectML n'est plus le backend actif pour le training de MiniMind. La validation finale M5 a isolé une divergence numérique spécifique à la chaîne de traitement DirectML FP16 testée. Les résultats historiques ci-dessous sont conservés pour la reproductibilité et comme référence technique ; le développement de ROCm est documenté séparément.

Ce document suit l'évaluation terminée de DirectML et le passage du training de MiniMind à ROCm.

Le projet distingue entre **compatibilité fonctionnelle** et **utilisabilité pratique**. Une étape de training peut s'exécuter correctement sur DirectML tout en restant impraticable pour un training réel du monde réel en raison de limitations de performance, de retours à l'CPU, de contraintes mémoire ou de surcoût du backend.

------------------------------------------------------------------------

<a id="m1-----directml-foundation-"></a>
## M1 --- Fondation DirectML ✅

Établir et valider l'environnement de base DirectML.

-   [x] Configuration d'un environnement Windows + DirectML
-   [x] Remplacer la dépendance PyTorch orientée CUDA par `torch-directml`
-   [x] Initialiser un dispositif DirectML
-   [x] Déplacer les modèles et les tenseurs de MiniMind vers DirectML
-   [x] Valider le passage en avant (forward pass)
-   [x] Valider le calcul de la perte (loss computation)
-   [x] Valider le passage en arrière (backward pass)
-   [x] Valider l'étape de l'optimiseur AdamW
-   [x] Ajouter des tests de compatibilité DirectML avec Pytest
-   [x] Valider l'installation à partir d'un environnement virtuel propre

------------------------------------------------------------------------

<a id="m2-----minimind-training-compatibility-"></a>
## M2 --- Compatibilité avec le training de MiniMind ✅

Adapter la pipeline de training de MiniMind afin que ses opérations de training s'exécutent sur DirectML.

-   [x] Vérifier le code spécifique à CUDA dans la pipeline de training
-   [x] Introduire la gestion du dispositif compatible DirectML
-   [x] Adapter les outils de précision mixte et les éléments spécifiques à CUDA lorsque nécessaire
-   [x] Exécuter un job de pré-training minimal sur DirectML
-   [x] Vérifier que la perte de training évolue correctement
-   [x] Enregistrer un point de sauvegarde de modèle
-   [x] Charger correctement un point de sauvegarde
-   [x] Exécuter une inférence à l'aide du point de sauvegarde formé

> M2 valide la **compatibilité fonctionnelle**, pas la performance pratique du training.

------------------------------------------------------------------------

<a id="m3-----full-training-pipeline-compatibility-"></a>
## M3 --- Compatibilité complète de la pipeline de training ✅

Valider les principales étapes de training de MiniMind avec DirectML.

-   [x] Valider le pré-training
-   [x] Valider l'entraînement supervisé (SFT)
-   [x] Valider l'entraînement LoRA
-   [x] Valider les autres étapes d'entraînement supportées par MiniMind
-   [x] Valider la compatibilité des points de sauvegarde entre les étapes de training
-   [x] Documenter les opérations DirectML non supportées ou partiellement supportées
-   [x] Valider la cohérence du dispositif à travers les étapes de training

> La pipeline de training est fonctionnellement compatible avec DirectML, mais cela ne garantit pas que le training à l'échelle complète soit pratiquement utilisable.

------------------------------------------------------------------------

<a id="m4-----directml-viability--performance-investigation-"></a>
## M4 --- Étude de la faisabilité et des performances de DirectML ✅

Déterminer si DirectML est pratiquement utilisable pour le training de MiniMind et identifier les principaux goulets de performance.

<a id="baseline-benchmarking"></a>
### Benchmark de référence

-   [x] Benchmark des combinaisons de taille de batch / longueur de séquence supportées
-   [x] Identifier les configurations qui échouent en raison de limitations de mémoire ou de backend
-   [x] Confirmer que les benchmarks de compatibilité synthétiques ne représentent pas nécessairement la performance réelle de formation

<a id="real-training-performance"></a>
### Performance de formation réelle

-   [x] Mesurer le débit réel de pré-formation
-   [x] Mesurer le temps d'itération et le nombre d'échantillons/lettres traités par seconde
-   [x] Mesurer l'utilisation de mémoire GPU pendant une formation réelle
-   [x] Comparer les résultats des benchmarks synthétiques avec une formation sur un ensemble de données réel
-   [x] Estimer la durée de formation réaliste

<a id="directml-bottleneck-investigation"></a>
### Analyse du goulot d'étranglement de DirectML

-   [x] Identifier les opérations de fallback sur CPU
-   [x] Mesurer l'impact de performance des fallbacks sur CPU
-   [x] Étudier les surcoûts de synchronisation et d'exécution de DirectML
-   [x] Identifier les opérations responsables d'une formation inattendument lente
-   [x] Déterminer si les goulot d'étranglement peuvent être atténués

<a id="practical-viability"></a>
### Viableté pratique

-   [x] Tester des configurations optimisées de taille de batch / de longueur de séquence
-   [x] Tester des séries de formation plus longues lorsque la performance le permet
-   [x] Évaluer la stabilité lors d'une formation prolongée
-   [x] Déterminer si la formation avec DirectML est pratiquement viable
-   [x] Documenter les limitations de performance connues
-   [x] Valider les chemins de formation Dense et MoE avec DirectML
-   [x] Valider le runner complet de cross-trainer (`9/9`)

> L'étude M4 est complète comme étude historique sur la compatibilité et la performance de DirectML. Ses résultats d'exécution, de débit, de mémoire et de tests de fumée restent valables comme preuve d'ingénierie, mais des vérifications de qualité de formation ultérieures ont montré que les checkpoints localement formés avec DirectML n'étaient pas acceptables. La conclusion de viabilité pratique de M4 est donc supprimée.
>
> Ce baseline M4 est conservé comme preuve historique de validation. M5 a ensuite remplacé les mises à jour de paramètres en FP16 directes par des calculs en FP16 et une optimisation des poids principaux en FP32, après une étude approfondie sur la qualité des checkpoints à long terme.

------------------------------------------------------------------------

<a id="m5-----directml-final-evaluation-"></a>
## M5 --- Évaluation finale de DirectML ✅

Finaliser l'analyse de DirectML avec une validation de qualité de modèle de formation complète et déterminer si DirectML devrait rester un backend de formation pour MiniMind.

<a id="finalization-work"></a>
### Travail de finalisation

-   [x] Appliquer les optimisations de performance identifiées
-   [x] Documenter les workflows de formation supportés avec DirectML
-   [x] Documenter les workflows périodiques de sauvegarde et de reprise
-   [x] Corriger la normalisation de croisement de DirectML avec padding ignoré
-   [x] Valider la pré-formation en FP16 + FP32 sur deux tours complets
-   [x] Rejeter les mises à jour directes en FP16 d'AdamW après des diagnostics à long terme
-   [x] Benchmark une synchronisation réduite du CPU pour diagnostics
-   [x] Évaluer les checkpoints finaux de manière qualitative
-   [x] Comparer les checkpoints formés avec DirectML et ROCm sur un même ensemble d'évaluation
-   [x] Ré-évaluer le checkpoint de DirectML sous ROCm
-   [x] Comparer le même checkpoint de DirectML sous DirectML/ROCm en FP16 et FP32
-   [x] Isoler le blocage restant à la chaîne numérique de DirectML FP16 testée

<a id="final-evidence"></a>
### Preuves finales

Sur les mêmes échantillons `1,000` (`204,327` tokens valides), évalués sous ROCm :

```text
DirectML-trained: loss 6.880233, perplexity 972.8529
ROCm-trained:     loss 1.846587, perplexity   6.3382
```

En utilisant le même checkpoint entraîné avec DirectML et un batch fixe (`873` tokens valides) :

```text
DirectML FP16: 5.733902
ROCm FP16:     6.230469

DirectML FP32: 6.231627
ROCm FP32:     6.231628
```

DirectML FP16 sans SDPA a produit des logits non-finis, tandis que DirectML et ROCm étaient en accord avec une précision approximative de `1e-6` en FP32.

<a id="final-outcome"></a>
### Résultat final

-   [x] La compatibilité d'exécution de DirectML est documentée
-   [x] Les performances et limitations de DirectML sont documentées
-   [x] La qualité du checkpoint complet est rejetée
-   [x] Le blocage restant est classé comme une divergence numérique en FP16 de DirectML
-   [x] L'entraînement de DirectML est fermé pour ce projet
-   [x] Le code, les tests et les preuves de DirectML sont conservés comme une étude de faisabilité terminée

> Cette roadmap s'arrête intentionnellement à M5. Le développement de ROCm est suivi dans une autre roadmap en dehors de l'archive de DirectML.

------------------------------------------------------------------------

<a id="project-goal"></a>
## Objectif du projet

Le projet DirectML a évalué la possibilité d'entraîner MiniMind de manière fiable sur une carte graphique AMD Radeon sous Windows via `torch-directml`.

L'étude est terminée. DirectML a pu exécuter le pipeline d'entraînement et passer des validations de compatibilité, de performances et de fumée, mais les checkpoints finaux n'ont pas atteint une qualité acceptable du modèle. La validation croisée entre backends a isolé l'échec restant à la chaîne numérique FP16 testée de DirectML.

La livraison finale est donc l'étude de faisabilité conservée, et non un backend d'entraînement actif de DirectML.
