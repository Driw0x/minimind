[English](update_log.md) | [Français](update_log.fr.md) | [简体中文](update_log.zh-CN.md)

[README du projet](../../README.fr.md)

<a id="minimind-----update-log"></a>
# MiniMind --- Journal des mises à jour

> **Archive historique de DirectML — mise à jour le 2026-09-24**
>
> DirectML n'est plus le backend actif pour le training de MiniMind. La validation finale M5 a isolé une divergence numérique spécifique au chemin testé de DirectML FP16. Les résultats historiques ci-dessous sont conservés pour la reproductibilité et comme référence technique ; le développement de ROCm est documenté séparément.

Ce document fournit un registre chronologique des principales modifications apportées à la branche DirectML de MiniMind.

Pour les problèmes techniques, leurs causes et les solutions, consultez [`directml_issues.md`](directml_issues.fr.md).

Pour les décisions architecturales durables et les leçons techniques, consultez [`project_memory.md`](project_memory.fr.md).

------------------------------------------------------------------------

<a id="m1-----directml-foundation"></a>
# M1 --- Fondation de DirectML

-   Configuration de PyTorch 2.4.1 avec `torch-directml`.
-   Ajout de la détection et de l'initialisation du dispositif DirectML.
-   Ajout du dispositif DirectML comme dispositif d'exécution explicite de MiniMind.
-   Intégration de DirectML dans le traitement existant des dispositifs de MiniMind.
-   Validation de la localisation des modèles et des tenseurs sur DirectML.
-   Validation du passage en avant et du calcul de perte.
-   Validation du passage en arrière et du calcul des gradients.
-   Validation de l'exécution de l'optimiseur AdamW.
-   Validation d'une étape de training minimale complète.
-   Confirmation que le fallback connu d'AdamW sur CPU DirectML ne bloque pas le training.
-   Ajout de tests automatisés de compatibilité avec DirectML.
-   Validation de l'installation à partir d'un environnement virtuel propre.

------------------------------------------------------------------------

<a id="m2-----minimind-training-on-directml"></a>
# M2 --- Training de MiniMind sur DirectML

-   Audit du code spécifique à CUDA dans le pipeline de training de MiniMind.
-   Ajout de la gestion du dispositif compatible DirectML dans le flux de training.
-   Adaptation des outils spécifiques à CUDA lorsque nécessaire.
-   Ajout de la prise en charge de DirectML pour `eval_llm.py`.
-   Validation d'une configuration légère de training utilisant :

``` text
hidden_size = 128
num_hidden_layers = 2
```

-   Exécution d'une pré-formation légère de MiniMind sur DirectML.
-   Validation du calcul de perte de training.
-   Validation de la génération des poids du modèle.
-   Sauvegarde et rechargement avec succès d'un point de contrôle de training.
-   Validation de l'évaluation à l'aide d'un point de contrôle formé sur DirectML.
-   Validation de la génération de texte sur DirectML.
-   Confirmation du flux complet de DirectML :

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

-   Identification d'une échec lors du chargement d'un point de contrôle causé par un incompatibilité de l'architecture du modèle plutôt que par DirectML.
-   Conservation du modèle `128 / 2-layer` comme configuration légère de régression sur DirectML.

------------------------------------------------------------------------

<a id="m3-----full-training-pipeline"></a>
# M3 --- Pipeline de training complet

-   Extension de la gestion du dispositif dans les outils de training de MiniMind.
-   Suppression des hypothèses restantes selon lesquelles l'exécution du modèle utilise toujours CUDA.
-   Centralisation de la gestion de compatibilité avec les backends et les dispositifs dans les outils communs de training.
-   Extension de la compatibilité avec DirectML dans les étapes principales de training de MiniMind.
-   Validation de la pré-formation.
-   Validation de l'entraînement par apprentissage supervisé (SFT).
-   Validation de l'entraînement LoRA.
-   Validation de DPO.
-   Validation de GRPO.
-   Validation de PPO.
-   Validation de l'apprentissage par renforcement pour agents.
-   Validation de la distillation du savoir.
-   Ajout d'une gestion du modèle de récompense conscient du dispositif pour les workflows d'entraînement d'alignement.
-   Conserve le modèle MiniMind entraînable sur DirectML tout en exécutant le modèle de récompense incompatibles sur CPU.
-   Assure que les entrées du modèle de récompense sont transférées sur le dispositif d'exécution approprié.
-   Ajout d'une gestion partagée des séquences de tokens générés vides.
-   Ajout d'une gestion sécurisée des sélections de tokens vides lors de l'entraînement par distillation.
-   Ajout de tests de compatibilité de points de sauvegarde à travers les étapes d'entraînement prises en charge.
-   Vérification de la compatibilité de sauvegarde/chargement des points de sauvegarde sans introduire des sémantiques spécifiques à DirectML.
-   Documentation des opérations DirectML non prises en charge ou partiellement prises en charge.
-   Ajout de fixtures de tests déterministes pour la validation de l'entraînement léger.
-   Ajout d'outils de développement pour l'audit de compatibilité DirectML.
-   Ajout d'un outil PowerShell pour exécuter séquentiellement les étapes principales d'entraînement.
-   Ajout d'un benchmark de compatibilité DirectML pour `batch_size` et `max_seq_len`.
-   Identification de résultats de benchmark incohérents causés par un choix physique du GPU non contrôlé.
-   Ajout de la sélection explicite d'un adaptateur DirectML pour le GPU dédié souhaité.
-   Réexécution du benchmark de compatibilité sur le GPU dédié sélectionné explicitement.
-   Établissement d'une matrice de compatibilité de référence sur `directml:1`.
-   Vérification de toutes les configurations testées jusqu'à `16 × 340`.
-   Vérification de `32 × 128`, tandis que `32 × 256` a échoué.
-   Confirmation que le succès du benchmark ne doit pas être interprété comme une stabilité garantie dans l'entraînement complet.
-   Confirmation que la pipeline principal d'entraînement est prêt pour des validations prolongées de performance et de stabilité.

------------------------------------------------------------------------

<a id="m4-----performance--stability"></a>
# M4 --- Performance & Stability

-   Lancement de la validation de performance et de stabilité de l'entraînement sur données réelles avec DirectML.
-   Ajout d'entraînements réels limités utilisant `--max_steps` afin de rendre le benchmark DirectML pratique.
-   Confirmation de l'exécution continue de l'entraînement préalable avec le jeu de données de préentraînement MiniMind réel.
-   Benchmarking de l'entraînement sur l'adaptateur `directml:1` sélectionné explicitement.
-   Test de l'entraînement réel avec la configuration complète `max_seq_len = 340`.
-   Vérification de l'exécution sur données réelles avec `batch_size = 8` et `max_seq_len = 340`.
-   Vérification d'une exécution réelle limitée à 100 étapes avec `batch_size = 8` et `max_seq_len = 340`.
-   Confirmation que `32 × 340` échoue en raison d'une mémoire GPU insuffisante lors de l'entraînement réel.
-   Confirmation que `16 × 340` peut compléter un cycle d'accumulation mais échoue plus tard en raison d'une mémoire GPU insuffisante.
-   Établissement de `8 × 340` avec l'accumulation de gradients `8` comme configuration de référence actuelle pour l'entraînement réel.
-   Mesure d'environ `11.62 GB` de mémoire maximale dédiée au GPU lors du benchmark de référence.
-   Utilisation de l'accumulation des gradients pour préserver une taille effective de batch plus grande tout en maintenant la taille de batch physique compatible avec les contraintes de mémoire de DirectML.
-   Ajout d'étapes de préchauffage avant la mesure de performance afin d'éviter d'inclure les surcoûts d'initialisation dans le temps d'état stable.
-   Mesure initiale de formation réelle rapportée environ `5.280 s/iteration`, `1.52 samples/s` et `304.81 effective tokens/s`.
-   Analyse de la mesure de formation réelle inattendument lente.
-   Identification d'une synchronisation inutile entre DirectML et CPU dans le chemin de mesure de formation.
-   Suppression de la synchronisation inutile dans le chemin critique de formation et de mesure.
-   Réexécution du même chargeur de référence `8 × 340` pour 100 étapes après la correction.
-   Mesure environ `0.482 s/iteration`, `16.61 samples/s` et `3341.37 effective tokens/s`.
-   Mesure environ `5647.46 padded tokens/s`.
-   Réduction de la durée d'itération mesurée d'environ `10.95×` par rapport à l'essai initial.
-   Réduction de la durée estimée d'un épisode de formation de environ `9.70 days` à `21.24 hours` (`0.89 days`).
-   Confirmation que l'amélioration de performance a été obtenue sans réduction de l'architecture du modèle, de la taille du batch, de la longueur de séquence ou de la configuration d'accumulation des gradients.
-   Confirmation que le fallback CPU AdamW `aten::lerp.Scalar_out` reste présent mais ne explique pas entièrement le ralentissement initial.
-   Établissement de l'exécution basée sur des étapes limitées comme méthode préférée pour la validation de la performance et de la stabilité de DirectML.
-   Comparaison de la mesure synthétique de compatibilité DirectML avec le comportement réel de formation de MiniMind.
-   Confirmation que les configurations passant la mesure synthétique de compatibilité ne sont pas garanties d'être pratiques ou stables avec des données d’entraînement réelles.
-   Identification de `batch_size` et `max_seq_len` comme des contraintes mémoire interagissant plutôt que comme des limites de compatibilité indépendantes.
-   Confirmation que des lots physiques plus petits offrent un écart significativement plus large pour des longueurs de séquence complètes.
-   Identification du fallback CPU DirectML lors de l'exécution d'AdamW pour des opérations non prises en charge telles que `aten::lerp.Scalar_out`.
-   Confirmation que ce fallback ne bloque pas la formation mais introduit un potentiel goulot d'étranglement.
-   Séparation de la compatibilité du dispositif DirectML de la performance de formation en chaîne complète.
-   Confirmation que le choix explicite d'adaptateur physique reste nécessaire pour obtenir des résultats de benchmark reproductibles sur des systèmes multi-GPU.
-   Documentation de la différence entre les résultats de compatibilité synthétique et les observations de formation avec données réelles.
-   Consolidation de la méthodologie et des résultats de benchmark DirectML pour les futures vérifications de régression.

Conclusions actuelles :

``` text
DirectML compatibility
        ↓
Synthetic benchmark
        ↓
Real-data bounded training
        ↓
Performance / fallback analysis
        ↓
Practical configuration
```

La configuration de référence pratique actuelle pour la validation de données réelles DirectML est :

``` text
Device:              directml:1
Batch size:          8
Max sequence length: 340
Gradient accumulation: 8
```

Cette configuration est la **base de validation et de benchmark** actuelle.

Validation finale M4 en plus :

-   a introduit un chemin stable DirectML en FP16 utilisant un échelle de perte statique `1024` et un epsilon AdamW `1e-4` ;
-   a confirmé que l'epsilon par défaut d'AdamW `1e-8` devient non-finie après la première mise à jour de l'optimiseur dans le chemin pur en FP16 testé ;
-   a validé la matrice complète de compatibilité en FP16 limitée par la première mise à jour de l'optimiseur ;
-   a terminé une phase de pré-formation de `1000` étapes avec des pertes finies tout au long ;
-   a effectué `125` mises à jour de l'optimiseur pendant cette phase ;
-   identifié une valeur mesurée approximative `7,769.62 MB` en VRAM dédiée en FP16 ;
-   réduit la valeur maximale de VRAM dédiée par environ `33.15%` par rapport à la mesure précédente en FP32 ;
-   propage la prise en charge FP16 de DirectML et le support global `--max_steps` dans les étapes Pré-entraînement, SFT complet, LoRA, Distillation, GRPO, RL Agent et PPO ;
-   ajouté `tests/test_all_trainers.py` pour une validation explicite de l'entraîneur DirectML en mode fumée ;
-   valide avec succès tous les entraîneurs inclus dans ce kit de validation ;
-   conclut que la configuration DirectML testée est pratiquement viable.
-   étend le workflow de validation avec une couverture des étapes de pré-entraînement dense et MoE / SFT complet ;
-   identifie un comportement non pris en charge de dispersion dans la chaîne de routage MoE dispersé dans DirectML ;
-   ajoute un chemin de compatibilité MoE sans dispersion spécifique à DirectML tout en conservant le routage dispersé original sur CPU/CUDA ;
-   valide les étapes de pré-entraînement MoE, SFT complet MoE et distillation dense-étudiant/MoE-enseignant ;
-   corrige le traitement parallèle du DataLoader dans Agent RL sur Windows en déplaçant `collate_fn` dans le portée du module ;
-   isole les NaN en FP16 de l'agent à la récomputation complète de la politique et de la référence à l'aide du masque d'attention ;
-   évite le chemin problématique du masque d'attention pour la récomputation valide de l'agent à droite et conserve le calcul en FP32 pour les calculs numériquement sensibles dans le RL ;
-   achève avec succès le kit de validation final de l'entraîneur DirectML :

``` text
All trainer smoke tests passed
Passed: 9/9
```

M4 est terminé. Le projet passe maintenant à **M5 --- Finalisation du projet**.

------------------------------------------------------------------------

<a id="m5-----project-finalization"></a>
# M5 --- Finalisation du projet

-   diagnostique une chute de l'entraînement dense à long terme dans laquelle le chemin d'optimiseur pur en FP16 précédent restait fini mais convergeait vers une copie de token.
-   Ajout de poids maîtres en FP32 pour l'entraînement dense en FP16 sur DirectML et validation du chemin corrigé à l'étape globale `1100` avec une perte diagnostic moyenne `6.7506`, une précision Top-1 `6.30%` et un taux de répétition `0.89%`.

-   Ajout de paramètres spécifiques aux entraîneurs (taille de lot, longueur de séquence, accumulation) au workflow d'entraînement séquentiel, tout en conservant la configuration M4 `8 × 340` documentée comme référence de base pour l'entraînement dense sur DirectML.
-   Ajout de points de sauvegarde périodiques spécifiques aux entraîneurs et documentation de la récupération d'entraînement interrompu avec `--from_resume 1`.
-   Identifie une réduction incorrecte de DirectML pour `cross_entropy(..., ignore_index=-100)` sur des lots causaux paddés.
-   Remplace la réduction implicite par une normalisation explicite de la perte sur les tokens valides.
-   Déplace les outils de validation de perte DirectML ciblés de `scripts/` à `tests/`.
-   Valide une nouvelle exécution corrigée de pré-entraînement FP16 + FP32-maître à l'étape `100` avec une perte `7.3925`, une précision Top-1 `3.26%` et un taux de répétition `3.26%`.
-   Valide la même exécution à l'étape `1000` avec une perte `6.7785`, une précision Top-1 `6.76%` et un taux de répétition `0.76%`.

-   Redémarre l'entraînement dense à partir de zéro avec le chemin corrigé de calcul en FP16 + poids maîtres en FP32 + normalisation de perte sur tokens valides et les paramètres d'entraînement préalables.
-   Termine avec succès le premier épisode complet d'entraînement dense avec une ressource physique `batch_size = 32`.
-   A validé le point de contrôle de l'époque 1 sur les échantillons `0–255` avec une perte globale pondérée par les tokens `6.1017`, une précision Top-1 `15.07%`, un taux de répétition Top-1 `0.83%`, et une entropie moyenne `5.3160`.
-   Confirmé un accord exact entre la perte du modèle et la perte normalisée manuelle par tokens (`6.48799419`, différence `0` sur le batch de référence).
-   Identifié la perte diagnostique agrégée précédemment rapportée autour de `12–13` comme un problème d'agrégation `diagnose_pretrain.py` plutôt qu'une divergence d'entraînement.
-   A commencé à corriger `diagnose_pretrain.py` pour utiliser le chemin réel de perte de batch et l'agrégation pondérée par les tokens.

-   Rétesté la perte du causal-LM DirectML avec des logits identiques utilisant `sum / valid`, par token `none → FP32 mean`, et des réductions de référence sur CPU en FP32.
-   Confirmé que l'approche initiale `reduction="sum" / valid_tokens` était également incorrecte sur le chemin DirectML FP16 testé.
-   Remplacée par `reduction="none"` suivie d'une moyenne en FP32 sur les tokens valides non-remplissage.
-   Validé le chemin final de perte par batch de taille 32 avec la perte du modèle `6.30274916` par rapport à la référence en FP32 sur CPU `6.30530691` (différence environ `0.00256`).
-   Reclassé les points de contrôle intermédiaires de deux époques / de l'entraînement complet antérieur comme des artefacts d'investigation car leur passage en arrière utilisait la réduction obsolète `sum / valid`.
-   L'entraînement dense final doit être redémarré à partir de zéro avec des poids maîtres en FP32 et le chemin final de perte par token en FP32-valide-moyenne.
-   Validé le chemin final DirectML FP16 + poids maîtres FP32 + perte par token en croisement de moyenne par token à travers l'étape `1200`.
-   Les pertes d'entraînement et les pertes manuelles ont resté dans une plage environ `0.003` sur deux régions d'évaluation indépendantes de 256 échantillons.
-   À l'étape `1200`, la perte de l'entraînement a atteint `6.3752` et `6.3568`, la précision Top-1 a atteint `8.66%` et `8.91%`, et le taux de répétition Top-1 a resté limité à `1.96%` et `2.08%`.
-   Confirmé qu'il n'y avait pas de récurrence du collassus d'autocopie historique ; aucune récurrence n'a été observée au cours de l'époque complète suivante.
-   Terminé l'époque 1 de l'entraînement dense final DirectML en utilisant le calcul en FP16, les poids maîtres en FP32 et la perte par token en moyenne valide FP32.
-   Sur les échantillons `0–255`, validé la perte d'entraînement `6.0612`, la perte manuelle `6.0637`, la précision Top-1 `13.71%`, le taux de répétition `0.81%`, et l'entropie moyenne `5.3487`.
-   Sur les échantillons `100000–100255`, validé la perte d'entraînement `6.0177`, la perte manuelle `6.0201`, la précision Top-1 `13.80%`, le taux de répétition `0.71%`, et l'entropie moyenne `5.3436`.
-   Confirmé un accord entre la perte d'entraînement et la perte manuelle dans une plage environ `0.0025` après une époque complète et aucune récurrence du collassus d'autocopie historique.
-   Terminé l'époque 2 de l'entraînement dense final corrigé DirectML.
-   Évalué le point de contrôle final sur `4,096` échantillons provenant de quatre régions de jeu de données ; la perte d'entraînement et la perte manuelle ont resté étroitement alignées et les taux de répétition Top-1 ont resté en dessous de `1%`.
-   Confirmé par une génération qualitative que le collassus historique de répétition de tokens n'a pas réapparu.
-   A commencé par conserver le checkpoint de l'époque 2 comme base de pré-entraînement dense, puis l'a rejeté après validation qualitative finale et de qualité de modèle croisée entre backends.
-   Comparé la précision mixte CUDA de l'origine avec le chemin DirectML et confirmé que le chemin d'origine conserve les paramètres de modèle en FP32 tout en utilisant l'automatisation BF16 pour les opérations de calcul compatibles.
-   Testé un chemin de tensor BF16 DirectML similaire à l'origine ; le backend DirectML actuel a rejeté `BFloat16` avant même la première étape d'entraînement.
-   Rétesté l'entraînement dense sans poids maîtres en FP32, en utilisant des mises à jour directes d'AdamW sur les paramètres de modèle en FP16.
-   À l'étape `1000`, le checkpoint rapide en FP16 direct a produit une perte diagnostique `9.2573`, une précision Top-1 `0.31%` et un taux de répétition `0.29%`.
-   À l'étape `2000`, la perte diagnostique s'est améliorée jusqu'à `8.3102` et la précision Top-1 jusqu'à `0.92%`, tandis que le taux de répétition a augmenté jusqu'à `1.20%`.
-   À l'étape `5000`, la perte diagnostique s'est améliorée davantage jusqu'à `7.3196`, mais le taux de répétition Top-1 a augmenté brusquement jusqu'à `39.24%`, tandis que le taux de répétition sur les données réelles restait environ `0.29%`.
-   Rejeté le chemin d'optimiseur direct en FP16 comme stratégie d'entraînement durable et conservé le calcul en FP16 avec des poids maîtres en FP32 et des mises à jour d'AdamW en FP32.
-   A commencé à optimiser le chemin stable en limitant la synchronisation des scalaires DirectML vers CPU aux frontières `log_interval` au lieu de chaque batch ou mise à jour d'optimiseur.
-   A conservé le clipping des gradients, les mises à jour maîtres en FP32 et la synchronisation maître-modèle inchangée à chaque étape d'optimiseur.
-   A effectué une comparaison entre les entraîneurs stables et synchronisés par journal sur `1000` étapes avec la même configuration de pré-entraînement DirectML.
-   A mesuré `2003.79 s` pour l'entraîneur stable et `1930.75 s` pour l'entraîneur synchronisé par journal.
-   A réduit le temps moyen par étape de `2.0038 s` à `1.9307 s`, correspondant à une amélioration de vitesse de `1.038×` et une réduction de temps de `3.65%`.
-   La validation de la capacité de traitement de l'entraîneur synchronisé par journal est terminée ; la validation de la qualité des checkpoints reste en attente avant qu'il ne devienne l'entraîneur de référence final.

------------------------------------------------------------------------

<a id="m5-----final-directml-quality-assessment"></a>
# M5 --- Évaluation finale de la qualité DirectML

-   Re-évalué les checkpoints de pré-entraînement locaux dense après l'exécution, la perte et les corrections de l'optimiseur.
-   Confirmé que le tokenizer encode et décode correctement et que le jeu de données `pretrain_t2t_mini.jsonl` est lisible et cohérent.
-   Évalué le checkpoint officiel upstream `pretrain_768.pth` avec le même chemin `eval_llm.py`.
-   Confirmé une génération cohérente à partir du checkpoint officiel sur CPU et `directml:1`.
-   Confirmé que le checkpoint de pré-entraînement DirectML local de l'époque 1 est incohérent.
-   Confirmé que le checkpoint de pré-entraînement DirectML local de l'époque 2 est également incohérent.
-   Éliminé le tokenizer, le jeu de données, le chargement du checkpoint et le chemin d'inférence DirectML comme explication principale de la différence de qualité.
-   Conclu que le chemin d'entraînement DirectML personnalisé n'a pas produit une qualité de modèle acceptable, malgré des pertes finies, une correspondance avec les pertes diagnostiques et une exécution complète d'époque.
-   Réclassifié la conclusion précédente sur les M4/M5 « pratiquement viable » de DirectML comme un résultat d'exécution/performances uniquement, et non comme une validation finale de la qualité du modèle.
-   Arrêté tout travail de formation complète de DirectML.
-   Préservé tous les codes, benchmarks, tests, problèmes et commandes de compatibilité avec DirectML comme matériel d'étude de faisabilité historique.

------------------------------------------------------------------------
-   Auréolé un point de pré-entraînement ROCm uniquement comme backend de référence pour la validation finale de DirectML et confirmé une génération cohérente.
-   Évalué les points de pré-entraînement DirectML et ROCm sur les mêmes échantillons `1,000` (tokens `204,327` valides) sous ROCm.
-   Mesuré la perte du point de pré-entraînement DirectML `6.880233` / la perplexité `972.8529`.
-   Mesuré la perte du point de pré-entraînement ROCm `1.846587` / la perplexité `6.3382`.
-   Re-évalué le point de contrôle DirectML sous ROCm et confirmé que la génération restait incohérente, prouvant que le comportement dégradé est stocké dans le point de contrôle plutôt que d'être uniquement un effet d'inférence DirectML.
-   Comparé le même point de contrôle DirectML et le batch de tokens `873` corrigé entre backends : perte FP16 DirectML `5.733902` versus perte FP16 ROCm `6.230469`.
-   Répété la même comparaison en FP32 : DirectML `6.231627` versus ROCm `6.231628`, avec des statistiques de logits correspondantes à environ `1e-6`.
-   Confirmé que DirectML FP16 sans SDPA produit des logits non-finis tandis que ROCm reste stable.
-   La perte du modèle DirectML FP16 correspond au recalcul de l’entropie croisée sur CPU en FP32 à partir des mêmes logits DirectML (`5.733902` contre `5.736305`) ; la réduction d’entropie croisée conservée n’est donc pas la cause restante.
-   Classifié le blocage final comme une divergence numérique dans le chemin forward testé de DirectML FP16.
-   Fermé le travail de formation DirectML et transféré le développement futur des backends à un autre plan de développement ROCm.
