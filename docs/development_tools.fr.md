[English](development_tools.md) | [Français](development_tools.fr.md) | [简体中文](development_tools.zh-CN.md)

[README du projet](../README.fr.md)

<a id="development-tools"></a>
# Outils de développement

<a id="purpose"></a>
## Objectif

MiniMind utilise désormais deux programmes diagnostiques pour la validation du modèle et de l'ensemble de données :

- `scripts/diagnose_validation.py` compare les points de contrôle du modèle et regroupe les flux de diagnostic et de validation de la formation initiale.
- `scripts/diagnose_sft.py` regroupe l'analyse de la qualité de l'ensemble de données SFT et l'analyse de la distribution source au niveau des blocs.

Les deux outils fusionnés ont été vérifiés de bout en bout dans l'environnement ROCm actuel. Les outils précédents `diagnose_training.py`, `diagnose_sft_data.py` et `diagnose_sft_blocks.py` sont obsolètes et peuvent être supprimés.

---

<a id="1-diagnose_validationpy"></a>
## 1. `diagnose_validation.py`

<a id="goal"></a>
### Objectif

`diagnose_validation.py` compare un ou plusieurs poids de MiniMind à l'aide de trois vues complémentaires :

1. méta-données d’entraînement et statistiques de l'ensemble de données ;
2. comparaison de pertes locale sur des échantillons aléatoires réproductibles tirés des ensembles de données de pré-formation et de SFT ;
3. génération déterministe sur un benchmark de prompts externe.

Le premier poids passé par `--weights` est utilisé comme référence de comparaison.

<a id="modes"></a>
### Modes

Si `--mode` est omis, le script demande interactivement :

```text
[0] Local data validation + training audit
[1] External prompt validation
[2] Full validation
```

Les mêmes modes peuvent être sélectionnés directement :

```text
--mode local
--mode external
--mode full
```

<a id="training-audit"></a>
### Audit de formation

Les modes local/plein rapportent :

- tailles de l'ensemble de données et des fichiers de poids ;
- méta-données inférées du point de reprise ;
- époque, étape, taille du monde et taux d'apprentissage de l'optimiseur sauvegardés lorsqu'ils sont disponibles ;
- vocabulaire du tokenizer et les identifiants des tokens spéciaux ;
- statistiques de l'ensemble de données sur des échantillons aléatoires ;
- comptage moyen des entrées et des tokens supervisés.

Le taux d'apprentissage de l'optimiseur sauvegardé est la valeur stockée dans le point de reprise. Il n'est pas nécessairement le taux d'apprentissage initial de formation.

<a id="local-validation"></a>
### Validation locale

Le script choisit les mêmes indices aléatoires pour chaque point de contrôle comparé. Cela rend les comparaisons de pertes réproductibles et évite le biais d'évaluer uniquement les premières enregistrements de l'ensemble de données.

Pour chaque poids, il rapporte :

```text
weight on pretrain | loss | perplexity | supervised tokens
weight on SFT      | loss | perplexity | supervised tokens
```

Il imprime ensuite les écarts de perte par rapport au premier poids.

Une perte de pré-formation plus élevée après SFT est interprétée comme un risque de pertes de connaissances. Une perte de SFT plus faible signifie que le point de contrôle s'adapte mieux à la distribution de SFT.

<a id="external-validation"></a>
### Validation externe

Le benchmark externe utilise un ensemble fixe de prompts en chinois couvrant :

- explication factuelle ;
- code Python ;
- calculs arithmétiques ;
- photosynthèse ;
- suivi d'instructions ;
- connaissance RAM vs SSD ;
- conseils pratiques ;
- correction de code.

La génération est déterministe (`do_sample=False`). Pour chaque réponse, l'outil rapporte le nombre de tokens générés et un ratio de répétition de 3-grammes de caractères.

Ce ratio de répétition est uniquement un indicateur de dégénérescence. La véracité factuelle, la logique, la correction du code et le suivi d'instructions nécessitent encore une vérification manuelle.

<a id="reference-results"></a>
### Résultats de référence

Configuration du modèle :

```text
Parameters: 63.91M
hidden_size: 768
layers: 8
MoE: false
```

Échelle de l'ensemble de données :

```text
Pretrain samples: 1,270,238
SFT samples     :   895,718
```

L'exécution vérifiée de l'outil fusionné (`seed=42`, 256 enregistrements échantillonnés) a mesuré :

```text
Pretrain average input/supervised tokens: 205.57 / 205.57
SFT average input/supervised tokens     : 478.06 / 402.80
```

Ces valeurs sont des diagnostics dépendant de l'échantillon, et non des estimations globales des comptages de tokens dans un corpus.

Validation du point d'arrêt historique (`seed=42`, 256 échantillons par ensemble de données) a produit :

|Poids|Perte de pré-entraînement|Perte de SFT|
| --- | ---: | ---: |
| `pretrain` | 1.8275 | 2.5207 |
| `full_sft` | 1.9992 | 1.5663 |

Par rapport à `pretrain` :

```text
Pretrain loss delta: +0.1717
SFT loss delta     : -0.9543
```

Interprétation : L'entraînement complet SFT améliore considérablement l'ajustement à la distribution SFT, tout en provoquant un oubli modéré sur la distribution de pré-entraînement.

L'essai externe de point d'arrêt a produit :

```text
pretrain average generated tokens: 246.8
pretrain average repetition      : 0.795

full_sft average generated tokens: 212.5
full_sft average repetition      : 0.361
```

L'entraînement complet SFT améliore ainsi la stabilité de la génération et réduit fortement les répétitions sévères, tout en laissant des limites importantes en matière de précision factuelle et de raisonnement sur plusieurs prompts.

<a id="current-dense-full-data-checkpoints"></a>
### Points de contrôle Denses actuels

Les points de contrôle Denses actuellement conservés ont été entraînés sur les ensembles de données complets T2T :

```text
pretrain_768.pth  -> pretrain_t2t.jsonl
full_sft_768.pth -> sft_t2t.jsonl
```

`diagnose_validation.py` utilise par défaut les ensembles mini par référence, comme ensembles de référence fixes, sauf si des chemins spécifiques sont transmis explicitement.

Validation actuelle (`seed=42`, 256 échantillons par ensemble de référence) :

|Poids|Perte de référence de pré-entraînement|Perte de référence de SFT|
| --- | ---: | ---: |
| `pretrain` | 1.6372 | 2.8685 |
| `full_sft` | 1.8730 | 1.3771 |

Par rapport à `pretrain` :

```text
Pretrain loss delta: +0.2358
SFT loss delta     : -1.4913
```

L'essai externe déterministe complet SFT a produit :

```text
average generated tokens: 213.5
average repetition      : 0.244
```

La moyenne de répétition de l'entraînement complet n'est pas conservée comme métrique fiable car plusieurs générations ont atteint la limite de tokens mais ont été décodées en réponse visible vide.


<a id="current-moe-full-data-checkpoints"></a>
### Points de contrôle MoE sur les données complètes

Les entraînements MoE Pretrain et Full SFT et leur évaluation initiale sur AMD ROCm sont terminés : `out/pretrain_768_moe.pth` a été entraîné sur `pretrain_t2t.jsonl`, puis `out/full_sft_768_moe.pth` sur `sft_t2t.jsonl`, initialisé à partir de MoE Pretrain. Configuration : dimension cachée de 768, 8 couches, 4 experts par couche, routage top-1, 198,42 M paramètres totaux et 63,94 M paramètres actifs annoncés.

Avec `seed=42` et 256 échantillons par ensemble de référence Mini :

|Poids|Perte de référence Pretrain|Perte de référence SFT|Perplexité Pretrain / SFT|Répétition externe|
| --- | ---: | ---: | --- | --- |
| `pretrain_768_moe.pth` | 1.5468 | 2.1544 | 4.70 / 8.62 |0.842 (réserve méthodologique)|
| `full_sft_768_moe.pth` | 1.7544 | 1.2994 | 5.78 / 3.67 | 0.262 |

Les références Mini sont des échantillons issus des données d'entraînement. Ces pertes mesurent l'ajustement aux distributions et l'oubli, pas la généralisation sur un test indépendant. Les huit prompts externes utilisent un décodage déterministe et un chat template, y compris pour Pretrain. À l'inverse, `eval_llm.py` utilise BOS + prompt brut pour Pretrain et échantillonne ses sorties. Le score `0.842` comporte donc cette réserve méthodologique.

MoE Full SFT présente des pertes de référence inférieures au Dense Full SFT, mais une répétition légèrement supérieure (`0.262` contre `0.244`). Des erreurs arithmétiques (`12 - 5 + 3` évalué à `16`), du code Fibonacci incorrect, un suivi imparfait des consignes de format, des répétitions et des troncatures subsistent. Ces résultats n'établissent pas de supériorité générale du MoE ; voir [`generation_quality.md`](generation_quality.fr.md).

<a id="baseline-status"></a>
### Statut de référence

Le point de contrôle actuel `full_sft_768.pth` est la base Dense complète SFT conservée, entraînée sur l'ensemble complet de données SFT. Les diagnostics montrent que :

- le point de contrôle se charge et évalue correctement sur ROCm;
- le SFT améliore l'ajustement à la distribution SFT;
- un oubli modéré sur la distribution de pré-entraînement reste mesurable;
- la génération externe est considérablement moins répétitive que le point de contrôle de pré-entraînement;
- les limitations restantes en matière de précision factuelle, de raisonnement et de suivi des instructions sont des limitations du modèle ou des données, et non des échecs du logiciel de diagnostic.

<a id="recommended-commands"></a>
### Commandes recommandées

Comparaison interactive complète :

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0
```

Audit local non interactif :

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode local --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --eval_samples 256 --stat_samples 256 --batch_size 4
```

Audit externe non interactif :

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode external --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --max_new_tokens 256
```

Audit complet + essai externe :

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode full --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --eval_samples 256 --stat_samples 256 --batch_size 4 --max_new_tokens 256
```

Audit MoE + benchmark externe, depuis la racine du dépôt :

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode full --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 1 --eval_samples 256 --stat_samples 256 --batch_size 4 --max_new_tokens 256 --seed 42
```

`--use_moe 1` sélectionne les points de contrôle `_moe.pth` sous `out/` ; les chemins de référence par défaut restent `dataset/pretrain_t2t_mini.jsonl` et `dataset/sft_t2t_mini.jsonl`. Les comparaisons Dense et MoE nécessitent des exécutions séparées, car `--use_moe` s'applique à tous les poids d'une même exécution.

---

<a id="2-diagnose_sftpy"></a>
## 2. `diagnose_sft.py`

<a id="goal-1"></a>
### Objectif

`diagnose_sft.py` audit la JSONL elle-même du SFT. Il combine une analyse détaillée de la qualité des échantillons avec une analyse de distribution au niveau des blocs dans une seule balayage de données.

Il vérifie :

- la validité JSON;
- conversations manquantes ou vides;
- répartition des rôles;
- partage des appels à outils;
- distributions des échantillons initiaux versus échantillons aléatoires;
- longueurs des échantillons utilisateur, assistant et totaux;
- troncature par rapport à `max_seq_len`;
- anomalies de séquence de rôles;
- duplication exacte de prompt / réponse;
- répétition des n-grammes de caractères;
- exemples suspects;
- décalages de distribution source entre blocs de fichiers.

<a id="modes-1"></a>
### Modes

```text
--mode dataset   # detailed FIRST vs RANDOM analysis
--mode blocks    # block-level analysis only
--mode all       # both analyses (default)
```

<a id="current-dataset-findings"></a>
### Résultats sur l'ensemble de données

L'ensemble de données SFT contient :

```text
Total lines             : 895,718
Valid lines             : 895,718
Invalid JSON            : 0
Missing conversations   : 0
Empty conversations     : 0
Conversations with tools: 74,832
Tool share              : 8.35%
```

Le début du fichier diffère fortement d'un échantillon aléatoire.

Premiers 5 000 échantillons :

```text
messages / conversation mean: 7.40
conversation tokens mean     : 663.02
truncated at 768              : 25.34%
assistant repetition mean    : 0.13
```

Échantillons aléatoires 5 000 :

```text
messages / conversation mean: 2.83
conversation tokens mean     : 491.90
truncated at 768              : 5.60%
assistant repetition mean    : 0.17
```

Cela montre que le JSONL est hétérogène et est physiquement ordonné par source/type.

<a id="block-level-findings"></a>
### Résultats au niveau de blocs

En utilisant des blocs de 100 000 lignes et 2 000 échantillons tokenisés par bloc :

|Bloc|Lignes|Séquence unique|Outils|Moyenne des tokens|Troncature|Répétition|
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 |1 à 100 000| 28.88% | 0.00% | 612.9 | 21.65% | 0.153 |
| 2 |100 001 à 200 000| 100.00% | 0.00% | 421.5 | 1.05% | 0.256 |
| 3 |200 001 à 300 000| 100.00% | 0.00% | 492.2 | 1.70% | 0.204 |
| 4 |300 001 à 400 000| 100.00% | 0.00% | 490.1 | 1.65% | 0.209 |
| 5 |400 001 à 500 000| 100.00% | 0.00% | 488.5 | 1.90% | 0.210 |
| 6 |500 001 à 600 000| 48.30% | 0.00% | 558.6 | 10.30% | 0.203 |
| 7 |600 001 à 700 000| 100.00% | 0.00% | 527.2 | 5.05% | 0.099 |
| 8 |700 001 à 800 000| 73.58% | 28.93% | 486.7 | 2.25% | 0.082 |
| 9 |800 001 à 895 718| 56.04% | 47.96% | 361.5 | 3.30% | 0.118 |

Le fichier contient donc des limites claires de source : des données multi-étapes longues, plusieurs régions à une seule étape, et des zones riches en outils vers la fin.

Le taux d'arrêt estimé est d'environ 5 à 6 %, mais les blocs 1 et 6 sont fortement affectés.

Les prompts et réponses dupliqués sont particulièrement fréquents dans les blocs riches en outils. Ces duplications ne doivent pas être supprimées de manière brute car les modèles répétés d'outils peuvent être légitimes.

<a id="full-sft-training-dataset"></a>
### Ensemble complet de données d'entraînement SFT

L'ensemble complet `sft_t2t.jsonl` contient 5 099 432 échantillons. L'analyse par bloc montre une distribution plus importante de contextes longs par rapport à l'ensemble Mini SFT.

Estimations pondérées issues de l'analyse des blocs de 100 000 lignes :

```text
Single-turn share        : ~81.6%
Tool-conversation share  : ~7.7%
Mean conversation length : ~826.8 tokens
Truncated at 768         : ~34.3%
```

Plusieurs blocs ultérieurs dépassent 50 % de troncature au niveau `max_seq_len=768`, certains s'approchant même de 100 %. Ces valeurs sont des estimations basées sur les blocs et les échantillons, et non des statistiques exactes de tokenisation au niveau de l'ensemble complet.


<a id="training-order-note"></a>
### Note sur l'ordre d'entraînement

Le traineur ne consomme pas le JSONL de manière séquentielle lors de l'entraînement normal. Il utilise des indices mélangés (`torch.randperm`) en entraînement non distribué et `DistributedSampler` avec `set_epoch` en entraînement distribué.

Il en résulte que l'ordre des blocs est utile pour diagnostiquer l'hétérogénéité de la source, mais n'est pas preuve que le modèle a été entraîné d'abord sur le bloc 1 puis sur les données d'outils en dernier.

<a id="recommended-commands-1"></a>
### Commandes recommandées

Audit complet de l'ensemble SFT :

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode all --samples 5000 --first_samples 5000 --block_size 100000 --sample_per_block 2000 --max_seq_len 768
```

Audit complet de l'ensemble d'entraînement :

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t.jsonl --tokenizer model --mode all --samples 5000 --first_samples 5000 --block_size 100000 --sample_per_block 2000 --max_seq_len 768
```


Analyse détaillée d'échantillons uniquement :

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode dataset --samples 5000 --first_samples 5000 --max_seq_len 768
```

Analyse par bloc uniquement :

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode blocks --block_size 100000 --sample_per_block 2000 --max_seq_len 768
```

---

<a id="verification-workflow"></a>
## Workflow de vérification

Les outils fusionnés ont déjà effectué avec succès le workflow de validation complet sur l'environnement actuel. Pour les modifications futures, relancer les vérifications suivantes.

Vérifiez d'abord la syntaxe :

```powershell
python -m py_compile scripts\diagnose_validation.py scripts\diagnose_sft.py
```

Vérifiez les deux interfaces de ligne de commande :

```powershell
python -m scripts.diagnose_validation --help
python -m scripts.diagnose_sft --help
```

Ensuite, exécutez un test rapide avant les diagnostics complets :

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode local --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --eval_samples 16 --stat_samples 16 --batch_size 1
```

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode all --samples 100 --first_samples 100 --block_size 100000 --sample_per_block 50 --max_seq_len 768
```

Si ces vérifications se terminent correctement, exécutez les commandes complètes documentées ci-dessus et comparez les résultats avec les valeurs de référence dans ce document.
