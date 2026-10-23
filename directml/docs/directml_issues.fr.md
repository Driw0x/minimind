[English](directml_issues.md) | [Français](directml_issues.fr.md) | [简体中文](directml_issues.zh-CN.md)

[README du projet](../../README.fr.md)

<a id="minimind-----directml-technical-issues"></a>
# MiniMind --- Problèmes techniques DirectML

> **Archive historique DirectML — mis à jour le 2026-09-24**
>
> DirectML n'est plus le backend actif pour le training de MiniMind. La validation finale M5 a isolé une divergence numérique spécifique à la chaîne FP16 testée de DirectML. Les résultats historiques ci-dessous sont conservés pour la reproductibilité et comme référence technique ; le développement ROCm est documenté séparément.

Ce document enregistre les problèmes de compatibilité techniques rencontrés lors de l'adaptation de MiniMind à DirectML.

Les observations relatives aux benchmarks et aux performances sont documentées séparément dans [`directml_benchmarks.md`](directml_benchmarks.fr.md). Les limitations actuelles sont résumées ci-dessous ; les causes détaillées et les décisions restent dans l'historique des issues.


<a id="current-directml-limitations"></a>
## Limitations actuelles de DirectML

|Fonctionnalité / opération|État|Gestion conservée|
| --- | --- | --- |
|AdamW `aten::lerp.Scalar_out`|Partielle|Fallback automatique sur CPU|
|Masque causal du modèle de récompense|Non pris en charge|Modèle de récompense sur CPU|
| `torch.compile` |Non pris en charge|Maintenir désactivé|
|Équivalent CUDA AMP/autocast|Non disponible|Chemin FP16 spécifique à DirectML|
|Routing sparse MoE|Partielle|Chemin de compatibilité sans scatter|
|Masque d'attention de séquence complète de l'agent|Partielle|Éviter le chemin problématique du masque|
|Réduction de croisement `ignore_index`|Partielle|Perte par token + moyenne des tokens valides en FP32|
|Consistance numérique FP16 de DirectML|**Bloquant**|Ne pas utiliser DirectML pour la formation complète|

La limitation bloquante est la divergence numérique FP16 isolée lors de la validation M5. DirectML reste utile uniquement comme cible de compatibilité historique et de vérification de retour.

------------------------------------------------------------------------

<a id="directml-device-displayed-as-privateuseone0"></a>
# Affichage du dispositif DirectML comme `privateuseone:0`

<a id="problem"></a>
## Problème

Après initialisation de DirectML, PyTorch affiche :

``` text
privateuseone:0
```

au lieu d'un nom de dispositif comme `directml:0`.

<a id="cause"></a>
## Cause

`torch-directml` intègre DirectML dans PyTorch via le backend `PrivateUse1`.

Par conséquent, `privateuseone:0` est la représentation interne attendue du dispositif DirectML dans PyTorch.

<a id="solution"></a>
## Solution

Aucune modification de code n'était requise.

La localisation du dispositif a été vérifiée en examinant les paramètres du modèle, les entrées et les sorties.

<a id="decision"></a>
## Décision

`privateuseone:0` est considéré comme le comportement attendu de DirectML.

L'option de dispositif visible pour l'utilisateur reste :

``` text
directml
```

------------------------------------------------------------------------

<a id="directml-not-available-as-an-original-device-option"></a>
# DirectML Non disponible comme option initiale

<a id="problem-1"></a>
## Problème

La version initiale de MiniMind n'a pas fourni DirectML comme dispositif d'exécution.

<a id="cause-1"></a>
## Cause

MiniMind n'était pas initialement conçu autour de `torch-directml`, qui nécessite une initialisation explicite via :

``` python
import torch_directml

device = torch_directml.device()
```

<a id="solution-1"></a>
## Solution

`directml` a été ajouté comme option de dispositif explicite.

Le dispositif résultant continue d'utiliser des opérations standard de PyTorch :

``` python
model.to(device)
tensor.to(device)
```

<a id="decision-1"></a>
## Décision

DirectML devrait réutiliser le pipeline d'exécution existant de MiniMind au lieu d'introduire une implémentation de formation séparée.

------------------------------------------------------------------------

<a id="adamw-cpu-fallback"></a>
# Fonctionnalité de fallback en CPU pour AdamW

<a id="problem-2"></a>
## Problème

Lors d'une étape d'optimiseur AdamW, DirectML signale :

``` text
The operator 'aten::lerp.Scalar_out' is not currently supported
on the DML backend and will fall back to run on the CPU.
```

<a id="cause-2"></a>
## Cause

DirectML ne réalise pas toutes les opérations utilisées internement par `torch.optim.AdamW`.

`torch-directml` exécute automatiquement l'opération non prise en charge sur CPU.

<a id="solution-2"></a>
## Solution

L'étape complète de formation a été validée malgré l'avertissement :

``` text
DirectML backward pass: OK
DirectML optimizer step: OK
DirectML zero_grad: OK
DirectML training step: OK
```

<a id="decision-2"></a>
## Décision

Le fallback est accepté car il ne empêche pas une formation correcte.

Il reste documenté car il peut affecter la performance.

------------------------------------------------------------------------

<a id="torchcompile-unsupported-on-directml"></a>
# `torch.compile` Non pris en charge sur DirectML

<a id="problem-3"></a>
## Problème

Le chemin de formation archivé de DirectML ne prend pas en charge `torch.compile`.

<a id="solution-3"></a>
## Solution

Exécuter les workflows de DirectML avec :

```text
--use_compile 0
```

<a id="decision-3"></a>
## Décision

La compilation reste désactivée dans l'archive de DirectML. Aucun autre remède n'est poursuivi car DirectML n'est plus le backend actif de formation.

------------------------------------------------------------------------

<a id="checkpoint-and-model-architecture-mismatch"></a>
# Incompatibilité entre point d'arrêt et architecture du modèle

<a id="problem-4"></a>
## Problème

`eval_llm.py` a échoué lors du chargement d'un point d'arrêt de validation avec un matching strict du dictionnaire d'état.

<a id="cause-3"></a>
## Cause

L'architecture du modèle d'évaluation ne correspondait pas à l'architecture utilisée lors de la formation.

La compatibilité du point d'arrêt dépend de paramètres tels que :

``` text
hidden_size
num_hidden_layers
vocab_size
```

Le point d'arrêt de validation utilisé :

``` text
hidden_size = 128
num_hidden_layers = 2
```

<a id="solution-4"></a>
## Solution

Le modèle d'évaluation doit être instancié avec un modèle compatible.

Pour la configuration de validation :

``` powershell
--hidden_size 128 `
--num_hidden_layers 2
```

<a id="decision-4"></a>
## Décision

Les échecs lors du chargement du point d'arrêt ne devraient pas être attribués automatiquement à DirectML.

La compatibilité de l'architecture doit être vérifiée en premier.

------------------------------------------------------------------------

<a id="reward-model-fails-on-directml"></a>
# Le modèle de récompense échoue sur DirectML

<a id="problem-5"></a>
## Problème

Lors de la formation GRPO, le modèle principal de MiniMind fonctionne sur DirectML tandis que le modèle de récompense échoue lorsqu'il est forcé sur le même appareil.

<a id="cause-4"></a>
## Cause

Le chemin d'inférence du modèle de récompense utilise des opérations non fiabilisées par DirectML.

Le workflow initial supposait également que les modèles participants pouvaient partager le même appareil d'exécution.

<a id="solution-5"></a>
## Solution

Le modèle formable reste sur DirectML tandis que le modèle de récompense s'exécute sur CPU :

``` text
Trainable model → DirectML
Reward model    → CPU
```

Les entrées du modèle de récompense sont déplacées vers CPU avant l'inférence.

Les valeurs de récompense sont transférées vers le dispositif de formation lorsqu'elles sont nécessaires.

<a id="decision-5"></a>
## Décision

Le modèle de récompense reste intentionnellement sur CPU sous DirectML.

La correction et la stabilité prennent la priorité par rapport à la force de chaque composant sur le GPU.

------------------------------------------------------------------------

<a id="duplicated-device-handling-across-trainers"></a>
# Gestion dupliquée du dispositif entre les formateurs

<a id="problem-6"></a>
## Problème

Au fur et à mesure que le support de DirectML s'est étendu, des gestionnaires spécifiques au dispositif ont commencé à apparaître dans les formateurs individuels.

Cela risquait de créer une logique de backend dupliquée et incohérente.

<a id="cause-5"></a>
## Cause

DirectML introduit des exigences spécifiques au dispositif pour chaque composant, ce qui ne correspond pas à l'hypothèse initiale d'un seul appareil d'exécution partagé.

<a id="solution-6"></a>
## Solution

Le traitement de la compatibilité entre dispositif et backend a été centralisé dans des outils partagés de formation.

Ces outils permettent une localisation commune de :

-   la résolution du dispositif ;
-   l'initialisation de DirectML ;
-   la localisation du modèle ;
-   les exigences spécifiques aux composants en matière de dispositif ;
-   les règles futures de compatibilité avec les backends.

<a id="decision-6"></a>
## Décision

Les entraîneurs individuels doivent rester concentrés sur leur algorithme de formation.

La compatibilité avec les backends doit être intégrée dans l'infrastructure partagée des entraîneurs.

------------------------------------------------------------------------

<a id="empty-token-handling-during-generation"></a>
# Gestion des tokens vides lors de la génération

<a id="problem-7"></a>
## Problème

La formation basée sur la génération pourrait produire une séquence de tokens vide.

Le passage de ce résultat vide en aval pourrait créer un échantillon de formation invalide.

<a id="cause-6"></a>
## Cause

Le chemin de génération supposait que la sortie générée contenait toujours du contenu utile en termes de tokens.

Il n'y avait pas de protection partagée contre une séquence vide.

<a id="solution-7"></a>
## Solution

Une gestion des tokens vides a été ajoutée aux outils partagés de formation.

Les séquences générées sont validées avant le traitement en aval, et une fallback sécurisée est utilisée lorsque nécessaire.

<a id="decision-7"></a>
## Décision

Les outils partagés de génération ne doivent pas retourner de séquences de tokens vides non utilisables.

Cette protection doit être intégrée dans les outils partagés plutôt que dans les implémentations individuelles de GRPO ou PPO.

------------------------------------------------------------------------

<a id="empty-distillation-masks"></a>
# Masks vides de distillation

<a id="problem-8"></a>
## Problème

Certains échantillons SFT tronqués peuvent ne pas contenir de tokens supervisés au sein de la longueur configurée de la séquence.

Cela a causé le calcul de la divergence de KL de distillation de recevoir des logits vides et de échouer.

<a id="cause-7"></a>
## Cause

La perte de distillation supposait que chaque échantillon de formation contenait au moins un token supervisé après troncature et masquage.

Cette hypothèse ne tient pas toujours pour les échantillons courts ou tronqués de validation.

<a id="solution-8"></a>
## Solution

La perte de distillation détecte maintenant les sélections de tokens vides et retourne une perte nulle reliée au graphe de calcul.

<a id="decision-8"></a>
## Décision

Les pertes de formation opérant sur des sélections de tokens masquées doivent gérer de manière sécurisée les sélections valides vides.

Ceci est traité comme un problème de robustesse de formation plutôt qu'une limitation de DirectML.

------------------------------------------------------------------------

<a id="checkpoint-compatibility-across-training-stages"></a>
# Compatibilité des points de sauvegarde entre les étapes de formation

<a id="problem-9"></a>
## Problème

Les modifications de DirectML pourraient potentiellement affecter la compatibilité des points de sauvegarde entre les étapes de formation de MiniMind.

Les étapes pertinentes incluent :

``` text
Pretraining
SFT
DPO
GRPO
PPO
```

<a id="cause-8"></a>
## Cause

Les étapes de formation dépendent de structures d'architecture compatibles et de dictionnaires d'état.

Le traitement du dispositif doit rester indépendant de la représentation du point de sauvegarde.

<a id="solution-9"></a>
## Solution

La compatibilité des points de sauvegarde est testée de manière indépendante de l'exécution de DirectML.

<a id="decision-9"></a>
## Décision

DirectML ne doit pas introduire un format de sauvegarde spécifique à un backend.

Les sémantiques existantes des points de sauvegarde de MiniMind sont préservées.

------------------------------------------------------------------------

<a id="training-measurement-introduced-directml-synchronization-overhead"></a>
# Mesure de formation introduisant un surcoût de synchronisation avec DirectML

<a id="problem-10"></a>
## Problème

Le benchmark initial basé sur des données réelles a rapporté approximativement :

``` text
5.280 s / iteration
```

ce qui impliquait une durée d'entraînement préalable estimée d'environ `9.70 days / epoch`.

Cela a fait apparaître un entraînement DirectML considérablement plus lent que prévu.

<a id="cause-9"></a>
## Cause

Le chemin de mesure de l'entraînement a introduit une synchronisation inutile entre l'exécution asynchrone de DirectML et le CPU.

La fréquente matérialisation des valeurs produites par le dispositif sur le CPU a fait en sorte que le benchmark mesurait non seulement le travail réel d'entraînement, mais aussi les surcoûts de synchronisation et d'installation.

<a id="solution-10"></a>
## Solution

Le chemin critique d'entraînement et de mesure a été ajusté afin d'éviter les synchronisations inutiles entre DirectML et CPU à chaque étape.

Le même charge de travail de référence a ensuite été réexécuté sur 100 étapes :

``` text
batch_size = 8
max_seq_len = 340
gradient_accumulation_steps = 8
```

Le benchmark corrigé a mesuré :

``` text
Average iteration time:    0.482 s
Samples / second:          16.61
Effective tokens / second: 3341.37
Estimated epoch duration:  21.24 h
```

<a id="decision-10"></a>
## Décision

Les mesures de performance ne doivent pas introduire de synchronisation inutile dans le chemin critique d'entraînement.

Avant d'attribuer une performance lente à DirectML au niveau du backend, il faut investiguer les surcoûts liés à la mesure, au journalisation et à la synchronisation.

------------------------------------------------------------------------

<a id="multiple-gpus-and-directml-device-selection"></a>
# Plusieurs GPU et sélection du dispositif DirectML

<a id="problem-11"></a>
## Problème

L'ordinateur de développement expose deux adaptateurs graphiques :

``` text
Integrated GPU (iGPU)
Dedicated GPU (dGPU)
```

Cela a rendu incertain le GPU physique utilisé par DirectML pendant l'entraînement et le benchmark.

<a id="cause-10"></a>
## Cause

L'adaptateur graphique intégré et l'adaptateur graphique dédié sont deux adaptateurs graphiques distincts compatibles avec DirectML.

Par conséquent :

``` text
CPU
 ≠
Integrated GPU
 ≠
Dedicated GPU
```

La sélection implicite du dispositif ne fournit pas suffisamment de certitude sur le GPU physique exécutant la charge de travail.

<a id="solution-11"></a>
## Solution

La sélection du dispositif DirectML doit cibler explicitement le GPU souhaité.

Pour l'entraînement de MiniMind, le carte graphique dédiée doit être sélectionnée.

Conceptuellement :

``` text
Available DirectML adapters
        ↓
Identify dedicated GPU
        ↓
Explicitly select adapter
        ↓
Create DirectML device
        ↓
MiniMind training
```

L'utilisation du GPU peut alors être vérifiée contre l'adaptateur correspondant dans Gestionnaire des tâches de Windows.

<a id="decision-11"></a>
## Décision

La carte graphique dédiée est le dispositif prévu pour l'entraînement de MiniMind.

Les benchmarks de performance ne doivent être comparés que lorsqu'ils ont été obtenus avec le même GPU physique explicitement sélectionné.

<a id="directml-fp16-adamw-instability"></a>
# Instabilité du DirectML FP16 AdamW

<a id="problem-12"></a>
## Problème

L'entraînement préalable en FP16 DirectML devint non-finie immédiatement après la première mise à jour de l'optimiseur lorsque AdamW utilisait son epsilon par défaut :

``` text
eps = 1e-8
```

Un échelonnement statique avec un facteur de mise à l'échelle de `1024` n'a pas résolu le problème seul.

<a id="cause-11"></a>
## Cause

Les gradients étaient finis avant l'étape de mise à jour de l'optimiseur, ce qui a isolé l'échec à la chaîne d'optimisation plutôt qu'à la phase avant ou après.

L'epsilon par défaut d'AdamW est trop petit pour le chemin d'entraînement pur en FP16 validé par DirectML.

<a id="solution-12"></a>
## Solution

Le chemin en FP16 DirectML utilise :

``` text
Static loss scale = 1024
AdamW epsilon     = 1e-4
```

Ce comportement est centralisé dans des outils communs de gestion de l'entraînement et réutilisé par les entraîneurs en FP16 DirectML.

<a id="decision-12"></a>
## Décision

L'entraînement en FP16 DirectML ne doit pas utiliser l'epsilon par défaut d'AdamW sans validation.

L'épsilon validé pour le chemin d'optimisation direct-FP16 historique est `1e-4`. Le chemin d'optimisation Dense FP32-master conservé utilise AdamW `eps = 1e-8`.

------------------------------------------------------------------------

<a id="inconsistent-bounded-run-support-across-trainers"></a>
# Support incohérent de la mise en œuvre limitée des entraîneurs

<a id="problem-13"></a>
## Problème

L'implémentation initiale de M4 pour la mise en œuvre limitée a ajouté `--max_steps` à la pré-entraînement en premier, tandis que d'autres entraîneurs attendaient encore une exécution complète basée sur des époques.

Cela a empêché un run de fumée consolidé de contrôler de manière cohérente chaque charge de travail d'entraînement.

<a id="solution-13"></a>
## Solution

La gestion globale limitée `--max_steps` a été propagée à tous les entraîneurs utilisés par le kit de fumée DirectML FP16.

Un lanceur Python a été ajouté à :

``` text
tests/test_all_trainers.py
```

Il démarre les entraîneurs séquentiellement et s'arrête à la première panne.

<a id="decision-13"></a>
## Décision

La mise en œuvre limitée fait partie de la stratégie commune de validation des workflows entraînables DirectML.

Les tests de fumée lourds sont exécutés explicitement plutôt que d'être traités comme des tests légers ordinaires Pytest.


------------------------------------------------------------------------

<a id="moe-routing-uses-unsupported-scatter-behavior-on-directml"></a>
# L'usage de MoE Routing utilise un comportement de dispersion non pris en charge sur DirectML

<a id="problem-14"></a>
## Problème

Le chemin d'agrégation MoE en haut de la chaîne a échoué lors de la validation DirectML.

L'échec observé en premier a eu lieu autour de la logique d'expert secondaire auxiliaire utilisant `F.one_hot`. Remplacer cette opération seul a permis à la passe avant de progresser, mais l'exécution inverse a toujours rencontré un comportement de dispersion non pris en charge dans le chemin de routage dispersé.

<a id="cause-12"></a>
## Cause

L'implémentation MoE dispersée en haut de la chaîne repose sur des opérations de routage indexées et de dispersion similaires qui ne sont pas entièrement prises en charge par le chemin avant/après testé DirectML.

<a id="solution-14"></a>
## Solution

Un chemin de routage MoE sans dispersion spécifique à DirectML a été ajouté.

Conceptuellement :

``` text
Gate scores
    ↓
Top-k expert selection
    ↓
Broadcast routing mask
    ↓
Differentiable routing weights
    ↓
Evaluate experts
    ↓
Weighted expert combination
```

CPU et CUDA conservent l'implémentation originale de routage dispersé.

<a id="decision-14"></a>
## Décision

DirectML utilise le chemin de compatibilité MoE sans dispersion.

Le mode de rejet privilégie l'exécution correcte au lieu de l'efficacité de MoE dispersé et devrait donc ne pas être utilisé pour inférer les performances natives de MoE dispersé.

------------------------------------------------------------------------

<a id="agent-rl-compatibility-on-windows-and-directml-fp16"></a>
# Compatibilité Agent RL sur Windows et DirectML FP16

<a id="problem-15"></a>
## Problème

L'Agent RL a rencontré deux problèmes indépendants lors de la validation finale de M4.

Premièrement, les workers DataLoader de Windows n'ont pas pu importer un `collate_fn` défini localement.

Deuxièmement, après avoir corrigé le multiprocessing, la première étape d'entraînement de l'Agent a produit une perte non-finie.

<a id="cause-13"></a>
## Cause

Le multiprocessing sur Windows utilise le lancement de processus, donc le DataLoader `collate_fn` doit être importable dans le portée du module.

Pour l'échec numérique, des vérifications ciblées sur des valeurs finies ont montré :

``` text
rewards     = finite
advantages  = finite
behavior    = finite
policy      = non-finite
reference   = non-finite
```

Des vérifications supplémentaires ont confirmé que les logits de la politique contenaient des NaN lors du chemin complet de récomputation de la politique et de la référence utilisant `attention_mask=full_mask`.

La génération de rollout reste finie.

<a id="solution-15"></a>
## Solution

L'Agent `collate_fn` a été déplacé dans la portée du module pour garantir la compatibilité avec le multiprocessing Windows.

Pour les lots d'Agent droite-paddés, la récomputation complète de la politique et de la référence dans le chemin de séquence évite le chemin problématique de masque d'attention DirectML FP16.

Les calculs de probabilité logarithmique numériquement sensibles, de KL et de ratios sont effectués en FP32.

<a id="decision-15"></a>
## Décision

Le traitement de la compatibilité spécifique à l'Agent est conservé car il corrige la chaîne FP16 DirectML sans nécessiter une modification plus large des sémantiques d'attention de MiniMind.

Après ces modifications, l'entraînement de l'Agent RL a réussi et le jeu complet de tests de fumée a été terminé avec :

``` text
Passed: 9/9
```

------------------------------------------------------------------------

<a id="long-run-directml-fp16-pretraining-collapse"></a>
# Collassus de pré-entraînement DirectML FP16 à long terme

<a id="problem-16"></a>
## Problème

Un entraînement complet Dense a resté fini, mais a ensuite produit un modèle dégénéré se copiant lui-même.

Les diagnostics par imposition d'un modèle non entraîné ont montré environ :

``` text
Mean loss:           13.57
Top-1 accuracy:      0.18%
Top-1 repeat rate:  93.48%
```

Un modèle non entraîné ne présentait pas ce comportement.

<a id="cause-14"></a>
## Cause

La chaîne FP16 DirectML a mis à jour directement les paramètres du modèle FP16 avec AdamW. L'échelle statique des pertes a protégé les gradients, mais n'a pas fourni les poids maîtres en FP32 utilisés par l'optimisation mixte-precision conventionnelle.

<a id="solution-16"></a>
## Solution

L'entraînement Dense DirectML utilise maintenant :

``` text
FP16 model compute
        ↓
FP32 gradient unscale
        ↓
AdamW on FP32 master weights
        ↓
FP32 master weights copied back to FP16 model
```

Le chemin optimiseur en FP32 utilise l'épsilon AdamW `1e-8`.

Une validation sur 100 étapes a amélioré la perte diagnostic moyenne d'un modèle non entraîné de `8.89` à `7.48`. Après reprise à l'étape globale `1100`, la perte moyenne a atteint `6.75`, la précision Top-1 a atteint `6.30%`, et le taux de répétition Top-1 a resté faible à `0.89%`.

<a id="decision-16"></a>
## Décision

Une perte finie seule n'est pas suffisante pour valider l'entraînement DirectML FP16.

L'entraînement Dense utilise des poids maîtres en FP32 pour les mises à jour de l'optimiseur, et les diagnostics par imposition d'un modèle enseignant sont conservés pour détecter un collassus d'entraînement silencieux.

------------------------------------------------------------------------

<a id="directml-cross-entropy-reduction-and-valid-token-normalization"></a>
# Réduction de l'entropie DirectML et normalisation des tokens valides

<a id="problem-17"></a>
## Problème

Sur la chaîne DirectML FP16 testée, l'entropie causale du modèle avec `ignore_index=-100` n'a pas reproduit la normalisation attendue des tokens valides.

La valeur par défaut `reduction="mean"` a divisé la perte effective par toutes les positions décalées au lieu de diviser seulement par les tokens valides et non ignorés.

Un premier remède utilisant :

``` python
loss = F.cross_entropy(x, y, ignore_index=-100, reduction="sum")
loss = loss / (y != -100).sum().clamp_min(1)
```

a été ensuite trouvé incorrect sur la chaîne DirectML FP16 testée.

<a id="cause-15"></a>
## Cause

Les premiers tests CPU/DirectML ont confirmé le dénominateur incorrect de la réduction DirectML lorsqu'il y avait des tokens de remplissage ignorés.

Une seconde comparaison de réduction a ensuite évalué les logits du même modèle avec :

``` text
A = DirectML reduction="sum" / valid tokens
B = DirectML reduction="none" → valid tokens → FP32 mean
C = CPU FP32 reference
```

Les résultats ont été :

``` text
Batch 8:
A = 14.1078
B =  6.6688
C =  6.6715

Batch 32:
A = 10.7772
B =  6.3027
C =  6.3053
```

`B` correspondait à la référence CPU FP32 avec une précision d'environ `0.003`, tandis que `A` ne le faisait pas.

<a id="solution-17"></a>
## Solution

La perte finale de la tâche causale DirectML utilise une entropie par token suivie d'une moyenne en FP32 sur les tokens valides :

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

Après application de ce changement, la perte d'entraînement du modèle sur le test de référence avec un batch de 32 est devenu :

``` text
Model outputs.loss:             6.30274916
DirectML none → FP32 mean:      6.30274916
CPU FP32 reference:             6.30530691
|model - CPU|:                  0.0025577545
```

<a id="decision-17"></a>
## Décision

L'entraînement causale DirectML ne doit pas dépendre ni de la réduction par défaut du backend ni de l'approche testée en FP16 `sum / valid_tokens`.

L'implémentation conservée calcule les pertes par token avec `reduction="none"` et effectue la moyenne des tokens valides en FP32.

------------------------------------------------------------------------

<a id="pretraining-diagnostic-loss-aggregation-mismatch"></a>
# Incohérence de l'agrégation de la perte de diagnostic d'entraînement

<a id="problem-18"></a>
## Problème

Après le premier époque complète de pré-entraînement dense corrigée, `diagnose_pretrain.py` a rapporté des pertes d'ensemble sur le chemin d'entraînement autour de `12–13`, malgré une amélioration de la précision Top-1 et un taux de répétition faible.

<a id="cause-16"></a>
## Cause

L'investigation a initialement suggéré un problème d'agrégation diagnostique, mais le test dédié de réduction a montré que le chemin d'entraînement actuel `reduction="sum" / valid_tokens` était incorrect sur le backend DirectML FP16 testé.

L'indicateur manuel de perte par token a conservé une utilité car il était proche de la référence indépendante sur CPU FP32.

<a id="solution-18"></a>
## Solution

Le diagnostic a été modifié pour utiliser une évaluation par lots, et l'implémentation de la perte du modèle a été remplacée par une perte de croisement par token DirectML suivie d'une moyenne par token valide en FP32.

Les résultats de validation par lots de taille 32 ont été :

``` text
Model outputs.loss:        6.30274916
Manual DirectML loss:      6.30274916
CPU FP32 reference:        6.30530691
Difference vs CPU:         0.0025577545
```

<a id="decision-18"></a>
## Décision

Les diagnostics de point d'arrêt doivent comparer le chemin d'entraînement du modèle avec une référence indépendante sur les logits identiques.

L'implémentation `reduction="none"` → moyenne par token valide en FP32 est le chemin de perte retenu pour DirectML.
------------------------------------------------------------------------

<a id="directml-bf16-tensor-execution-is-unsupported-in-the-current-stack"></a>
# L'exécution des tensors BF16 DirectML n'est pas prise en charge dans la pile actuelle

<a id="problem-19"></a>
## Problème

Une expérience de pré-entraînement similaire à l'origine a tenté d'utiliser directement les tensors du modèle `bfloat16`.

L'exécution a échoué avant la première étape d'entraînement avec :

``` text
Invalid or unsupported data type BFloat16
```

du chemin du backend DirectML.

<a id="cause-17"></a>
## Cause

La pile d'exécution testée PyTorch / `torch-directml` / `PrivateUse1` ne supporte pas le type de tensor BF16 requis pour ce travail.

Ce résultat établit une limitation de la pile logicielle actuelle. Il ne prouve pas par lui-même que le matériel GPU physique ne peut pas effectuer des opérations BF16.

L'expérience diffère également de la chaîne CUDA mixte-precision d'origine : l'origine conserve les paramètres du modèle en FP32 et utilise l'auto-cast BF16 pour les opérations de calcul compatibles, tandis que l'expérience DirectML a dû demander des tensors BF16 directement car une chaîne d'auto-cast DirectML équivalente n'est pas disponible.

<a id="solution-19"></a>
## Solution

Le DirectML BF16 n'est pas utilisé dans le chemin actuel de pré-entraînement MiniMind.

La stratégie mixte-precision conservée reste :

``` text
FP16 model compute
        ↓
FP32 master weights
        ↓
FP32 AdamW update
        ↓
FP32 master → FP16 model synchronization
```

<a id="decision-19"></a>
## Décision

Ne pas considérer le DirectML BF16 comme une alternative utilisable au CUDA BF16 autocast dans l'environnement actuel.

------------------------------------------------------------------------

<a id="fast-direct-fp16-optimizer-re-test"></a>
# Ré-test du optimiseur Direct FP16 rapide

<a id="problem-20"></a>
## Problème

Afin de réduire la surcharge des poids maîtres en FP32, le pré-entraînement dense a été rétesté avec un chemin expérimental rapide utilisant AdamW directement sur les paramètres du modèle en FP16.

La perte d'entraînement en temps réel a resté finie et a diminué, mais les diagnostics de point d'arrêt ont montré une apprentissage insuffisant suivi d'une répétition rapide.

Les résultats sur les échantillons `0–255` ont été :

``` text
Step 1000:
loss   = 9.2573
Top-1  = 0.31%
repeat = 0.29%

Step 2000:
loss   = 8.3102
Top-1  = 0.92%
repeat = 1.20%

Step 5000:
loss   = 7.3196
Top-1  = 1.80%
repeat = 39.24%
```

Le taux de répétition sur les données réelles a resté environ `0.29%`.

<a id="cause-18"></a>
## Cause

Le chemin expérimental a supprimé les paramètres maîtres en FP32 et a appliqué les mises à jour d'AdamW directement aux paramètres en FP16.

Les résultats renforcent la découverte antérieure selon laquelle l'échelle statique de perte peut protéger le signal de propagation inverse sans fournir suffisamment de précision pour des mises à jour fiables des paramètres en FP16 sur une longue durée.

<a id="solution-20"></a>
## Solution

Le chemin rapide d'optimisation directe en FP16 n'est pas conservé.

La pré-entraînement dense DirectML continue d'utiliser le calcul du modèle en FP16 avec des poids principaux en FP32 et des mises à jour AdamW en FP32.

<a id="decision-20"></a>
## Décision

Une perte de formation décroissante n'est pas suffisante pour valider l'optimisation directe en FP16.

Les poids maîtres FP32 restent nécessaires dans l’architecture Dense conservée pour le préentraînement DirectML FP16.

------------------------------------------------------------------------

<a id="diagnostic-cpu-synchronization-limited-to-logging-intervals"></a>
# Synchronisation CPU diagnostique limitée aux intervalles de journalisation

<a id="problem-21"></a>
## Problème

Le formateur stable FP16 + poids principaux FP32 effectuait une synchronisation DirectML-CPU plus fréquemment que nécessaire pour la mise à jour de l'optimiseur.

Exemples inclus :

``` text
finite-loss read on every batch
gradient-norm scalar read on optimizer updates
```

Ces lectures scalaires forcent le CPU à attendre les travaux asynchrones DirectML.

<a id="solution-21"></a>
## Solution

Une variante optimisée du formateur conserve intactement le chemin d'optimisation complet en FP32-principaux, tout en désactivant la matérialisation scalaire par étape pour les vérifications diagnostiques.

La perte et la norme du gradient la plus récente sont transférées au CPU et vérifiées uniquement lors :

``` text
step % log_interval == 0
```

ou au point final de journalisation.

L'opération de clipping du gradient se déclenche toujours à chaque mise à jour de l'optimiseur, et les poids principaux en FP32 restent synchronisés vers le modèle en FP16 après chaque étape de l'optimiseur.

<a id="decision-21"></a>
## Décision

La synchronisation CPU diagnostique doit suivre `log_interval` plutôt que le rythme de l'optimiseur lorsqu'il est possible.

Un benchmark à chaque étape `1000` a confirmé une amélioration de la bande passante de `2.0038 s/step` à `1.9307 s/step`, correspondant à une accélération `1.038×` et à une durée totale réduite `3.65%`.

L'impact sur la bande passante est validé pour la configuration testée. Une validation de qualité de point d'arrêt reste requise avant que le formateur optimisé ne soit considéré comme la référence finale de l'entraînement.

------------------------------------------------------------------------

<a id="m5-final-blocking-issue--directml-fp16-numerical-divergence"></a>
# Problème de blocage M5 — Divergence numérique FP16 DirectML

<a id="problem-22"></a>
## Problème

Le point d'arrêt entraîné par DirectML restait incohérent même lors de son évaluation sous ROCm, tandis que le point d'arrêt entraîné par ROCm produisait du texte cohérent.

Sur les mêmes exemples `1,000` (`204,327` tokens valides), évalués sous ROCm :

```text
DirectML-trained checkpoint: loss 6.880233, perplexity 972.8529
ROCm-trained checkpoint:     loss 1.846587, perplexity   6.3382
```

Cela confirme que le point d'arrêt DirectML lui-même est dégradé ; le problème n'est pas limité à la génération DirectML.

<a id="final-diagnosis"></a>
## Diagnostic final

En utilisant le même point d'arrêt entraîné par DirectML et le même batch fixe (`873` tokens valides) :

```text
DirectML FP16: 5.733902
ROCm FP16:     6.230469

DirectML FP32: 6.231627
ROCm FP32:     6.231628
```

Le chemin FP16 DirectML sans SDPA produisait uniquement des logits non-finis, tandis que ROCm restait stable. En revanche, les versions FP32 DirectML et ROCm étaient en accord à environ `1e-6`.

La perte DirectML FP16 elle-même était internement cohérente avec une recomputation du croisement entropique en FP32 sur le CPU à partir des mêmes logits DirectML (`5.733902` vs `5.736305`). Par conséquent, la réduction de l'entropie conservée n'est pas le problème fondamental restant : le chemin FP16 en avant produit déjà des logits matériellement différents.

<a id="decision-22"></a>
## Décision

Le blocage final est classé comme une **divergence numérique FP16 DirectML** dans la pile testée. Aucun travailaround supplémentaire pour DirectML n'est poursuivi.

L'implémentation DirectML reste archivée pour compatibilité, vérification de retour et reproductibilité. L'entraînement actif de MiniMind continue avec **ROCm**.
