[English](generation_quality.md) | [Français](generation_quality.fr.md) | [简体中文](generation_quality.zh-CN.md)

[README du projet](../README.fr.md)

<a id="model-output-quality"></a>
# Qualité de la sortie du modèle

<a id="objective"></a>
## Objectif

Documenter et comparer la qualité de génération des points de contrôle de MiniMind en utilisant les prompts d'essai fournis par le projet supérieur.

La comparaison actuelle couvre :

- Point de contrôle de pré-entraînement Mini
- Point de contrôle complet de SFT Mini
- Point de contrôle de pré-entraînement complet
- Point de contrôle complet de SFT

Ce document évalue uniquement les sorties des modèles. Il ne propose pas de nouvelles fonctionnalités d'inférence ni de modifications du pipeline d'entraînement.

---

<a id="current-evaluated-models"></a>
## Modèles actuellement évalués

|Modèle|Données d'entraînement|Statut|
| --- | --- | --- |
|Mini Pré-entraînement|Ensemble de données de pré-entraînement mini|Évalué|
|Mini Full SFT|Ensemble de données de pré-entraînement mini + ensemble de données de SFT mini|Évalué|
|Full Pretrain|Ensemble complet de données de pré-entraînement|Évalué|
|Full SFT|Ensemble complet de données de pré-entraînement + ensembles de données de SFT|Évalué|

Les modèles Mini actuels contiennent approximativement :

```text
Model parameters: 63.91M
```

---

<a id="quantitative-reference"></a>
## Référence quantitative

Mesures de validation actuelles :

|Métrique|Mini Pré-entraînement|Mini Full SFT|
| --- | ---: | ---: |
|Perte sur les données de pré-entraînement| 1.8275 | 1.9992 |
|Perte sur les données de SFT| 2.5207 | 1.5663 |
|Répétition externe| 0.795 | 0.361 |

Le point de contrôle Full SFT montre donc une amélioration claire sur l'objectif de SFT et une génération nettement moins répétitive.

La perte légèrement plus élevée sur les données de pré-entraînement après SFT est prévue être interprétée séparément de la qualité de génération.

<a id="full-data-dense-reference"></a>
### Référence dense sur l'ensemble des données

Les points de contrôle Dense actuels ont été entraînés sur les ensembles complets de données T2T et évalués sur les mêmes ensembles de référence Mini (`seed=42`, 256 échantillons chacun) :

|Métrique|Full Pretrain|Full SFT|
| --- | ---: | ---: |
|Perte sur les données de référence de pré-entraînement| 1.6372 | 1.8730 |
|Perte sur les données de référence de SFT| 2.8685 | 1.3771 |
|Répétition externe|Non conservée comme fiable| 0.244 |

L'agrégat de répétition du Full Pretrain n'est pas conservé car plusieurs générations ont atteint la limite de tokens lors de la décodage, produisant une réponse visible vide.


---

<a id="upstream-generation-tests"></a>
# Tests de génération du projet supérieur

<a id="evaluation-method"></a>
## Méthode d'évaluation

Les deux points de contrôle sont évalués à l'aide du mode d'essai automatique fourni déjà par le projet supérieur MiniMind.

Le même :

- script d'inférence ;
- configuration de génération ;
- tokenizeur ;
- prompts intégrés ;
- architecture du modèle ;

doit être utilisé pour les deux points de contrôle afin de faire isoler l'effet de l'entraînement.

Aucun benchmark personnalisé n'est introduit à cette étape.

---

<a id="mini-pretrain"></a>
## Mini Pré-entraînement

<a id="general-behavior"></a>
### Comportement général

Le point de contrôle de pré-entraînement Mini peut générer du texte, mais ses sorties montrent des limitations importantes pour l'utilisation conversationnelle.

Caractéristiques observées :

- suivi faible des instructions ;
- répétition fréquente ;
- continuation instable ;
- capacité limitée à produire une réponse directe ;
- structure conversationnelle faible ;
- contenu factuel peu fiable ;
- qualité de raisonnement limitée.

La mesure élevée d'external repetition (`0.795`) est cohérente avec le comportement répétitif visible dans la génération.

<a id="interpretation"></a>
### Interprétation

Le point d'arrêt démontre que la pré-formation a produit un modèle de langage fonctionnel, mais la pré-formation seule est insuffisante pour un comportement fiable d'obéissance aux instructions.

Il sert principalement de référence contre laquelle le Full SFT est évalué.

---

<a id="mini-full-sft"></a>
## Mini Full SFT

<a id="general-behavior-1"></a>
### Comportement général

Le point d'arrêt Mini Full SFT produit des sorties notablement plus structurées et conversationnelles que le point d'arrêt Mini Pré-formation.

Améliorations observées incluent :

- meilleure obéissance aux instructions ;
- réponses plus directes ;
- structure conversationnelle améliorée ;
- répétition considérablement réduite ;
- génération plus stable.

La mesure d'external repetition diminue de :

```text
0.795 -> 0.361
```

et la perte sur les données de SFT diminue de :

```text
2.5207 -> 1.5663
```

Ces résultats sont cohérents avec l'amélioration qualitative observée dans les tests automatiques supérieurs.

<a id="remaining-limitations"></a>
### Limites restantes

Les réponses générées peuvent être cohérentes sans être factuellement correctes.

Les limites actuelles incluent donc :

- connaissance factuelle peu fiable ;
- réponses incorrectes exprimées dans un langage plausibles ;
- capacité de raisonnement limitée ;
- possibilité de hallucinations ;
- sensibilité aux ensembles de données d’entraînement limités.

Une réponse cohérente doit donc être interprétée comme une réponse correcte.

---

<a id="mini-pretrain-vs-mini-full-sft"></a>
# Mini Pré-formation vs Mini Full SFT

|Critère|Mini Pré-formation|Mini Full SFT|
| --- | --- | --- |
|Génération de texte|Fonctionnelle|Fonctionnelle|
|Structure conversationnelle|Faible|Améliorée|
|Obéissance aux instructions|Faible|Améliorée|
|Répétition|Élevée|Beaucoup plus faible|
|Cohérence des réponses|Limitée|Améliorée|
|Fiabilité factuelle|Limitée|Toujours limitée|
|Qualité de raisonnement|Limitée|Toujours limitée|

<a id="current-conclusion"></a>
## Conclusion actuelle

Le Full SFT apporte une amélioration claire dans **la manière dont le modèle répond**.

Il améliore la stabilité conversationnelle, l'obéissance aux instructions et la répétition par rapport au point d'arrêt Mini Pré-formation.

Cependant, le modèle actuel Mini Full SFT ne devrait pas encore être considéré comme fiable factuellement.

Les résultats actuels démontrent donc :

```text
Mini Pretrain
    -> basic language modeling capability

Mini Full SFT
    -> improved instruction-following and conversational behavior
    -> factual and reasoning limitations remain
```

---

<a id="full-dataset-evaluation"></a>
# Évaluation sur l'ensemble complet

Le même protocole d'évaluation a maintenant été appliqué aux points d'arrêt Dense formés sur les ensembles complets de MiniMind T2T.

<a id="full-pretrain"></a>
## Full Pretrain

Le checkpoint de pré-entraînement complet est plus cohérent dans les tests automatiques en amont que le checkpoint de pré-entraînement miniaturisé et atteint une perte de référence de pré-entraînement inférieure (`1.6372` vs `1.8275`).

Il montre encore des répétitions, des dérives de continuation, des inexactitudes factuelles et un comportement limité en matière de suivi d'instructions, ce qui est attendu avant le SFT.

<a id="full-sft"></a>
## SFT complet

Le checkpoint de SFT complet produit des réponses cohérentes et stables au style d'assistant et améliore la perte de référence de SFT de `2.8685` à `1.3771`.

Son métrique externe déterministe de répétition est `0.244`.

Une vérification manuelle révèle encore des limitations connues :

- l'exactitude factuelle reste peu fiable ;
- la résolution arithmétique peut échouer ;
- la correction de code peut rester incorrecte ;
- le suivi strict des instructions est imparfait.

L'ensemble du dataset de SFT améliore donc le comportement et la stabilité de l'assistant, mais ne rend pas le modèle de 63,91M fiablement factuel ou correct en raisonnement.

<a id="current-comparison"></a>
## Comparaison actuelle

|Critère|Pré-entraînement miniaturisé|Pré-entraînement complet|SFT complet (mini)|SFT complet (plein)|
| --- | --- | --- | --- | --- |
|Suivi d'instructions|Faible|Toujours limité|Amélioré|Amélioré mais imparfait|
|Structure conversationnelle|Faible|Amélioré|Amélioré|Amélioré|
|Répétition|Élevée|Toujours présente|Réduite|Faible|
|Fiabilité factuelle|Limitée|Limitée|Limitée|Toujours limitée|
|Qualité de raisonnement|Limitée|Limitée|Limitée|Toujours limitée|
|Stabilité globale des sorties|Faible|Améliorée|Améliorée|Améliorée|

---

<a id="final-status"></a>
# Statut final

<a id="mini-models"></a>
## Modèles miniaturisés

- [x] Pré-entraînement miniaturisé effectué
- [x] SFT complet (mini) effectué
- [x] Tests automatiques en amont exécutés
- [x] Pré-entraînement miniaturisé et SFT complet (mini) comparés
- [x] Amélioration conversationnelle du SFT complet observée
- [x] Limites restantes en matière d'exactitude factuelle et de raisonnement identifiées

<a id="full-data-models"></a>
## Modèles à données complètes

- [x] Entraînement complet du pré-entraînement complet
- [x] Évaluation du pré-entraînement complet avec des tests automatiques en amont
- [x] Entraînement complet du SFT complet
- [x] Évaluation du SFT complet avec des tests automatiques en amont
- [x] Comparaison des checkpoints miniaturisés et à données complètes
- [x] Documentation de l'amélioration des performances en matière d'exactitude factuelle, de raisonnement et de génération avec les ensembles de données complets
