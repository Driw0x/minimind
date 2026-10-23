[English](README.md) | [Français](README.fr.md) | [简体中文](README.zh-CN.md)

<a id="minimind-directml--archived-experiment"></a>
# MiniMind DirectML — Expérimente archivé

> **Archivé / historique**
>
> Ce dossier contient l'adaptation précédente de MiniMind pour Windows et DirectML. Elle n'est plus le backend actif de formation.
>
> Le développement actif a été transféré vers **ROCm sur Windows** : [retour à la page principale README](../README.fr.md).

Cet expérimente a étudié la possibilité d'adapter la chaîne de formation orientée CUDA de MiniMind pour s'exécuter sur une carte graphique Radeon AMD sous Windows via `torch-directml`.

L'objectif était de remplacer ou de contourner les hypothèses d'exécution spécifiques à CUDA tout en préservant autant que possible la logique de formation originelle de MiniMind.

L'expérimente a obtenu une compatibilité d'exécution large, mais la validation finale de la qualité du modèle a montré que les points de contrôle localement formés avec DirectML n'étaient pas fiables. L'implémentation est donc conservée comme archive technique plutôt que maintenue comme backend actif.

---

<a id="experiment-outcome"></a>
## Résultat de l'expérimente

L'adaptation DirectML a pu :

- initialiser et utiliser la carte graphique Radeon dédiée via `torch-directml`;
- placer des modèles et des tenseurs sur les dispositifs DirectML;
- exécuter des passes avant et arrière;
- exécuter des étapes d'optimiseur;
- compléter des séries de formation limitées;
- compléter des époques de pré-formation entières;
- sauvegarder et recharger des points de contrôle;
- exécuter les flux principaux de formation de MiniMind via des tests de fumée DirectML.

Cependant, le succès d'exécution n'a pas conduit à des points de contrôle formés correctement.

Le résultat décisif de validation a été :

```text
Official upstream pretrain_768.pth
    CPU inference       -> coherent
    DirectML inference  -> coherent

Locally trained DirectML checkpoints
    epoch 1             -> incoherent
    epoch 2             -> incoherent
```

Les vérifications du tokenizer et des ensembles de données étaient propres, et le point de contrôle officiel upstream était généré de manière cohérente par le même chemin d'évaluation.

Les tests croisés entre backends ont réduit la défaillance à la chaîne numérique FP16 testée avec DirectML. Le point de contrôle formé avec DirectML correspond à ROCm en FP32, mais diverge de manière importante en FP16.

**La formation avec DirectML a été abandonnée pour ce projet.**

<a id="final-quantitative-validation"></a>
### Validation quantitative finale

Sur les mêmes échantillons `1,000` (`204,327` tokens valides), évalués sous ROCm :

|Point de contrôle|Perte|Perplexité|Génération|
| --- | ---: | ---: | --- |
|Formé avec DirectML| 6.880233 | 972.8529 |incohérent|
|Formé avec ROCm| 1.846587 | 6.3382 |cohérent|

En utilisant le même point de contrôle formé avec DirectML et un batch fixe (`873` tokens valides), les pertes DirectML/ROCm étaient `5.733902 / 6.230469` en FP16 mais `6.231627 / 6.231628` en FP32. Le chemin numérique FP16 DirectML sans SDPA produisait des logits non-finis.

Cela élimine la charge du point de contrôle et l'inférence générale comme explication finale et isole le comportement bloquant à la chaîne numérique FP16 testée avec DirectML.

---

<a id="why-this-archive-is-kept"></a>
## Pourquoi cet archive est conservé

Le travail sur DirectML reste utile comme étude de compatibilité documentée.

Il enregistre :

- quels hypothèses CUDA avaient dû être remplacées ou protégées;
- comment `torch-directml` a exposé la carte graphique AMD à PyTorch;
- quels opérateurs nécessitaient un traitement spécial ou des retours vers le CPU;
- comment la formation en FP16 a été adaptée;
- comment la correction de la formation a été testée;
- pourquoi les pertes finies et les époques réussies ne sont pas suffisantes pour valider le modèle ;
- pourquoi les tests de qualité de checkpoint doivent avoir lieu tôt ;

Le code est conservé pour la reproductibilité, la comparaison historique et la référence technique.

Il ne doit pas être utilisé comme implémentation par défaut du training de MiniMind.

---

<a id="historical-tested-configuration"></a>
## Configuration testée historiquement

L'expérience DirectML a été testée avec :

|Composant|Configuration|
| --- | --- |
|Système d'exploitation|Windows|
|GPU|AMD Radeon RX 7800 XT|
|RAM|32 Go|
|Python| 3.10 |
|PyTorch|2.4.1+cpu|
|Backend| `torch-directml` |
|adaptateur DirectML| `directml:1` |
|représentation du dispositif de PyTorch| `privateuseone:1` |

Ces valeurs décrivent uniquement l'expérience archivée.

Elles ne sont pas les exigences de dépendance ou de backend de la version active ROCm du dépôt.

---

<a id="what-was-adapted"></a>
## Ce qui a été adapté

L'implémentation archivée contient des modifications spécifiques à DirectML pour :

- la détection et l'initialisation du dispositif ;
- le remplacement ou la protection de la gestion du dispositif CUDA ;
- la localisation du modèle et des tenseurs ;
- la compatibilité avec le traineur ;
- des opérateurs DirectML non pris en charge ;
- le comportement de fallback sur CPU ;
- des fixtures de training déterministes ;
- des tests de compatibilité entre traineurs ;
- des benchmarks de training réels ;
- pré-entraînement dense en FP16 ;
- échelle de perte limitée ;
- mises à jour du gradient avec poids principaux en FP32 ;
- fonction de perte par token avec moyenne FP32 des tokens valides ;
- validation bornée `--max_steps` ;
- retours de routage MoE pour comportements non pris en charge de scatter ;
- corrections de compatibilité Windows/DirectML pour l'apprentissage par renforcement agent ;
- outils de training séquentiels sur Windows ;
- audit de compatibilité DirectML.

Ces modifications appartiennent à cette implémentation historique et ne doivent pas être copiées dans le code actif ROCm sauf si un problème actuel de ROCm nécessite une modification équivalente.

---

<a id="historical-installation"></a>
## Installation historique

Les commandes suivantes reproduisent le modèle d'environnement utilisé par l'expérience archivée DirectML.

Elles ne sont **pas** les instructions d'installation pour le projet actif ROCm.

<a id="1-enter-the-archived-implementation"></a>
### 1. Entrer dans l'implémentation archivée

À partir de la racine du dépôt :

```powershell
cd directml
```

<a id="2-create-a-python-310-environment"></a>
### 2. Créer un environnement Python 3.10

```powershell
py -3.10 -m venv .venv
```

<a id="3-activate-it"></a>
### 3. L'activer

```powershell
.\.venv\Scripts\Activate.ps1
```

<a id="4-install-the-historical-dependencies"></a>
### 4. Installer les dépendances historiques

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

L'environnement actif ROCm est maintenu séparément à la racine du dépôt.

---

<a id="historical-validation"></a>
## Validation historique

L'implémentation DirectML incluait un ensemble de tests automatisés pour la couche de compatibilité et la chaîne de formation.

Exécuter les tests archivés à partir de ce répertoire :

```powershell
pytest -q
```

Le test de validation plus lourd a utilisé :

```powershell
python tests/test_all_trainers.py
```

Ce test de fumée a exercé une formation bornée DirectML sur les principales workflows et a été utilisé pour prouver la compatibilité d'exécution.

Le fait de passer ces tests n'a pas prouvé la qualité d'un modèle entraîné. Cette distinction a constitué la principale leçon de l'expérience.

---

<a id="known-directml-constraints"></a>
## Contraintes connues de DirectML

L'expérience a rencontré des limitations spécifiques au backend et des opérations non prises en charge.

Certaines opérations nécessitaient un recours au CPU ou des implémentations alternatives. Un exemple connu était le recours associé à `aten::lerp.Scalar_out`.

Les résultats détaillés sont conservés dans :

- [Limites de DirectML](docs/directml_issues.fr.md#current-directml-limitations)
- [Problèmes de DirectML](docs/directml_issues.fr.md)
- [Benchmark de DirectML](docs/directml_benchmarks.fr.md)
- [Audit de DirectML](docs/directml_audit.fr.md)

Ces documents décrivent l'environnement historique de DirectML et ne doivent pas être interprétés comme des limitations de ROCm.

---

<a id="archived-project-structure"></a>
## Structure du projet archivé

```text
directml/
│
├── checkpoints/             # Historical/local training checkpoints
├── dataset/                 # Dataset code used by the experiment
│
├── docs/
│   ├── development-tools.md
│   ├── directml_audit.md
│   ├── directml_benchmarks.md
│   ├── directml_issues.md
│   ├── directml_limitations.md
│   ├── project_memory.md
│   ├── roadmap.md
│   ├── training_commands.md
│   └── update_log.md
│
├── logs/
│   └── directml/            # Preserved training logs
│
├── model/                   # Historical model implementation
├── out/                     # Historical/local generated weights
├── scripts/                 # DirectML development and benchmark utilities
├── tests/                   # DirectML compatibility tests
├── trainer/                 # DirectML-adapted training pipeline
│
├── eval_llm.py
├── requirements.txt
└── README.md
```

Les points de contrôle locaux et les artefacts générés ne sont pas nécessairement suivis par Git selon la configuration du dépôt.

---

<a id="documentation"></a>
## Documentation

<a id="directml-compatibility"></a>
### Compatibilité avec DirectML

- [Limites de DirectML](docs/directml_issues.fr.md#current-directml-limitations) — opérations non prises en charge, contraintes de compatibilité et recours au CPU.
- [Problèmes de DirectML](docs/directml_issues.fr.md) — problèmes rencontrés lors du développement, enquêtes et décisions.
- [Benchmark de DirectML](docs/directml_benchmarks.fr.md) — résultats expérimentaux de compatibilité et de performance.
- [Audit de DirectML](docs/directml_audit.fr.md) — audit du code spécifique à CUDA et des préoccupations de compatibilité avec DirectML.

<a id="historical-development-material"></a>
### Matériel historique de développement

- [Outils de développement](docs/development-tools.fr.md) — audit, génération de fixtures, outils d'entraînement et de benchmarking.
- [Commandes d'entraînement](docs/training_commands.fr.md) — configurations historiques d'entraînement avec DirectML et commandes de reprise.
- [Mémoire du projet](docs/project_memory.fr.md) — décisions architecturales et leçons techniques tirées de l'expérience.
- [Journal des mises à jour](docs/update_log.fr.md) — enregistrement chronologique de l'adaptation à DirectML.
- [Carte de voie](docs/roadmap.fr.md) — carte de voie historique de la phase de développement de DirectML.

---

<a id="original-minimind-documentation"></a>
## Documentation originale de MiniMind

Les fichiers README d'origine sont conservés en dehors de cet archivage :

```text
../docs/original/
```

- [README en chinois original](../docs/original/README.md)
- [README en anglais original](../docs/original/README_en.md)

Le projet original est disponible à :

[MiniMind — jingyaogong/minimind](https://github.com/jingyaogong/minimind)

---

<a id="main-lesson"></a>
## Principal enseignement

Le résultat le plus important de l'expérience DirectML était méthodologique :

```text
training runs
+ finite losses
+ successful checkpoints
≠ validated model training
```

La validation du backend doit également inclure le comportement du modèle résultant.

Le point de contrôle officiel de MiniMind a fonctionné par le même chemin d'inférence sur CPU et DirectML, tandis que les points de contrôle localement entraînés avec DirectML n'ont pas produit de sorties cohérentes. Les tests de compatibilité en FP32 ont été identiques, tandis que les tests en FP16 avec DirectML ont divergé.

C'est pourquoi les travaux d'entraînement ont été transférés vers ROCm au lieu d'ajouter davantage de contournements de précision pour DirectML.

---

<a id="current-project"></a>
## Projet actuel

Pour la mise en œuvre active Windows + AMD, voir :

**[MiniMind ROCm Windows](../README.fr.md)**

---

<a id="license"></a>
## Licence

Cette adaptation archivée reste partie intégrante de la branche MiniMind et suit la licence du dépôt.

Voir [../LICENSE](../LICENSE).
