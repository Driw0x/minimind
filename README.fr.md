[English](README.md) | [Français](README.fr.md) | [简体中文](README.zh-CN.md)

<a id="minimind-rocm-windows"></a>
# MiniMind ROCm Windows

Une adaptation Windows + AMD ROCm de [MiniMind](https://github.com/jingyaogong/minimind).

Ce fork conserve la structure du projet original MiniMind au plus près de l'origine tout en validant des entraînements natifs sur GPU AMD sous Windows via ROCm et PyTorch.

> **Backend actif :** ROCm sur Windows   **Cible principale :** AMD Radeon RX 7800 XT (`gfx1101`)

---

<a id="about-this-fork"></a>
## À propos de ce fork

Le projet original MiniMind est principalement développé autour d'environnements PyTorch orientés CUDA.

Ce fork se concentre sur le lancement et la validation du même pipeline d'entraînement sur une carte graphique AMD Radeon sous Windows en utilisant ROCm.

L'objectif n'est pas de réinventer MiniMind. L'objectif est de conserver au maximum la logique d'entraînement d'origine et de faire uniquement les modifications nécessaires pour un chemin d'exécution correct sur ROCm/Windows.

Les priorités actuelles sont :

- conserver les points de contrôle validés Dense Pretrain et Full SFT comme références baselines ;
- valider la qualité des points de contrôle dès le début, et non seulement leur exécution réussie ;
- réduire au minimum et isoler les modifications spécifiques à ROCm ;
- préserver la compatibilité avec la structure d'origine de MiniMind ;
- valider le chemin d'entraînement MoE par rapport à la base Dense.

Les entraînements MoE Pretrain et MoE Full SFT, ainsi que leur évaluation initiale sur AMD ROCm, sont terminés. Les points de contrôle conservés sont `pretrain_768_moe.pth` et `full_sft_768_moe.pth` ; voir la [comparaison de qualité](docs/generation_quality.fr.md).
Dépôt Hugging Face : [Driw0x/minimind-rocm-pytorch](https://huggingface.co/Driw0x/minimind-rocm-pytorch).

Les plans de projet et les commandes réproductibles d'entraînement sont documentés séparément sous [`docs/`](docs/).

---

<a id="legacy-directml"></a>
## Ancienne implémentation DirectML

L'ancienne adaptation Windows + DirectML est conservée comme étude de compatibilité et de faisabilité terminée sous :

**[`directml/`](directml/README.fr.md)**

Elle contient l'implémentation historique, les benchmarks, les tests, la documentation et l'analyse finale de la qualité du modèle. Elle est archivée et n'est pas partie du chemin d'entraînement actif.

---

<a id="rocm-and-pytorch-device-semantics"></a>
## Sémantique des dispositifs ROCm et PyTorch

PyTorch sur ROCm réutilise intentionnellement l'API Python orientée CUDA. Une carte graphique AMD fonctionnant via ROCm est donc adressée avec `cuda` / `cuda:0`.

Ne **pas** remplacer le dispositif PyTorch par `rocm`, `hip`, ou un nom de dispositif personnalisé. Les interfaces PyTorch nommées CUDA sont attendues sur le backend ROCm/HIP.

Pour la vérification de l'environnement exécutable, voir [Vérifier l'environnement ROCm](#verify-the-rocm-environment).

---

<a id="target-environment"></a>
## Environnement cible

Les cibles actives de migration sont :

|Composant|Cible|
| --- | --- |
|Système d'exploitation|Windows|
|GPU|AMD Radeon RX 7800 XT|
|Architecture GPU| `gfx1101` |
|Backend de calcul|AMD ROCm|
|Framework|PyTorch pour ROCm|
|Dispositif d'entraînement|`cuda:0` via le backend HIP de PyTorch|

Les versions de ROCm, de PyTorch et du pilote AMD doivent rester compatibles entre elles.

Avant d'installer ou de mettre à jour l'environnement, vérifiez les informations de compatibilité officielles :

- [Documentation ROCm](https://rocm.docs.amd.com/)
- [Matrice de compatibilité ROCm](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [Installation de PyTorch pour ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

---

<a id="installation"></a>
## Installation

<a id="1-clone-the-repository"></a>
### 1. Cloner le dépôt

```powershell
git clone https://github.com/Driw0x/minimind.git
cd minimind
```

<a id="2-create-the-rocm-virtual-environment"></a>
### 2. Créer l'environnement virtuel ROCm

Utilisez une version de Python prise en charge par la combinaison ROCm/PyTorch sélectionnée dans le tableau de compatibilité d'AMD.

Exemple :

```powershell
py -3.13 -m venv .venv
```

Activez-la :

```powershell
.\.venv\Scripts\Activate.ps1
```

Mettez à jour pip :

```powershell
python -m pip install --upgrade pip
```

<a id="3-install-rocm-enabled-pytorch"></a>
### 3. Installez PyTorch avec support ROCm

Les dépendances spécifiques à ROCm sont conservées séparément dans :

```text
requirements-rocm.txt
```

Installez-les en premier :

```powershell
pip install -r requirements-rocm.txt
```

Cela maintient les paquets spécifiques au backend de PyTorch séparés des dépendances MiniMind de pointe.

<a id="4-install-minimind-dependencies"></a>
### 4. Installez les dépendances MiniMind

Installez les dépendances restantes du projet à partir de :

```powershell
pip install -r requirements.txt
```

L'ordre recommandé d'installation est donc :

```text
ROCm / PyTorch dependencies
        ↓
MiniMind dependencies
```

Les versions de ROCm, de PyTorch et du pilote AMD doivent rester compatibles entre elles. Lorsque vous changez de version, vérifiez la documentation actuelle d'AMD sur la compatibilité.

---

<a id="verify-the-rocm-environment"></a>
## Vérifiez l'environnement ROCm

Avant d'exécuter MiniMind, vérifiez que PyTorch peut accéder à la carte graphique Radeon :

```powershell
python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Caractéristiques attendues :

```text
hip: <ROCm/HIP version>
cuda available: True
device: AMD Radeon RX 7800 XT
```

Un calcul minimal sur GPU peut également être testé avec :

```python
import torch

assert torch.cuda.is_available()
assert torch.version.hip is not None

x = torch.randn(2048, 2048, device="cuda")
y = x @ x

print(y.device)
print(torch.cuda.get_device_name(0))
```

L'appareil du tenseur sera toujours rapporté comme :

```text
cuda:0
```

Ceci est attendu avec PyTorch ROCm.

---

<a id="project-documentation"></a>
## Documentation du projet

La documentation est divisée selon les responsabilités :

- [`docs/roadmap.md`](docs/roadmap.fr.md) — les jalons, les règles de formation mini-vs-pleine, les dépendances par étape et les critères de validation ;
- [`docs/training_commands.md`](docs/training_commands.fr.md) — les commandes de formation reproductibles, les sorties de points de contrôle et l'utilisation de reprise ;

- [`docs/development_tools.md`](docs/development_tools.fr.md) — les outils diagnostiques et les résultats de référence ;
- [`docs/generation_quality.md`](docs/generation_quality.fr.md) — les comparaisons de qualité de génération de points de contrôle ;
- `docs/memory.md` (document absent de cet état du dépôt) — le plan de conception et d'évaluation de la mémoire en temps réel ;
- [`docs/original/`](docs/original/) — la documentation de pointe de MiniMind conservée.

---

<a id="project-structure"></a>
## Structure du projet

```text
minimind/
│
├── dataset/                  # Dataset loading
│
├── docs/                     # Project documentation
│   ├── roadmap.md            # Active milestones and validation plan
│   ├── training_commands.md  # Reproducible training commands
│   ├── development_tools.md  # Diagnostic tools and reference results
│   ├── generation_quality.md # Checkpoint generation-quality comparisons
│   ├── memory.md             # Runtime-memory design and evaluation plan
│   └── original/             # Original MiniMind documentation
│
├── images/                   # Original project images/resources
├── model/                    # Active MiniMind model implementation
├── scripts/                  # Evaluation, conversion and serving utilities
├── trainer/                  # Active ROCm/Windows training code
│
├── eval_llm.py
├── requirements.txt          # MiniMind dependencies
├── requirements-rocm.txt     # ROCm/PyTorch dependencies
├── .gitignore
├── LICENSE
└── README.md
```

Les points de contrôle générés, les environnements virtuels, les caches et les sorties de formation locales sont des artefacts de fonctionnement et ne doivent pas être committés sauf s'ils sont intentionnellement conservés comme preuve expérimentale.

---

<a id="original-minimind-documentation"></a>
## Documentation originale de MiniMind

Les fichiers README d'origine sont conservés sous :

```text
docs/original/
```

- [Fichier README en chinois](docs/original/README.zh-CN.md)
- [Fichier README en anglais](docs/original/README.md)

Pour l'utilisation d'origine, les modèles, les ensembles de données et les détails du projet, consultez ces documents ou le dépôt d'origine :

[github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)

---

<a id="differences-from-upstream"></a>
## Différences par rapport à l'origine

Cette branche se concentre actuellement sur la compatibilité Windows + ROCm d'AMD.

Les différences prévues par rapport à l'origine sont limitées aux éléments nécessaires pour cet environnement, tels que :

- Gestion des dépendances ROCm/Windows ;
- Vérification de l'exécution de PyTorch HIP sur matériel Radeon ;
- Suppression d'hypothèses strictement spécifiques à NVIDIA lorsqu'elles bloquent ROCm ;
- Comportements de précision mixte et de formation compatibles avec ROCm lorsque nécessaire ;
- Vérification du backend et des contrôles de qualité des points de contrôle ;
- Documentation de la configuration Windows AMD.

Comme PyTorch ROCm utilise l'API Python orientée CUDA, les appels PyTorch portant le nom de CUDA ne doivent pas être modifiés simplement parce que le GPU physique est un AMD.

Les modifications doivent rester minimales et ne doivent pas dévier de la logique de formation upstream sans une raison vérifiée.

---

<a id="upstream-project"></a>
## Projet principal

Ce dépôt est basé sur :

[MiniMind — jingyaogong/minimind](https://github.com/jingyaogong/minimind)

MiniMind et le code source original sont développés par leurs auteurs respectifs.

Ce fork se concentre sur l'adaptation et la validation de ce projet pour une formation native sur Windows utilisant des GPU Radeon AMD via ROCm.

---

<a id="license"></a>
## Licence

Ce fork conserve la licence du projet original MiniMind.

Voir [LICENSE](LICENSE) pour les termes de licence applicables.

<!-- multilingual-documentation -->

## Documentation complémentaire

- [dataset/dataset.md](dataset/dataset.fr.md)
- [directml/dataset/dataset.md](directml/dataset/dataset.fr.md)
