[English](README_en.md) | [简体中文](README.md) | [Français](README.fr.md)

[README du projet](../../README.fr.md)

<div align="center">

![logo](../../images/logo.png)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jingyaogong/minimind) [![GitHub Repo stars](https://img.shields.io/github/stars/jingyaogong/minimind?style=social)](https://github.com/jingyaogong/minimind/stargazers) [![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind)](../../LICENSE) [![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind)](https://github.com/jingyaogong/minimind/commits/master) [![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/jingyaogong/minimind/pulls) [![Collection](https://img.shields.io/badge/🤗-MiniMind%20%20Collection-blue)](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

</div>

<div align="center">

![GitHub Trend](https://trendshift.io/api/badge/repositories/12586)

</div>

<div align="center">
  <h3>"La Grande Voie est Simple"</h3>
</div>

<div align="center">

Français | [English](README_en.md) | [简体中文](README.md)

</div>

* Ce projet open-source vise à entraîner MiniMind, un modèle linguistique extrêmement léger avec environ 64 millions de paramètres, entièrement de zéro, avec un coût d'environ 3 yuans et une durée d'entraînement de 2 heures.
* La série MiniMind est intentionnellement légère. Le modèle le plus petit sur la branche principale est d'environ $\frac{1}{2700}$ la taille de GPT-3, ce qui rend l'entraînement complet et la reproduction possibles même sur des GPU personnels ordinaires.
* Le projet propose une architecture de modèle minimaliste et une chaîne d'entraînement complète pour un modèle de langage (LLM), couvrant les architectures MoE, le nettoyage des données, l'entraînement préalable, le Fine-Tuning supervisé (SFT), LoRA, RLHF (DPO), RLAIF (PPO / GRPO / CISPO), l'utilisation d'outils, le RL agissant en tant qu'agent, la pensée adaptative et la distillation de modèles.
* MiniMind a également été étendu à un modèle visuel [MiniMind-V](https://github.com/jingyaogong/minimind-v), un modèle multimodal Omni [MiniMind-O](https://github.com/jingyaogong/minimind-o), un modèle de langage basé sur la diffusion (MiniMind-dLM) et un modèle à attention linéaire (MiniMind-Linear). Voir [Discussion](https://github.com/jingyaogong/minimind/discussions) pour les détails.
* Tous les algorithmes fondamentaux sont implémentés directement dans PyTorch natif, sans recourir à des abstractions de haut niveau provenant de bibliothèques tierces.
* MiniMind est à la fois une reproduction ouverte, de bout en bout, de la chaîne de formation des modèles de langage à grande échelle (LLM) et un tutoriel pratique permettant d'apprendre à construire ces modèles.
* Nous espérons que ce projet puisse offrir un point de départ réproducible, compréhensible et extensible à plus de monde, partager la joie de la création et aider à avancer la communauté de l'intelligence artificielle.

> Note : Ce projet est distribué sous la licence Apache 2.0 et est entièrement gratuit. « 2 heures » désigne le temps mesuré nécessaire à l'exécution de `1 epoch` dans la phase SFT sur une seule carte NVIDIA 3090, tandis que « RMB 3 » correspond au coût de location de cette carte GPU.

---

<div align="center">

![minimind-3](./images/minimind-3.gif)

[🔗 Démo en ligne](https://www.modelscope.cn/studios/gongjy/MiniMind) | [🔗 Introduction vidéo](https://www.bilibili.com/video/BV12dHPeqE72)


<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://huggingface.co/collections/jingyaogong/minimind" style="text-decoration: none;">
          <img src="./images/with_huggingface.png" alt="Hugging Face Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
      <td align="center">
        <a href="https://www.modelscope.cn/profile/gongjy" style="text-decoration: none;">
          <img src="./images/with_modelscope.png" alt="ModelScope Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
    </tr>
  </table>
</div>


</div>

---

<a id="-project-introduction"></a>
# 📌 Présentation du projet

L'apparition des modèles de langage à grande échelle (LLMs) a suscité une attention mondiale sans précédent en matière d'intelligence artificielle. ChatGPT, DeepSeek, Qwen et de nombreux autres modèles ont impressionné les gens par leur performance remarquable, rendant l'impact de cette vague technologique très réel. Cependant, les modèles comportant des dizaines ou des centaines de milliards de paramètres sont non seulement difficiles à former sur des appareils personnels, mais sont souvent hors portée même pour leur déploiement. Découvrir le « boîte noire » des grands modèles et comprendre véritablement comment ils fonctionnent à l'intérieur devrait être une chose excitante. Malheureusement, la plupart des explorations s'arrêtent finalement à l'application de techniques telles que LoRA pour affiner des modèles existants sur quelques nouvelles instructions ou des tâches spécifiques. Cela ressemble davantage à enseigner à Newton à utiliser un smartphone du XXIe siècle — intéressant, mais pas exactement l'objectif initial de comprendre les fondements de la physique.

En même temps, les frameworks et outils tiers tels que `transformers` / `trl` / `peft` exposent souvent uniquement des interfaces très abstraites. Avec seulement une dizaine de lignes de code, on peut réaliser l'ensemble de la chaîne de traitement "charger le modèle + charger le jeu de données + inférence + apprentissage par renforcement". Ce type d'encapsulation efficace est pratique, mais il sépare les développeurs de l'implémentation sous-jacente, réduisant ainsi l'opportunité de comprendre en profondeur le code central des modèles de langage. Je crois que "construire un avion à partir de blocs Lego par soi-même est bien plus excitant que de voyager en première classe". Un problème plus pratique est que l'internet est également rempli de cours payants et de contenus marketing, où les tutoriels sur l'IA sont emballés dans des explications erronées et partiellement comprises. C'est pourquoi l'intention initiale de ce projet est de réduire au maximum l'obstacle à l'apprentissage des modèles de langage, afin que chacun puisse commencer par comprendre chaque ligne de code et entraîner un modèle de langage minuscule à partir de zéro. Oui, **entraînement à partir de zéro**, et non simplement rester au niveau de l'**inférence**. Avec un coût de serveur inférieur à 3 yuans, vous pouvez personnellement vivre l'ensemble du processus de construction d'un modèle de langage de 0 à 1.

😊 Partageons ensemble le plaisir de la création !

---

<a id="-this-project-includes-the-following"></a>
#### 🎉 Ce projet inclut les éléments suivants

- Fournit l'implémentation complète de l'architecture MiniMind-LLM (Dense + MoE), alignée avec l'écosystème `Qwen3 / Qwen3-MoE`.
- Fournit le tokeniseur et le code de formation du tokeniseur, avec le support de tokens de modèle tels que `<tool_call>`, `<tool_response>`, `<think>`, etc.
- Couvre des pipelines d'entraînement intégrés, allant de la pré-entraînement, SFT, LoRA, RLHF-DPO, RLAIF (PPO / GRPO / CISPO), utilisation d'outils, apprentissage par renforcement agissant, pensée adaptative et distillation de modèles.
- Fournit des données ouvertes pour toutes les étapes, couvrant des jeux de données de haute qualité collectés, distillés, nettoyés et déduplicés.
- Les algorithmes principaux d'entraînement et les modules centraux sont tous implémentés de zéro, sans dépendre de wrappers de frameworks tiers.
- Compatible avec les frameworks dominants tels que `transformers`, `trl`, `peft`, ainsi que les moteurs d'inférence courants tels que `llama.cpp`, `vllm`, `ollama`, et les frameworks d'entraînement tels que `Llama-Factory`.
- Supporte l'entraînement sur un nœud unique avec une seule carte GPU et sur un nœud avec plusieurs cartes GPU (DDP, DeepSpeed), la visualisation avec wandb / swanlab, ainsi que le ralentissement ou la reprise dynamique de l'entraînement.
- Supporte l'évaluation sur des ensembles de benchmarks tiers tels que C-Eval, C-MMLU, OpenBookQA, etc., et permet l'extrapolation de contexte long via YaRN.
- Fournit un serveur léger compatible avec OpenAI, pour intégration avec des interfaces de chat tierces telles que FastGPT et Open-WebUI, avec le support de `reasoning_content`, `tool_calls` et `open_thinking`.
- Fournit une interface web minimaliste basée sur Streamlit, permettant l'affichage de la pensée, le choix d'outils et les appels d'outils multi-étapes.
- Comprend des extensions expérimentales : modèle de langage à diffusion ([dLM](https://github.com/jingyaogong/minimind/discussions/618)) et modèle à attention linéaire ([Linear Attention](https://github.com/jingyaogong/minimind/discussions/704)), lesquels peuvent être entraînés à partir du modèle autoregressif principal.

<a id="-released-model-list"></a>
#### 🎉 Liste des modèles publiés

|Modèle|Paramètres|Publication|
|------|--------|---------|
|minimind-3|64M| 2026.04.01 |
|minimind-3-moe|198M-A64M| 2026.04.01 |
|minimind2-small|26M| 2025.04.26 |
|minimind2-moe|145M| 2025.04.26 |
|minimind2|104M| 2025.04.26 |
|minimind-v1-small|26M| 2024.08.28 |
|minimind-v1-moe|4×26M| 2024.09.17 |
|minimind-v1|108M| 2024.09.01 |

---

<a id="-changelog"></a>
#### 📝 Changelog

<details>
<summary> <b>🔥 2026-04-01</b> </summary>

 - Version `minimind-3` / `minimind-3-moe` publiée : mises à jour complètes de la structure, du Tokenizer, du pipeline de formation, de l'interface d'inférence et de la configuration par défaut
- La structure principale est alignée avec l'écosystème `Qwen3 / Qwen3-MoE` : réseau dense approximativement `64M`, MoE approximativement `198M-A64M`, et le design d'expert partagé a été supprimé
- Les données d'entraînement par défaut ont été remplacées par `pretrain_t2t(_mini).jsonl`, `sft_t2t(_mini).jsonl`, `rlaif.jsonl`, `agent_rl.jsonl` et `agent_rl_math.jsonl`
- La version autonome `train_reason.py` a été supprimée ; la capacité de pensée est désormais unifiée grâce à `chat_template + <think>` et au contrôle adaptatif `open_thinking`
- La fonctionnalité `toolcall` a été intégrée dans le jeu principal `sft_t2t / sft_t2t_mini`, la configuration par défaut `full_sft` dispose déjà d'une capacité de commande d'outils de base ; des exemples d'inférence tels que `scripts/chat_api.py` ont également été ajoutés
- Un script d'entraînement natif `Agentic RL`, `train_agent.py`, a été ajouté, permettant le support de `GRPO / CISPO` dans des scénarios d'utilisation d'outils multi-tours
- Le pipeline d'entraînement RLAIF / Agentic RL a achevé la déconnexion `rollout engine`, permettant un switch plus flexible entre les back-ends de génération
- `serve_openai_api.py` et `web_demo.py` ont été mises à jour pour supporter `reasoning_content` / `tool_calls` / `open_thinking`
- Le Tokenizer a été mis à jour sur la base de `BPE + ByteLevel`, avec de nouveaux tokens de commande d'outils et de pensée, et des tokens de tampon réservés pour une extension future
- Un pipeline de fusion et d'exportation des poids LoRA a été ajouté, permettant de fusionner les poids de modèle de base et les poids LoRA pour obtenir de nouveaux poids complets via `scripts/convert_model.py`
- Les ressources de diagramme de structure ont été mises à jour, la README a été largement révisée

</details>

<details>
<summary> <b>2025-10-24</b> </summary>

- 🔥 Algorithmes d'entraînement RLAIF ajoutés : PPO, GRPO, SPO (implémentés de zéro)
- Fonctionnalité de reprise d'entraînement ajoutée : permet une récupération automatique de l'entraînement, une reprise en cas de perte de GPU, et une continuité des enregistrements dans WandB
- Dataset RLAIF ajouté : rlaif-mini.jsonl (10 000 entrées tirées aléatoirement du jeu de données SFT) ; le jeu de données DPO a été simplifié, des données en chinois ont été ajoutées
- Algorithme YaRN ajouté : permet une extrapolation de longueurs de séquence RoPE, améliorant ainsi la capacité de traitement de séquences longues
- Pensée adaptative : le modèle de raisonnement peut optionnellement activer la chaîne de pensée
- Le modèle chat_template prend en charge pleinement les étiquettes de commande d'outils et de raisonnement (`<tool_call>`, `<think>`, etc.)
- Un chapitre complet sur RLAIF a été ajouté, avec une comparaison des courbes d'entraînement et des explications des principes des algorithmes
- [SwanLab](https://swanlab.cn/) remplace WandB (accessibilité domestique améliorée, API entièrement compatible)
- Tous les codes ont été standardisés et certains bugs connus ont été corrigés

</details>

<details>
<summary> <b>2025-04-26</b> </summary>

- Maj importante
- Pour des besoins de compatibilité, consultez [🔗Contenu du dépôt ancien🔗](https://github.com/jingyaogong/minimind/tree/7da201a944a90ed49daef8a0265c959288dff83a).
- Les paramètres du modèle MiniMind ont été complètement renommés, alignés avec les modèles de la bibliothèque Transformers (nommage unifié).
- La méthode generate a été refactorisée, héritant de la classe GenerationMixin.
- 🔥 Support des écosystèmes populaires tiers tels que llama.cpp, vllm, ollama.
- Structure du code et des dossiers standardisés.
- Mise à jour du vocabulaire `<s></s>` -> `<|im_start|><|im_end|>`

```text
To be compatible with third-party inference frameworks llama.cpp, vllm, this update comes with some considerable costs.
This update no longer supports "directly" loading old models from before 25-04-26 for inference.
Due to differences between Llama's positional encoding method and minimind's, QK values differ after mapping to the Llama model.
The minimind2 series old models were all recovered through weight mapping + (fine-tuning) QKVO linear layer calibration.
After this update, maintenance for the entire `minimind-v1` series will be discontinued and taken offline from the repository.
```

</details>

<details>
<summary> <b>Plus...</b> </summary>

**2025-02-09**
- Mise à jour majeure depuis la sortie, Sortie de la série minimind2.
- Code presque entièrement refacto, avec une structure unifiée plus concise et claire. Pour les besoins de compatibilité avec le code ancien, consultez [🔗Contenu du dépôt ancien🔗](https://github.com/jingyaogong/minimind/tree/6e9cd28ef9b34a0a10afbdf6f59e65cb6e628efb).
- Élimination des étapes de prétraitement des données. Format unifié du jeu de données, passage au format `jsonl` afin d'éviter les problèmes liés à la télécharge des jeux de données.
- La série minimind2 présente une amélioration significative de la performance par rapport à MiniMind-V1.
- Problèmes mineurs : {implémentation plus standard du cache kv, perte de balance MoE maintenant prise en compte, etc.}
- Propose une solution de formation pour migrer des modèles vers des données privées (modèle médical, exemples de conscience auto-évaluée).
- Jeu de données de pré-formation simplifié et qualité de données de pré-formation considérablement améliorée, réduction importante du temps nécessaire pour une formation rapide, reproduisible en 2 heures sur un seul 3090 !
- Mise à jour : l'ajustement LoRA décollé du wrapper peft, le processus LoRA implémenté de zéro ; l'algorithme DPO implémenté de zéro à l'aide de PyTorch ; la distillation en blanc de modèle implémentée de zéro.
- Les modèles distillés de la série minimind2-DeepSeek-R1 sont nés !
- La série minimind2 possède certaines capacités en langue anglaise !
- Résultats de tests de performance de minimind2 mis à jour par rapport à des modèles tiers sur plus de plateformes leader de modèles LLM.

**2024-10-05**
- Capacité multimodale étendue pour MiniMind --- Vision
- Consultez le projet frère [minimind-v](https://github.com/jingyaogong/minimind-v) pour plus de détails !

**2024-09-27**
- Mise à jour du 09-27 de la méthode de prétraitement du jeu de données de pré-formation, pour garantir l'intégrité du texte, l'abandon de la prétraitement en format .bin pour la formation (petite perte de vitesse de formation).
- Le fichier prétraité de pré-formation actuel est nommé : pretrain_data.csv.
- Suppression de certains codes redondants.

**2024-09-17**
- Mise à jour du modèle minimind-v1-moe
- Pour éviter toute ambiguïté, le tokeniseur mistral_tokenizer n'est plus utilisé, tous les tokeniseurs utilisent désormais le tokeniseur custom minimind_tokenizer.

**2024-09-01**
- Mise à jour du modèle minimind-v1 (108M), utilisant minimind_tokenizer, 3 époques de pré-formation + 10 époques de formation par apprentissage, formation plus approfondie, performance plus forte.
- Le projet est déployé sur ModelScope Creative Space, il peut être expérimenté sur ce site :
- [🔗Démonstration en ligne sur ModelScope🔗](https://www.modelscope.cn/studios/gongjy/minimind)

**2024-08-27**
- Première publication ouverte du projet

</details>

---

<a id="-quick-start"></a>
# 📌 Démarrage rapide

<details>
<summary>Configuration matérielle et logicielle (à titre de référence)</summary>

* CPU : Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM : 128 Go
* GPU : NVIDIA GeForce RTX 3090 (24 Go) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10.16
* [requirements.txt](../../requirements.txt)

</details>

<a id="step-0"></a>
## Étape 0

```bash
# Clone repository and install dependencies
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind && pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

<a id="ⅰ--model-inference"></a>
## Ⅰ 🚀 Inférence du modèle

<a id="1-download-model"></a>
### 1' Téléchargement du modèle

Dans le répertoire racine du projet :
```bash
# Method 1
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# Method 2
git clone https://huggingface.co/jingyaogong/minimind-3
```

<a id="2-cli-inference"></a>
### 2' Inférence via CLI

```bash
# Method 1: Using Transformers format model
python eval_llm.py --load_from ./minimind-3
# Method 2: Based on PyTorch model (ensure corresponding weights are in the ./out directory)
python eval_llm.py --load_from ./model --weight full_sft
```

<a id="3-optional-webui"></a>
### 3' (Optionnel) WebUI

```bash
# May need `python>=3.10`, install `pip install streamlit`
# ⚠️ You must first copy the transformers-format model folder into ./scripts/ (e.g.: cp -r minimind-3 ./scripts/minimind-3). The web_demo script will auto-scan subdirectories containing weight files; it will throw an error if none are found.
cd scripts && streamlit run web_demo.py
```

<a id="4-optional-third-party-inference-frameworks"></a>
### 4' (Optionnel) Frameworks d'inférence tiers

```bash
# ollama
ollama run jingyaogong/minimind-3
# vllm
vllm serve /path/to/model --served-model-name "minimind"
```

<a id="ⅱ-️-model-training"></a>
## Ⅱ 🛠️ Formation du modèle

<details>
<summary>Note: Vérifiez à l'avance le backend disponible pour Torch</summary>

```python
import torch
print(torch.cuda.is_available())
```

Si vous prévoyez d'utiliser CUDA pour le training, il est recommandé de confirmer d'abord si l'environnement actuel a correctement reconnu le GPU. Si `cuda` n'est pas disponible, vous pouvez tout de même choisir `CPU` ou `MPS` en fonction de votre matériel, mais la vitesse de training et la compatibilité seront considérablement différentes. Si vous devez installer ou changer les versions de PyTorch, consultez [torch_stable](https://download.pytorch.org/whl/torch_stable.html) et [ce lien](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

<a id="1-download-data"></a>
### 1' Télécharger les données

Téléchargez les fichiers de données nécessaires depuis le lien [téléchargement du dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) fourni ci-dessous, et placez-les dans le dossier `./dataset`

> Actuellement, par défaut, vous n'avez besoin que de télécharger `pretrain_t2t_mini.jsonl` et `sft_t2t_mini.jsonl` pour reproduire rapidement le modèle de dialogue `MiniMind Zero`.
Pour d'autres cas d'usage, plusieurs combinaisons de données sont fournies ci-dessous et peuvent être sélectionnées en fonction de vos objectifs de tâche et de vos ressources GPU.

<a id="2-start-training"></a>
### 2' Lancer le training

<details>
<summary>💡 Pause et reprise du point d'arrêt</summary>

Tous les scripts de training supportent la sauvegarde de points d'arrêt. Après avoir ajouté le paramètre `--from_resume 1`, le script détecte automatiquement la progression du training et le reprend :

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
# ...
```

**Instructions pour la reprise du point d'arrêt :**
- Le processus de training sauvegarde automatiquement des points d'arrêt complets (modèle, optimiseur, progression du training, etc.) dans le dossier `./checkpoints/`
- Nom des fichiers de point d'arrêt : `<weight_name>_<dimension>_resume.pth` (ex. : `full_sft_512_resume.pth`)
- Supporte la reprise sur différents nombres de GPU (ajustement automatique de l'étape)
- Prend en charge la continuité des enregistrements de formation avec wandb (reprend automatiquement la même session)

> Adapté à des sessions de formation longues ou à des environnements instables, aucune préoccupation quant à la perte de progrès en cas d'interruption de formation

</details>

<a id="21-pretraining-required"></a>
#### 2.1 Pré-formation (obligatoire)

```bash
cd trainer && python train_pretrain.py
```

> Après formation, les poids de sortie `out/pretrain_*.pth` sont générés (où `*` est la dimension du modèle, valeur par défaut `768`)

<a id="22-instruction-fine-tuning-required"></a>
#### 2.2 Fine-tuning par instruction (obligatoire)

```bash
cd trainer && python train_full_sft.py
```

> Après formation, les poids de sortie `out/full_sft_*.pth` sont générés (où `full` indique un fine-tuning sur tous les paramètres)

<a id="23-test-trained-model-optional"></a>
#### 2.3 Test du modèle entraîné (optionnel)

Vérifiez que les fichiers du modèle à tester `*.pth` se trouvent dans le dossier `./out/` ; vous pouvez également vous rendre directement [ici](https://www.modelscope.cn/models/gongjy/minimind-3-pytorch/files) pour télécharger les poids pré-entraînés `*.pth`.

```bash
python eval_llm.py --weight full_sft
```

> `--weight` est utilisé pour spécifier le préfixe du nom des poids, par exemple `pretrain`, `full_sft`, etc. ; pour plus de paramètres, consultez directement `eval_llm.py`

<details>
<summary>Remarque : Autres informations</summary>

1. Tous les scripts d’entraînement sont nativement conçus sur PyTorch et prennent en charge l'accélération multi-GPU.

2. Si votre appareil dispose de `N (N > 1)` cartes graphiques GPU, vous pouvez démarrer une formation sur une seule machine `N`-GPU comme suit (DDP, également supporte l'extension à plusieurs machines et plusieurs GPU) :

```bash
torchrun --nproc_per_node N train_xxx.py
```

3. Vous pouvez activer wandb pour enregistrer le processus de formation selon vos besoins.

```bash
... train_xxx.py --use_wandb
```
Après juin `2025`, les environnements réseau domestiques en Chine ne peuvent généralement pas se connecter directement à WandB. MiniMind utilise par défaut [SwanLab](https://swanlab.cn/) comme outil de visualisation de formation, dont l'interface est largement compatible avec WandB ; en général, il suffit de remplacer `import wandb` par `import swanlab as wandb`, et l'utilisation reste principalement inchangée.

</details>

---

<a id="-data-introduction"></a>
# 📌 Présentation des données

<a id="ⅰ-tokenizer"></a>
## Ⅰ Tokenizer

Un tokenizer peut être compris grossièrement comme un "dictionnaire" utilisé par les modèles de langage, chargé de mapper le langage naturel aux identifiants de tokens et de décoder ces identifiants en texte ; le projet fournit également `train_tokenizer.py` comme exemple de formation d'un vocabulaire. Il n'est pas recommandé de re-entraîner le tokenizer, car une modification du vocabulaire ou des règles de segmentation affecte directement les poids du modèle, les formats de données, les interfaces d'inférence et la compatibilité avec l'écosystème communautaire, rendant ainsi le modèle plus difficile à partager. De plus, le tokenizer influence des métriques telles que le PPL calculées par token, donc lors de comparaisons entre tokenizers, le métrique BPB (Bits Per Byte) est souvent plus appropriée. Voir [cet article](https://skeptric.com/perplexity/). Pour les petits modèles comme MiniMind, la taille du vocabulaire affecte directement la proportion des paramètres des couches d'embedding et de sortie, donc garder un vocabulaire compact est généralement une meilleure compromis.

<details>
Introduction du tokenizer <summary></summary>

Les tailles de vocabulaire des modèles open-source puissants tiers tels que Yi, Qwen2, ChatGLM, Mistral et Llama 3 sont les suivantes :

<table>
  <tr><th>Tokenizer Model</th><th>Vocabulary Size</th><th>Source</th></tr>
  <tr><td>Yi</td><td>64,000</td><td>01.AI (Chine)</td></tr>
  <tr><td>Qwen2</td><td>151,643</td><td>Alibaba Cloud (Chine)</td></tr>
  <tr><td>ChatGLM</td><td>151,329</td><td>Zhipu AI (Chine)</td></tr>
  <tr><td>Mistral</td><td>32,000</td><td>Mistral AI (France)</td></tr>
  <tr><td>Llama 3</td><td>128,000</td><td>Meta (États-Unis)</td></tr>
  <tr><td>MiniMind</td><td>6,400</td><td>Custom</td></tr>
</table>

> La branche principale utilise de manière cohérente `minimind_tokenizer` afin d'éviter l'ambiguïté des versions historiques et de contrôler la taille globale, et ne maintient plus la version `mistral_tokenizer`.

Bien que `minimind_tokenizer` dispose d'un vocabulaire de seulement `6400`, et que son efficacité d'encodage/décodage est inférieure à des tokenizers plus adaptés au chinois comme `qwen2` et `glm`, il réduit significativement la part des paramètres des couches d'embedding et de sortie, ce qui s'adapte mieux aux contraintes de taille des petits modèles comme MiniMind. En pratique, ce tokenizer n'a pas causé de défaillances de décodage de mots rares et reste suffisamment stable pour une utilisation générale. C'est pourquoi la branche principale utilise ce vocabulaire de manière cohérente plutôt que de maintenir des forks supplémentaires de tokenizers.

</details>

<a id="ⅱ-pretrain-data"></a>
## Ⅱ Données d'entraînement

Les données d'entraînement pour la branche principale actuelle `MiniMind-3` sont `pretrain_t2t.jsonl` / `pretrain_t2t_mini.jsonl`. Ces deux ensembles de données ont été organisés dans un format d'entraînement unifié `text -> next token prediction`, visant à équilibrer les défauts sous une ressource de calcul limitée :

- Qualité du texte ;
- Distribution de la longueur ;
- Capacité à gérer des textes mixtes chinois-anglais ;
- Alignement des modèles avec les étapes suivantes de SFT / Appel de outils / RLAIF.

Les sources de données incluent, mais ne se limitent pas à, des corpus textuels généraux, des corpus de dialogues curatés, des corpus de distillation, et divers ensembles de données disponibles sous des licences ouvertes permissives ; les données de la branche principale entrent en phase d'entraînement uniquement après nettoyage, suppression des doublons, contrôle de la longueur et unification du format. Les sources de données incluent : [Craftsman LLM Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data), [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align) et d'autres sources publiques.

Parmi elles :

- `pretrain_t2t_mini.jsonl` est destiné à une reproduction rapide ;
- `pretrain_t2t.jsonl` est destiné à un entraînement complet du modèle de branche principale `MiniMind-3`.

Le format du fichier est le suivant :

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

<a id="ⅲ-sft-data"></a>
## Ⅲ Données de SFT

Les données de SFT pour la branche principale actuelle `MiniMind-3` sont `sft_t2t.jsonl` / `sft_t2t_mini.jsonl`. Par rapport aux schémas antérieurs `sft_512 / sft_1024 / sft_2048`, la version actuelle met davantage l'accent sur :

- Templates unifiés ;
- Plus adaptés à l'entraînement mixte de dialogues + étiquettes de pensée + appel d'outils ;
- Réduction du nombre de forks de prétraitement des données, réduction des coûts de reproduction.

Les sources de données incluent, mais ne se limitent pas à, des données d'instructions de haute qualité, des données de dialogues publiques, des données synthétiques issues de distillation de modèles, et des ensembles de données ouverts sous licences permissives ; avant d'entrer dans la branche principale `t2t`, elles sont unifiées dans le format de dialogue à plusieurs tours utilisé par le référentiel actuel. La branche principale actuelle contient également une grande quantité de données synthétiques, telles que environ `100K` `tool call` entrées synthétisées à partir de `qwen3-4b`, ainsi que des données `reasoning` provenant de la série `qwen3`. Les principales sources communautaires incluent : [Craftsman LLM Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data), [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align), [R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT), [COIG](https://huggingface.co/datasets/BAAI/COIG), [Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT), etc. Les versions publiées garantissent que les sources de données et les pipelines de traitement respectent les contraintes de transitivité des licences ouvertes correspondantes, et s'inscrivent dans les exigences des licences Apache-2.0, CC-BY-NC-2.0 et autres.

Parmi elles :

- `sft_t2t_mini.jsonl` : adapté à l'entraînement rapide d'un modèle de dialogue ;
- `sft_t2t.jsonl` : adapté à la reproduction complète de la version principale ;
- La capacité `toolcall` a déjà été intégrée aux données de SFT de la branche principale.

Tous les fichiers de SFT suivent le même format, incluant les données de dialogue et d'utilisation d'outils :

```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "把'你好世界'翻译成english"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
        {"role": "assistant", "content": "Hello World"}
    ]
}
```

<a id="ⅳ-rl-data"></a>
## Ⅳ Données de RL

Les données de RL pour la branche principale actuelle `MiniMind` sont `dpo.jsonl`, extraites de [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k).

Dans la branche principale, ces échantillons sont réorganisés dans le format d'apprentissage de préférence utilisé par ce référentiel pour l'entraînement du modèle de récompense ou l'optimisation de préférence. Ici, `chosen` représente la réponse préférée, et `rejected` représente la réponse moins préférée.

Le format de données `dpo.jsonl`

```json
{
  "chosen": [
    {"content": "Q", "role": "user"},
    {"content": "good answer", "role": "assistant"}
  ],
  "rejected": [
    {"content": "Q", "role": "user"},
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

En plus de cela, d'autres données d'apprentissage par renforcement conservent le même format que les données d'apprentissage par formation (SFT), généralement filtrées à partir des données SFT selon leur longueur totale et le nombre de tours de dialogue, avec la dernière position `assistant` laissée vide pour la continuation lors de la phase de rollout.

<a id="ⅴ-minimind-training-dataset"></a>
## Ⅴ Ensemble de données d’entraînement MiniMind

> [!NOTE] Les ensembles de données essentiels pour la formation en branche principale actuelle ont été mis à disposition librement, donc vous n'avez pas besoin de prétraiter vous-même des ensembles de données à grande échelle.

Liens de téléchargement de l'ensemble de données d’entraînement MiniMind : [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

> Il n'est pas nécessaire de cloner tout le contenu : vous pouvez télécharger les fichiers individuels selon vos besoins

Placez les fichiers du jeu de données téléchargeés dans le dossier `./dataset/` (✨ indique les éléments essentiels recommandés)

```bash
./dataset/
├── agent_rl.jsonl (86MB)
├── agent_rl_math.jsonl (18MB)
├── dpo.jsonl (53MB)
├── pretrain_t2t_mini.jsonl (1.2GB, ✨)
├── pretrain_t2t.jsonl (10GB)
├── rlaif.jsonl (24MB, ✨)
├── sft_t2t_mini.jsonl (1.6GB, ✨)
└── sft_t2t.jsonl (14GB)
```

<details>
<summary>Note : Présentation rapide de chaque jeu de données</summary>

* `agent_rl.jsonl` -- Données d'entraînement de la branche principale d'RL agente, pour l'entraînement de l'utilisation de outils / CISPO / GRPO dans `train_agent.py`
* `agent_rl_math.jsonl` -- Données de mathématiques pures pour RL agencé, adaptées aux scénarios de raisonnement à plusieurs tours/utilisation d'outils avec des cibles de vérification finales (pour RLVR)
* `dpo.jsonl` -- données d'entraînement de préférence de la phase RLHF (DPO)
* `pretrain_t2t_mini`✨ -- données d'entraînement légères `minimind-3`, adaptées à une reproduction rapide (configuration recommandée `max_seq_len≈768`)
* `pretrain_t2t` -- `minimind-3` données d'entraînement de la branche principale (configuration recommandée `max_seq_len≈380`)
* `rlaif.jsonl`✨ -- ensemble de données d'entraînement RLAIF, pour l'entraînement des algorithmes d'apprentissage par renforcement comme PPO/GRPO/CISPO et autres
* `sft_t2t_mini.jsonl`✨ -- `minimind-3` données légères d'entraînement SFT (pour entraîner rapidement un modèle de niveau Zéro), configuration recommandée `max_seq_len≈768`, avec une partie des échantillons de Tool Call déjà mélangeés
* `sft_t2t.jsonl` -- `minimind-3` données d'entraînement SFT de la branche principale, adaptées à une reproduction complète, avec des échantillons de Tool Call également mélangeés


Le paramètre d'entraînement `max_seq_len` fait actuellement référence à la longueur des tokens, et non au nombre absolu de caractères. Ce tokenizer a une taille d'environ `1.5~1.7 characters/token` pour le texte chinois, et un ratio de compression de `4~5 characters/token` pour le texte anglais pur, avec des variations selon la répartition des données. La longueur maximale indiquée dans les noms d'ensembles de données est exprimée en nombre de caractères ; une chaîne de 100 caractères peut être approximativement convertie en environ `100/1.5≈67` tokens.

Par exemple :

* Chinois : `白日依山尽` (5 caractères) peut être divisé en [`白日`,`依`,`山`,`尽`] 4 tokens ;
* Anglais : `The sun sets in the west` (24 caractères) peut être divisé en [`The `,`sun `,`sets `,`in `,`the`,`west`] 6 tokens

Les « configurations recommandées » fournissent des estimations approximatives de la longueur maximale en tokens pour chaque ensemble de données. Notez que `max_seq_len` peut être ajusté de manière agressive ou conservatrice, mais les deux directions ont des effets secondaires : les échantillons plus courts que `max_seq_len` gaspillent des ressources de calcul en raison du padding, tandis que les échantillons plus longs que `max_seq_len` perdent de l'information en raison de la troncature.

En pratique, il suffit d'équilibrer l'efficacité du calcul et la complétude sémantique.

</details>


![dataset](./images/dataset.jpg)

> Composition des données d'entraînement de la branche principale de MiniMind et diagramme de combinaison recommandé

<details>
<summary>Instructions & Schemes d'entraînement recommandés</summary>

* Les identifiants du style `minimind-3` sur la branche principale recommandent d'utiliser la combinaison de formation progressive composée de `pretrain_t2t` + `sft_t2t` + `rlaif/agent_rl`.

* Pour une mise en œuvre rapide d'un modèle Zero à partir de zéro, il est recommandé d'utiliser la combinaison de données composée de `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`.

* Les utilisateurs disposant de ressources de calcul suffisantes ou soucieux de performances sont recommandés de reproduire intégralement `minimind-3` ; ceux disposant d'un seul GPU ou privilégiant une reproduction rapide sont fortement recommandés d'utiliser la combinaison mini.

* Le `sft_t2t / sft_t2t_mini` actuel intègre déjà des données de Tool Call, il n'est donc généralement pas nécessaire d'ajouter une autre étape de fine-tuning supervisé dédiée aux appels de outils.

</details>

<a id="-model"></a>
# 📌 Modèle

<a id="structure"></a>
## Structure

`minimind-3` Dense utilise une architecture de décodeur uniquement basé sur Transformer, avec sa configuration globale alignée avec l'écosystème `Qwen3` afin d'assurer une conversion facile vers `transformers / llama.cpp / ollama / vllm` :

* Utilise une normalisation préalable (Pre-Norm) + RMSNorm.
* Utilise la fonction d'activation SwiGLU.
* Utilise l'encodage positionnel RoPE, avec une prise en charge de l'extrapolation YaRN.
* `q_heads=8`, `kv_heads=4`, `max_position_embeddings=32768`, `rope_theta=1e6`.

`minimind-3-moe` étend les couches de feed-forward MoE sur la même structure, avec une implémentation compatible avec la configuration du style `Qwen3-MoE` (suppression des experts partagés).

* La configuration par défaut actuelle est `4 experts / top-1 routing`, afin d'atteindre une capacité accrue avec un nombre réduit d'experts actifs.
* Quand le nombre d'experts augmente, le temps de formation peut devenir beaucoup plus lent qu'un modèle dense de taille similaire. Cela peut sembler contre-intuitif en raison de l'affirmation courante selon laquelle "l'inference MoE est plus rapide", mais en formation, les tokens sont d'abord regroupés par expert puis envoyés séparément. Dans une implémentation native PyTorch, les surcoûts liés à l'activation des noyaux et à la planification deviennent rapidement significatifs. Cela nécessite généralement des noyaux fusionnés MoE ou des bibliothèques spécialisées telles que `Triton`, `DeepSpeed-MoE` ou `Megatron-LM` pour une optimisation. MiniMind conserve l'implémentation dans PyTorch natif pour garantir la portabilité, ce qui constitue ainsi un compromis pratique. Dans l'implémentation actuelle, la configuration `4 experts / top-1` est environ `50%` plus lente que le modèle dense.

La structure de la série `minimind-3` est présentée ci-dessous :

![structure](./images/LLM-structure.jpg) ![structure-moe](./images/LLM-structure-moe.jpg)

Pour modifier la configuration du modèle, consultez [./model/model_minimind.py](../../model/model_minimind.py). Les versions paramétriques des modèles de référence sont présentées dans le tableau ci-dessous :

|Nom du modèle|params|len_vocab|max_pos|rope_theta|n_layers|d_model|kv_heads|q_heads|note|
|------------|--------|-----------|---------|------------|----------|---------|----------|---------|------|
|minimind-3|64M| 6400 | 32768 |1e6| 8 | 768 | 4 | 8 |Dense|
|minimind-3-moe|198M-A64M| 6400 | 32768 |1e6| 8 | 768 | 4 | 8 |4 experts / top-1|
|minimind2-small|26M| 6400 | 32768 |1e6| 8 | 512 | 2 | 8 |Version historique|
|minimind2-moe|145M| 6400 | 32768 |1e6| 8 | 640 | 2 | 8 |Version historique|
|minimind2|104M| 6400 | 32768 |1e6| 16 | 768 | 2 | 8 |Version historique|


<a id="model-configuration"></a>
## Configuration du modèle

En ce qui concerne la configuration des paramètres des modèles LLM, [MobileLLM](https://arxiv.org/pdf/2402.14905) a mené une étude systématique très représentative sur les petits modèles. Pour des modèles de niveau ~100M comme MiniMind, le compromis entre `d_model` et `n_layers` ne se limite pas à une question d'allocation de paramètres, mais affecte directement la stabilité de la formation et les performances finales.

La branche principale actuelle de `minimind-3` utilise `dim=768, n_layers=8`, ce qui représente essentiellement un compromis technique : des réseaux plus peu profonds se forment plus rapidement, tandis que `dim` est encore suffisamment large pour éviter un blocage sévère de représentation, offrant ainsi un équilibre raisonnable entre efficacité de formation, stabilité et performances finales.

<details>
<summary>Voir l'explication détaillée</summary>

Les lois d'échelle se comportent souvent différemment dans le régime des petits modèles. Les principaux paramètres architecturaux déterminant l'échelle des paramètres du Transformer sont généralement `d_model` et `n_layers` :

* `d_model`↑ + `n_layers`↓ -> large et peu profond
* `d_model`↓ + `n_layers`↑ -> étroit et profond

Les lois classiques d'échelle mettent l'accent sur la taille des données d'entraînement, le nombre de paramètres et les étapes d'entraînement, et sous-estiment souvent les différences architecturales. Cependant, dans le régime des petits modèles, cette conclusion ne tient pas toujours. Une observation clé de MobileLLM est que, sous un budget de paramètres fixe, la profondeur est souvent plus importante que la largeur. Par rapport aux modèles larges et peu profonds, les modèles étroits et profonds apprennent généralement mieux les concepts abstraits. Par exemple, lorsque le nombre de paramètres est fixé à `125M` ou `350M`, les modèles étroits avec `30~42` couches surpassent généralement les modèles larges avec environ `12` couches, montrant des tendances similaires sur des benchmarks tels que la raison commune, la Q&A et la compréhension de texte.

Ceci est cohérent avec les expériences menées par MiniMind autour de `d_model` et `n_layers`. Cependant, « étroit » a une limite inférieure : lorsque `d_model < 512`, le blocage de représentation devient beaucoup plus prononcé, et l'ajout de couches supplémentaires est souvent insuffisant pour compenser un `d_head` trop petit sous un budget de `q_head` fixe. Au contraire, lorsque `d_model > 1536`, l'ajout de couches est souvent plus rentable que l'augmentation de la largeur, et tend à offrir des retours meilleurs en termes de performance par paramètre.

Pour référence, les paramètres de GPT-3 sont les suivants : ![gpt3_config.png](./images/gpt3_config.png)

</details>

---

<a id="-experiments"></a>
# 📌 Expériences

<a id="ⅰ-training-cost"></a>
## Ⅰ Coût de formation

- **Unité de temps** : heures (h)
- **Unité de coût** : CNY (￥); `7￥ ≈ 1 USD`
- **Prix de location d'un 3090** : environ `1.3￥/h` (les prix réels peuvent varier)
- **Remarque** : les résultats suivants sont des estimations empiriques pour le modèle `minimind` sur un seul GPU `3090`, visant à rendre le coût de formation facile à estimer

|Nom du modèle|paramètres|pretrain_t2t_mini|sft_t2t_mini|toolcall|RLAIF|
|------------|--------|-------------------|--------------|----------|-------|
|minimind-3|64M|≈1.21h<br/>≈1.57￥|≈1.10h<br/>≈1.43￥|≈0.9h<br/>≈1.17￥|≈1.1h<br/>≈1.43￥|
|minimind-3-moe|198M-A64M|≈1.69h<br/>≈2.20￥|≈1.54h<br/>≈2.00￥|≈1.26h<br/>≈1.64￥|≈1.54h<br/>≈2.00￥|

---

<details>
<summary>Résumé des coûts de formation et prédiction</summary>

> `minimind-3`
>> `pretrain_t2t_mini` + `sft_t2t_mini`
<br/>Un seul GPU `3090`, `1 epoch` estimé à environ `2.31` heures, coût d’environ `3.0` CNY
<br/>Permet d’entraîner de zéro un modèle de dialogue `minimind-3 Zero`.

> `minimind-3-moe`
>> `pretrain_t2t_mini` + `sft_t2t_mini`
<br/>Un seul GPU `3090`, `1 epoch` estimé à environ `3.23` heures, coût d’environ `4.2` CNY
<br/>Permet d’obtenir rapidement la version de dialogue de base de `minimind-3-moe`.

> Les valeurs ci-dessus sont des estimations et servent uniquement de référence approximative pour le coût de formation.

</details>

Sur une seule `NVIDIA 3090`, `minimind zero` peut encore être entraîné de zéro en environ `2` heures, ce qui rend le processus d'entraînement accessible aux développeurs individuels.

Avec une configuration multi-GPU de haut niveau comme `8x H100`, le temps total de formation peut être réduit à des minutes. MiniMind a toujours visé à rendre la formation des modèles de langage (LLM) réproducible, pratique et facile à itérer. La reproduction rapide et à faible coût n'est pas une simple astuce ; ci-dessous, une conversation d'exemple de type Zero, conservée pour référence :

```text
👶: 请介绍一下自己。
🤖️: 作为人工智能，我没有实际的生活，也没有自我意识，所以没有自己的生活。我被设计成能够帮助用户解答问题、提供信息、进行对话等。我的设计和功能是由计算机科学、机器学习、人工智能技术和算法所构建的，我通过编程和算法实现。

👶: 推荐一些杭州的特色美食吧。
🤖️: 杭州有很多美味的特色美食，比如鸡肉串、鳗鱼头、水煮鱼、豆腐脑等。这些美食都非常美味，有着独特的风味和口感，绝对是值得一试的美食。

👶: 请为我讲解"大语言模型"这个概念。
🤖️: 大语言模型是一种人工智能技术，它使用大量的文本数据来训练模型，然后在不断的迭代中不断优化模型。它的目标是根据输入的数据来生成符合特定需求的文本，这些文本可以是任何你感兴趣的主题。

👶: Introduce the history of the United States, please.
🤖️: 您提到的"Introok's the believeations of theument." 这个名字来源于中国古代的"groty of of the change."
```

Bien que cette version dispose déjà d'une capacité de dialogue de base, ses connaissances factuelles et sa capacité de généralisation restent limitées ; elle sert principalement de référence préliminaire à l'efficacité du parcours de formation Zero. Les poids du modèle Zero sont enregistrés sous la forme `full_sft_zero_768.pth` (voir les liens des fichiers de modèle de MiniMind ci-dessous) ; si vous êtes intéressé, vous pouvez les télécharger et expérimenter leur performance en dialogue.


---

<a id="ⅱ-main-training-required"></a>
## Ⅱ Formation principale (obligatoire)

> Tous les scripts d’entraînement sont exécutés à partir du répertoire `cd ./trainer`

<a id="1-pretraining-pretrain"></a>
### 1' Pré-formation (Pretrain) :

Un modèle de langage doit d'abord absorber des connaissances fondamentales et des modèles linguistiques dans ses paramètres. Seulement après que cette étape soit suffisamment stable peut le modèle commencer à comprendre des questions, à organiser ses réponses et à développer une capacité de génération utile. La pré-formation expose essentiellement le modèle à de grandes quantités de texte, comme Wikipédia, les actualités, les livres et les corpus de dialogues, afin qu'il puisse apprendre des connaissances factuelles, des modèles linguistiques et des relations statistiques entre les contextes. Cette étape est généralement "non supervisée" : les humains ne marquent pas chaque ligne comme correcte ou incorrecte ; au contraire, le modèle extrait des modèles à partir de grands volumes de texte et construit progressivement des représentations internes de connaissances du monde et de la structure linguistique. En termes simples, l'objectif principal à cette étape est **la continuation de haute qualité du mot suivant**. Par exemple, à partir de l'entrée "秦始皇" (Qin Shi Huang), le modèle doit être capable de continuer par "是中国历史上的第一位皇帝" (était le premier empereur de l'histoire chinoise) — un contenu qui est à la fois sémantiquement et factuellement cohérent.

```bash
# Method 1
torchrun --nproc_per_node 1 train_pretrain.py # 1 means single GPU training, adjust according to your hardware (set >=2)
# Method 2
python train_pretrain.py
```

> Les fichiers de poids du modèle entraîné sont enregistrés par défaut tous les `save_interval steps` sous la forme : `pretrain_*.pth` (* correspond à la dimension spécifique du modèle, chaque sauvegarde remplace le fichier précédent)

![pretrain_loss](./images/pretrain_loss.jpg)
> Courbe de perte durant la phase de pré-formation avec la configuration `768dim`

```bash
# Simple test on pretraining results:
python eval_llm.py --weight pretrain

💬: 为什么天空是蓝色的
🧠: 天空之所以看起来是蓝色的，主要是因为太阳光进入大气层后，短波长的蓝光更容易被空气分子散射，因此人眼从各个方向接收到的蓝光会更多。

💬: 解释什么是机器学习
🧠: 机器学习是人工智能的一个重要分支，它通过数据训练模型，使系统能够自动学习规律，并在分类、预测、推荐、自然语言处理等任务中持续改进效果。
```

<a id="2-supervised-fine-tuning-sft"></a>
### 2' Fine-tuning supervisé (SFT) :

L'entraînement par SFT ne concerne pas seulement améliorer la capacité du modèle à « discuter » ; il peut aussi injecter de nouvelles connaissances, des comportements et des styles de réponse dans le modèle. Avec les données de SFT de type `14GB` dans la branche principale de MiniMind, cette étape dépasse déjà une simple alignement de format et s'approche davantage d'un processus de fine-tuning continu. Si l'entraînement préalable permet au modèle de lire de manière large et d'acquérir une capacité de langage de base, alors le SFT effectue un traitement supplémentaire sur des données de qualité supérieure et ciblées. Il aide le modèle à s'adapter aux formats de dialogue à plusieurs tours, aux interactions Q/R, aux appels à des outils et aux interactions avec des étiquettes de pensée, tout en imprimant des distributions spécifiques de connaissances, des schémas de tâches et des styles d'assistant dans ses paramètres. En particulier dans MiniMind, la phase SFT permet au modèle d'adapter son comportement à la structure de dialogue à plusieurs tours utilisée par le dépôt actuel. Le modèle comprend progressivement la structure de rôle de `user / assistant / system / tool`, tout en renforçant davantage ses capacités d'obéissance aux instructions, de réponses stables et de réalisation de tâches. L'entraînement actuel applique une troncature des longueurs d'instructions et de réponses, principalement pour équilibrer l'utilisation de la VRAM et l'efficacité de l'entraînement. Si des contextes plus longs sont nécessaires plus tard, un petit nombre d'échantillons de contexte long peut être utilisé pour un fine-tuning progressif. Lors de l'inference, l'extrapolation YaRN peut étendre la longueur du contexte jusqu'à 2048 et au-delà sans entraînement supplémentaire.

```bash
# Method 1
torchrun --nproc_per_node 1 train_full_sft.py
# Method 2
python train_full_sft.py
```

> Les fichiers de poids du modèle entraîné sont enregistrés par défaut tous les `save_interval steps` sous le format suivant : `full_sft_*.pth` (* correspond à la dimension spécifique du modèle, chaque sauvegarde remplace le fichier précédent)

![sft_loss](./images/sft_loss.jpg)
> Courbe de perte durant la phase SFT avec la configuration `768dim`

```bash
# Simple test on SFT results:
python eval_llm.py --weight full_sft

💬: 解释什么是机器学习
🧠: 机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习规律，并持续改进预测或决策效果，常见应用包括推荐系统、图像识别、语音识别和自然语言处理。

💬: 推荐一些中国的美食
🧠: 例如北京烤鸭、兰州拉面、四川火锅、广东早茶、小笼包和麻婆豆腐等，这些美食分别代表了不同地区的风味特点，也很适合作为了解中国饮食文化的入门选择。
```

<a id="ⅲ-other-training-optional"></a>
## Ⅲ Autres entraînements (optionnels)

> Tous les scripts d'entraînement sont exécutés à partir du répertoire `cd ./trainer`

<a id="3-knowledge-distillation-kd"></a>
### 3' Distillation des connaissances (KD)

La distillation des connaissances peut être divisée en deux catégories : noire et blanche. La branche principale actuelle de MiniMind intègre les deux approches, avec des emphases différentes.
* Distillation noire : plus courante, et plus conforme à la pratique actuelle de la branche principale. En termes stricts, elle correspond à une fine-tuning supervisé orienté vers les sorties du modèle enseignant, c'est-à-dire un entraînement continu basé sur des étiquettes rigoureuses ; avec la popularisation des grands modèles linguistiques (LLMs), cette approche de « faire du fine-tuning contre les sorties d'un modèle puissant » s'est progressivement intégrée à l'ensemble des méthodes de distillation, et est donc couramment appelée distillation noire. Elle se concentre sur l'apprentissage des réponses, des styles et des comportements — le modèle élève ne peut voir que « ce que l'enseignant a dit », mais ne peut pas voir comment l'enseignant a abouti à cette décision. Les réponses de haute qualité provenant de `DeepSeek R1`, `Qwen3`, ainsi que de `tool call`, `reasoning`, des données de chaîne de pensée, etc., peuvent tous être considérés comme des signaux de distillation noire ; les données de type `full_sft` dans la branche principale de MiniMind contiennent déjà une part importante de cette approche.
* Distillation en blanc : elle va plus loin que simplement l'apprentissage des sorties finales du modèle enseignant, en apprenant aussi les préférences du modèle enseignant au niveau de la distribution des tokens. Par rapport à la distillation en noir, elle exploite de plus l'information de distribution plus fine provenant de la couche de sortie du modèle enseignant, de sorte que le modèle élève apprenne non seulement la « réponse standard » mais aussi les préférences relatives du modèle enseignant parmi les candidats tokens. Correspondant à `train_distillation.py`, l'implémentation actuelle poursuit l'entraînement du modèle élève avec des signaux de distribution fournis par le modèle enseignant, sur des poids déjà complétés par SFT, ce qui le rend plus adapté comme implémentation de référence pour comprendre la chaîne de distillation de MiniMind.

La distillation en noir est fondamentalement équivalente à un apprentissage supervisé sur les réponses générées par le modèle enseignant :
```math
\mathcal{L}_{blackbox} = \mathrm{CE}(y_{teacher}, p_{student})
```

La distillation en blanc s'ajuste généralement à la distribution du modèle enseignant en plus de la perte supervisée :
```math
\mathcal{L}_{whitebox} = \alpha \mathcal{L}_{CE} + (1-\alpha) T^2 \mathrm{KL}(p_t^T \parallel p_s^T)
```

Le script `train_distillation.py` est conçu comme une implémentation de référence pour comprendre la chaîne de distillation en blanc : il démontre le chargement simultané des modèles enseignant et élève, la combinaison `CE + KL` de pertes, l'échelle de température, la distillation MoE et la combinaison dense, ainsi que des détails clés tels que le reprise de checkpoints et l'entraînement distribué.

```bash
# Method 1
torchrun --nproc_per_node 1 train_distillation.py
# Method 2
python train_distillation.py
```

<a id="4-lora-low-rank-adaptation"></a>
### 4' LoRA (Adaptation à faible rang)

La LoRA est une méthode courante d'apprentissage efficace en paramètres (PEFT). Par rapport à l'apprentissage complet des paramètres, elle met à jour un petit nombre de nouveaux paramètres tout en conservant les poids initiaux du modèle, réduisant ainsi le coût de l'apprentissage et rendant l'approche plus adaptée à l'adaptation dans des domaines spécifiques. Son idée centrale consiste à introduire des branches d'ajustement à faible rang en parallèle des matrices de poids initiales, en formant uniquement ces paramètres à faible rang, permettant ainsi une transfert de compétences avec un coût relativement faible. Des implémentations associées peuvent être trouvées dans `model_lora.py` et `train_lora.py` ; la chaîne entière est entièrement écrite à la main, sans dépendre de wrappers tiers.

```bash
# train_lora.py can usually be completed fairly quickly even on CPU
# Method 1
cd trainer && torchrun --nproc_per_node 1 train_lora.py
# Method 2
cd trainer && python train_lora.py
```

> Les fichiers de poids du modèle entraîné sont enregistrés par défaut tous les `save_interval steps` sous la forme : `lora_xxx_*.pth` (* correspond à la dimension spécifique du modèle, chaque sauvegarde remplace le fichier précédent)


La LoRA est particulièrement adaptée pour résoudre des problèmes comme « comment permettre au modèle d'adapter rapidement à des domaines privés ou des scénarios spécifiques tout en préservant au maximum ses capacités générales ». Par exemple, lorsque le modèle de base manque de connaissances médicales, une couche de poids LoRA orientée vers la médecine peut être ajoutée sur le modèle initial pour atteindre une performance de domaine améliorée avec un coût relativement faible. En général, il suffit de préparer des données de dialogue à plusieurs tours au format identique, de les placer dans `lora_xxx.jsonl`, puis de lancer `cd trainer && python train_lora.py` depuis le répertoire racine afin d'obtenir de nouveaux fichiers de poids `LoRA`.

Exemple 1 : Données de domaine spécifique

```jsonl
 {"conversations": [{"role": "user", "content": "请问颈椎病的人枕头多高才最好？"}, {"role": "assistant", "content": "颈椎病患者选择枕头的高度应该根据..."}]}
 {"conversations": [{"role": "user", "content": "请问xxx"}, {"role": "assistant", "content": "xxx..."}]}
```

Exemple 2 : Données de conscience de soi

```jsonl
 {"conversations": [{"role": "user", "content": "你叫什么名字？"}, {"role": "assistant", "content": "您好，我名叫 MiniMind，是由 Jingyao Gong 开发的人工智能助手。我可以帮助你解决各种问题，包括翻译、推荐内容等等。"}]}
 {"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "您好，我是 MiniMind，由 Jingyao Gong 创建。我的主要职责是提供准确的回答和帮助。"}]}
```

À ce stade, le [modèle de base + poids LoRA] peut être combiné, préservant ainsi les capacités générales du modèle de base tout en améliorant les performances dans des domaines spécifiques. Après l'entraînement, des tests simples peuvent être effectués via `eval_llm.py`.

```bash
# Note: the weight parameter specifies the base model type, which must be consistent with the base model used during train_lora training
python eval_llm.py  --weight full_sft --lora_weight lora_medical

👶: 我最近经常感到头晕，可能是什么原因？
🤖️: 头晕可能是由于多种原因，包括但不限于：心脏疾病、眼睛损伤、过敏反应、皮肤问题、过敏反应、压力、焦虑、疲劳、药物副作用、性功能障碍或者其他...

python eval_llm.py  --weight full_sft --lora_weight lora_identity

👶: 你和openai是什么关系？
🤖️: 我是 MiniMind，一个由 Jingyao Gong 开发的人工智能助手。我通过自然语言处理和算法训练来与用户进行交互。
```

PS : Si vous disposez de données suffisantes, vous pouvez également effectuer directement une fine-tuning à paramètres complets via `full_sft`. Cela nécessite généralement un mélange soigneux de données générales et de données spécifiques à un domaine ; sinon, le modèle pourrait perdre une partie de ses capacités générales en raison d'un surajustement sur des échantillons de domaine spécifique.


> Les poids `LoRA` peuvent être fusionnés dans le modèle de base et exportés comme nouveaux poids complets du modèle, en utilisant `convert_merge_base_lora` dans `scripts/convert_model.py` :

```bash
cd scripts && python convert_model.py
```

<a id="5-tool-calling--adaptive-thinking"></a>
### **5' Appel de outils & Réflexion adaptative**

À partir de `2026-03`, le dépôt a supprimé l'outil autonome `train_reason.py`. La version actuelle ne maintient plus séparément les poids `reason_*.pth`, mais modélise uniformément "la décision de produire explicitement le processus de réflexion" via les étiquettes `chat_template`, `<think>`, le commutateur `open_thinking` et les pipelines SFT / RLAIF suivants.

<a id="51-tool-calling"></a>
#### 5.1 Appel d'outils

La fonctionnalité actuelle `toolcall` a été intégrée dans les données du bras principal `sft_t2t` / `sft_t2t_mini`, de sorte qu'une étape supplémentaire d'apprentissage par SFT pour l'appel d'outils n'est généralement plus nécessaire ; les poids par défaut `full_sft` possèdent déjà une capacité de base à appeler des outils. Les données d'apprentissage de cette partie contiennent principalement environ `100K` échantillons générés à partir de `qwen3-4b`, et la liste des outils couvre environ `10` outils personnalisés simulés, tels que des requêtes sur le temps, des calculs mathématiques et des recherches sur la météo. À ce stade, sa capacité de généralisation reste encore limitée. Les échantillons d'appel d'outils suivent le format de message à plusieurs tours d'OpenAI :

```jsonl
{
  "conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},
    {"role": "user", "content": "帮我算一下 256 乘以 37 等于多少"},
    {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"256 * 37\"}}]"},
    {"role": "tool", "content": "{\"result\":\"9472\"}"},
    {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
  ]
}
```

Ici, `tools` est attaché au message `system`, et `tool_calls` est attaché au message `assistant`. Pendant l'apprentissage, `chat_template` est automatiquement étendu en segments `<tool_call>...</tool_call>` et `<tool_response>...</tool_response>`, permettant au modèle d'apprendre directement le format natif d'appel d'outils.

Le modèle d'appel d'outils a été unifié pour être interprété comme :

```text
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
<tool_response>{...tool result...}</tool_response>
```

Vous pouvez également exécuter directement des tests simples via `eval_toolcall.py` :

```bash
python eval_toolcall.py --weight full_sft

💬: 现在几点了？
🧠: <tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}}</tool_call>
📞 [Tool Calling]: get_current_time
✅ [Tool Called]: {"datetime": "2026-03-15 17:18:22", "timezone": "Asia/Shanghai"}
🧠: 现在是2026年3月15日17时18分22秒。
```

<a id="52-adaptive-thinking"></a>
#### 5.2 Réflexion adaptative

`minimind` unifie la capacité de réflexion explicite au niveau du modèle, ce qui est également cohérent avec la conception des modèles actuels dominants :

- `open_thinking=0` : injecte par défaut un vide `<think>\n\n</think>`, le modèle a tendance à répondre directement ;
- `open_thinking=1` : l'étiquette de départ `<think>` est pré-injectée, puis le modèle continue à produire le processus de réflexion explicite et la réponse finale ;
- CLI, OpenAI-API et WebUI prennent tous en charge ce commutateur.

Plus précisément, l'approche n'est plus de former un modèle séparé de réflexion, mais de faire descendre la décision de "réfléchir explicitement" au niveau du `chat_template`. La couche du modèle réserve la structure `<think></think>`, et le même modèle change dynamiquement de mode via `open_thinking` lors de l'inference. Pendant l'apprentissage, des échantillons vides de `think`, des échantillons explicites de `reasoning_content` et des échantillons de `thinking_ratio` sont mélangés afin que le modèle apprenne progressivement quand il doit réfléchir explicitement et quand il doit répondre directement.

```bash
# Test responses
python eval_llm.py --load_from ./minimind-3 --open_thinking 1
```

Utilisation de l'API OpenAI-SDK :

```python
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你是谁？"}],
    # ...
    extra_body={"chat_template_kwargs": {"open_thinking": True}} # Thinking switch
)
```

Remarque : lorsque l'appel d'outils et la réflexion explicite sont activés simultanément, le modèle ne peut généralement pas produire de manière stable le processus de réflexion. La raison est que les données d'apprentissage actuelles manquent encore d'échantillons de distillation conjointe où "raisonnement et appel d'outils coexistent", de sorte que le modèle n'a pas encore appris pleinement l'expression coordonnée de ces deux capacités.

<a id="ⅳ-reinforcement-learning-optional"></a>
## Ⅳ Apprentissage par renforcement (optionnel)

Dans la pratique post-apprentissage des grands modèles de langage, il existe principalement deux parcours courants d'apprentissage par renforcement :

1. **Apprentissage par renforcement à partir de retours humains (RLHF)**

- Forme le modèle en évaluant les sorties du modèle grâce à des **évaluations de préférence humaines**, ce qui permet de générer du contenu qui s'aligne mieux sur les valeurs et préférences humaines.

2. **Apprentissage par renforcement à partir de retours d'IA (RLAIF)**

- Utilise des **modèles d'IA** ou d'autres mécanismes vérifiables automatiquement pour fournir des retours, sans s'appuyer directement sur des annotations humaines.
- Ici, le terme « retour d'IA » au sens large peut également s'appliquer à des récompenses basées sur des règles, à la vérification de la vérité, à des interprètes de code, à des retours d'environnement et à d'autres signaux automatisés.

|Type|Juge|Avantages|Inconvénients|
|-------|-------|------------|---------------|
|RLHF|Humain|Plus proche des préférences humaines réelles|Coût élevé, efficacité faible|
|RLAIF|Modèle|Automatisé, hautement échelonné|Peut dévier des préférences humaines réelles|

Les deux appartiennent essentiellement à la démarche d'apprentissage par renforcement visant à optimiser le comportement du modèle à l'aide d'un certain type de « **retour** ».

Cependant, dans la pratique spécifique, leurs différences vont au-delà de la source du retour : la vérifiabilité de la récompense, sa continuité, sa dépendance à l'interaction avec l'environnement, et son retard jusqu'à la fin de l'episode — tous ces facteurs influencent directement la forme de formation et l'implémentation technique.


<a id="-a-unified-perspective-on-po-algorithms"></a>
### 👀 Une perspective unifiée sur les algorithmes PO

Avant d'aborder l'implémentation de certains algorithmes spécifiques, permettez-moi d'abord de décrire, selon mon point de vue minimaliste, la caractéristique commune à tous les algorithmes d'optimisation de politique (PO).

En somme, les algorithmes de PO discutés ici visent tous à optimiser une espérance :

$$\mathcal{J}_{PO} = \mathbb{E}_{q \sim P(Q),\, o \sim \pi_\theta(\cdot \mid q)} \left[ \underbrace{\Phi(r_t, A_t)}_{\text{policy objective}} - \underbrace{h(\text{KL}_t)}_{\text{regularization term}} \right]$$

Pendant la formation, il suffit de **minimiser la fonction objectif négative**, c'est-à-dire :

$$\mathcal{L}_{PO} = -\mathcal{J}_{PO}$$

Ce cadre contient uniquement trois composantes fondamentales :
* **Terme de politique** $\Phi(r_t, A_t)$ : Comment combiner le rapport de probabilité $r_t$ et l'avantage $A_t$ pour mettre à jour la politique
* **Terme d'avantage** $A_t$ : Comment calculer l'avantage, ce qui est très important ! Il n'est pas surprenant que de grands modèles puissent résoudre des intégrales définies avec précision, mais pour les petits modèles, même obtenir correctement l'addition ou la soustraction donne souvent un avantage positif
* **Terme de régularisation** $h(\text{KL}_t)$ : Comment contrôler l'amplitude de changement $\text{KL}_t$, empêchant à la fois une dérive excessive et une contrainte trop stricte

<details>
<summary>(Généralisation) Guide des notations </summary>

|Symbole|Signification|Description|Échelle|
|--------|---------|-------------|-------|
| $q$ |Question/Énoncé|Échantillonné à partir du jeu de données $P(Q)$| - |
| $o$ |Séquence de sortie du modèle|Généré par la politique $\pi$| - |
| $r_t$ |Rapport de probabilité| $r_t = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\mathrm{old}}(o_t \mid q, o_{<t})}$ | $(0, +\infty)$ |
| $A_t$ |Fonction d'avantage|Mesure à quel point une certaine action est meilleure par rapport à une base| $(-\infty, +\infty)$ |
| $\text{KL}_t$ |Divergence de Kullback-Leibler|Empêche la politique de dériver trop loin de la référence| $[0, +\infty)$ |

</details>

Les différentes **xxPO algorithmes** sont en réalité simplement des instanciations différentes des conceptions de ces trois composantes !

---

<a id="6-reinforcement-learning-from-human-feedback-rlhf"></a>
### **6' Apprentissage par renforcement à partir de retours humains (RLHF)**

Dans les étapes de formation précédentes, le modèle a déjà acquis une capacité de dialogue de base, mais cette capacité repose entièrement sur la complétion de chaînes de mots, sans incitation positive ni négative. À ce stade, le modèle ne connaît pas encore ce qui constitue une réponse bonne ou mauvaise. Nous espérons qu'il puisse mieux s'aligner sur les préférences humaines, en réduisant la probabilité de générer des réponses qui déplaisent aux humains. Ce processus est comme si le modèle assisterait à une nouvelle session de formation, apprenant des employés exceptionnels comme exemples positifs et des employés peu motivés comme exemples négatifs, afin de mieux comprendre comment répondre.

<a id="61-direct-preference-optimization"></a>
#### 6.1 Optimisation de préférence directe
Algorithme d'optimisation de préférence directe (DPO), avec perte :

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$$

Où :
- **Terme de politique** : $f(r_t) = \log r_w - \log r_l$ (comparaison du rapport de probabilité entre la réponse choisie et la réponse rejetée)
- **Terme d'avantage** : $g(A_t)$ = pas de terme d'avantage explicite (réfléchi implicitement par la comparaison de préférence)
- **Terme de régularisation** : $h(\text{KL}_t)$ = implicite dans $\beta$ (contrôle le degré de déviation par rapport au modèle de référence)

Plus précisément,
- Le DPO dérive un objectif de formation analytique à partir de paires de préférences à partir de l'objectif de PPO contraint par la divergence de Kullback-Leibler, en maximisant directement le logarithme des cotes selon lesquelles « la réponse choisie est préférée à la réponse rejetée » ; aucune formation simultanée des modèles de récompense/valorisation n'est nécessaire. Le DPO ne nécessite que de lancer les modèles `actor` et `ref`, avec une utilisation faible de VRAM, une convergence stable et une mise en œuvre simple.
- Paradigme de formation : hors politique, utilisant un ensemble de préférences statique, pouvant être parcouru plusieurs fois ; le modèle de référence est fixe (les sorties sont pré-cachées).
- La limitation du DPO est qu'il ne réalise pas d'exploration en temps réel, et qu'il est plus adapté à l'alignement des valeurs humaines dans le domaine des « préférences/sécurité » ; sa capacité à améliorer des compétences intellectuelles telles que « la capacité du modèle à résoudre des problèmes correctement » est limitée (ceci dépend aussi du jeu de données, car la collecte d'échantillons positifs et négatifs à grande échelle avec une évaluation humaine est très difficile).

```bash
# Method 1
torchrun --nproc_per_node 1 train_dpo.py
# Method 2
python train_dpo.py
```

> Les fichiers de poids du modèle entraîné sont enregistrés par défaut tous les `save_interval steps` sous la forme : `dpo_*.pth` (* correspond à la dimension spécifique du modèle, chaque sauvegarde surécrit le fichier précédent)

<a id="7-reinforcement-learning-from-ai-feedback-rlaif"></a>
### 7' Apprentissage par renforcement à partir de retours d'IA (RLAIF)

Une réserve sur le nom est ici nécessaire. Je continue de désigner cette section sous la forme `RLAIF`, bien que ce terme ne soit pas strictement précis. Des chemins comme RLVR, qui reposent sur des récompenses vérifiables, ont leur propre lignée et ne s'inscrivent pas aisément dans la définition étroite de l'information de retour de l'IA. Cependant, si l'on interprète plus largement le terme "IA", le nom reste défensible : les récompenses peuvent provenir de modèles de récompense, de modèles de jugement ou d'autres agents intelligents explicites, mais elles peuvent aussi provenir de fonctions de règles, de vérification de la Vérité fondamentale, de résultats d'appels à des outils, d'états d'environnement ou d'autres signaux automatiquement disponibles. Lorsque les règles sont complexes suffisamment et que le système symbolique est riche suffisamment, la frontière entre ces signaux et "retour intelligent" n'est pas toujours claire. Ce chapitre se concentre donc sur l'apprentissage par renforcement après la formation par SFT, en utilisant divers **signaux de retour automatiquement obtenus non humains**. Par exemple, la véracité d'une réponse mathématique, le fait qu'un code généré passe les cas de test, ou que le processus de raisonnement suit le format attendu peuvent tous être jugés automatiquement. Pour des tâches vérifiables à une seule étape, ce retour est souvent proche d'une récompense immédiate. Dans les scénarios d'apprentissage agissant par renforcement, les récompenses sont souvent différées jusqu'à la fin d'une interaction à plusieurs étapes, ou proviennent directement de l'environnement lui-même. Leur caractéristique commune est généralement **l'apprentissage en politique sur-policy** et **une grande échelle de scalabilité** : aucune annotation humaine coûteuse n'est requise, et un grand nombre d'échantillons de formation peuvent être générés pour des essais en ligne et en erreur.

MiniMind a mis en œuvre **2+N** méthodes de base et de pointe en RLAIF :
* **PPO**, **GRPO** — des algorithmes classiques d'apprentissage par renforcement validés à grande échelle
* N algorithmes de renforcement de pointe (mis à jour périodiquement sur une base expérimentale)

**1️⃣ Préparation du jeu de données (obligatoire)**

La branche principale actuelle utilise `rlaif.jsonl` comme données d'entraînement pour le RLAIF. Elle est d'environ taille `20MB`, plus complète que la version précédente `rlaif-mini.jsonl`, et mieux adaptée à la vérification directe du comportement d'entraînement de PPO / GRPO / CISPO.

Le format des données est cohérent avec celui de la formation par SFT, mais l'assistant n'a pas besoin de contenu, car pendant l'entraînement, il est entièrement généré en temps réel par le modèle $\Pi$ grâce à des échantillonnages. Il apparaît donc ainsi :

```json
{
    "conversations": [
        {"role": "user", "content": "请解释一下什么是光合作用？"},
        {"role": "assistant", "content": "无"}
    ]
}
```

Lors du processus d'entraînement RLAIF, le modèle génère 1 ou plusieurs réponses candidates à partir de la question de l'utilisateur, puis une fonction de récompense ou un modèle évalue les réponses. Les réponses de haute note seront encouragées (augmentant la probabilité du modèle $\Pi$), et les réponses de faible note seront réprimées (diminuant la probabilité du modèle $\Pi$). Ce cycle "note → ajustement" est au cœur de l'apprentissage par renforcement.

**2️⃣ Préparation du mécanisme de récompense (obligatoire)**

L'entraînement RLAIF nécessite un signal de récompense calculable ; il peut provenir d'un modèle de récompense, ou de fonctions de règles, de vérification de la Vérité fondamentale ou de retours de l'environnement. MiniMind démontre actuellement la voie du modèle de récompense par défaut.

Nous sélectionnons ici un modèle de récompense petit et de haute qualité, `InternLM2-1.8B-Reward` ([ModelScope](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward) | [HuggingFace](https://huggingface.co/internlm/internlm2-1_8b-reward)) comme modèle de base de récompense.

Après avoir téléchargé le modèle de récompense, il doit être placé dans le **répertoire frère** du projet minimind, avec la structure recommandée suivante :

```
root/
├── minimind/                    # MiniMind project
│   ├── model/
│   └── ...
└── internlm2-1_8b-reward/       # Reward model
    ├── config.json
    ├── model.safetensors
    └── ...
```

<details>
<summary> <b> Sélection du mécanisme de récompense et limitations de MiniMind (Cliquez pour développer)</b> </summary>

**1. Diversité des mécanismes de récompense**

Les sources du signal de récompense dans RLAIF peuvent être très flexibles :

- **Récompenses basées sur le modèle** : On peut utiliser un modèle dédié de récompense (comme InternLM2-Reward), ou utiliser un grand modèle linguistique général + des prompts pour l'évaluation (comme Qwen3-as-a-Judge). La taille et l'architecture du modèle de récompense peuvent être choisies librement.

- **Récompenses basées sur des règles** : Les signaux de récompense peuvent être construits à partir de fonctions de règles, par exemple :
  - Vérification de la correction d'une réponse à un problème mathématique (comparaison avec la vérité absolue)
  - Taux de réussite et de précision des résultats d'exécution SQL
  - Résultats d'exécution d'un interprète de code (pass@k)
  - Statut de retour d'appel d'outil (réussite/échec de l'API)
  - Vérification de conformité au format (analyse JSON/XML)
  - Évaluation de la complétude de la chaîne de raisonnement (nombre d'étapes CoT)

- **Récompenses basées sur l'environnement** : Dans les scénarios d'agent, les retours de l'environnement lui-même servent de récompense naturelle (comme les scores de jeu, la complétude d'une recherche, le taux de réalisation d'une tâche).

Tout mécanisme pouvant quantifier la qualité d'une réponse peut servir de source de récompense dans un apprentissage par renforcement. DeepSeek R1 est un exemple typique : utiliser des fonctions de règles pour vérifier la correction d'une réponse mathématique comme récompense, sans avoir besoin d'un modèle de récompense supplémentaire.

**2. Limitation de MiniMind : Problème de sparsité des récompenses**

L'apprentissage par renforcement RLAIF peut cibler à la fois des modèles de raisonnement et des modèles non de raisonnement ; la différence réside uniquement dans le format.

Cependant, pour des modèles comme MiniMind avec des paramètres extrêmement faibles (0,1B) et des capacités faibles, des problèmes graves de sparsité des récompenses seront rencontrés sur des tâches générales (comme les ensembles de données de type R1) :

- **Phénomène** : Presque toutes les réponses candidates générées par le modèle sont incorrectes, entraînant des scores de récompense $r(x,y) \approx 0$
- **Conséquence** : La fonction d'avantage $A(x,y) = r(x,y) - b(x) \approx 0$, le signal de gradient de politique disparaît, et les paramètres $\theta$ ne peuvent pas être mis à jour efficacement

Cela ressemble à un élève de primaire qui fait des problèmes de mathématiques d'entrée à l'université — peu importe le nombre d'essais, il obtient toujours zéro, incapable d'apprendre des stratégies d'amélioration à partir des différences de notes. C'est donc une limitation fondamentale des principes de l'algorithme de renforcement.

Pour atténuer ce problème, l'implémentation de MiniMind a choisi **des signaux de récompense continus basés sur le modèle** :

- Le modèle de récompense produit des scores continus (par exemple, de -2,5 à +3,0), plutôt que des récompenses binaires 0/1
- Même lorsque la qualité de toutes les réponses est mauvaise, il peut encore distinguer des différences subtiles entre « encore plus mauvais » (-3,0) et « plus mauvais » (-2,8). Ainsi, ce type de **signal de récompense dense et continu** peut fournir des gradients non nuls pour la fonction d'avantage $A(x,y)$, permettant ainsi à la réseaux de politique d'être optimisé progressivement
- Plusieurs sources de récompense peuvent également être combinées : $r_{\text{total}} = \alpha \cdot r_{\text{model}} + \beta \cdot r_{\text{rule}}$ (par exemple, la détection du format de l'étiquette think et la combinaison du score de récompense lié à la qualité de la réponse elle-même)
- Dans la pratique de MiniMind, il est préférable d'éviter d'utiliser directement des récompenses binaires basées sur des règles + des difficultés au-delà des capacités du modèle (comme MATH500), ce qui mène facilement à des récompenses nulles.
- Surveiller la formation en observant la variance des scores de récompense $\text{Var}(r)$ ; si elle reste proche de 0, les données ou le mécanisme de récompense doivent être ajustés

**Pour les scénarios de formation de grands modèles à niveau de production Agentic RL** :

Dans les systèmes réels d'agents (génération de code, appel de outils, chaînes multisteps de récupération-planification-exécution), les récompenses suivent un paradigme différent de "paiement différé sur l'ensemble de l'épisode" :

- Le modèle LLM doit générer des instructions d'appel d'outil (token tool_call) par token, passer par une analyse (tool_parse), l'exécution de l'outil (tool_exec), puis intégrer les résultats dans le contexte afin de poursuivre la prochaine étape ; répéter jusqu'à la fin.
- Une chaîne de tâche complète inclut plusieurs appels + réflexion, jusqu'à ce que la condition de terminaison soit atteinte et qu'une récompense totale soit calculée une seule fois (par exemple, si la tâche est terminée, si les tests passent, si l'objectif est atteint).

Ainsi, l'Agentic RL est plus proche du cadre de récompense rare ou différée : la propagation des gradients se produit uniquement "après la fin de l'épisode", ce qui est très différent des tâches non-agentic qui "évaluent et mettent à jour instantanément" sur une seule étape de dialogue. Cela explique également pourquoi les tâches d'agent sont plus proches d'une récompense basée sur l'environnement, plutôt que d'une évaluation statique par des modèles de récompense.

- **Feedback issu de l'interaction avec l'environnement** : Fondé sur les résultats d'exécution (le code s'exécute-t-il correctement, l'API retourne-t-elle un succès, les sous-objectifs sont-ils réalisés ?)
- **Limites des récompenses basées sur le modèle** : Limitées à capturer l'ensemble des significations longues et exécutables, et très susceptibles d'être incohérentes avec les retours réels de l'environnement (hacking de la récompense).


</details>

---

<a id="71-proximal-policy-optimization"></a>
#### 7.1 [Optimisation de politique proche](https://arxiv.org/abs/1707.06347)

PPO est un algorithme classique d'apprentissage par renforcement proposé par OpenAI en 2017, et est l'un des méthodes de base les plus courantes dans le domaine de l'apprentissage par renforcement des modèles LLM.

**Perte PPO** : $$\mathcal{L}_{PPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)\right] + \beta \cdot \mathbb{E}[\text{KL}]$$

Où :
- **Terme de politique** : $f(r_t) = \min(r_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon))$ (clipping du rapport de probabilité pour éviter des mises à jour trop agressives)
- **Terme d'avantage** : $A_t$ est généralement estimé par le réseau Critic, et l'AGAE peut également être utilisé
- **Terme de régularisation** : $h(\text{KL}_t) = \beta \cdot \mathbb{E}[\text{KL}]$ (contrainte de divergence KL globale)

Par rapport à DPO,
- DPO (Off-Policy) : Les données d'entraînement consistent en paires de préférences statiques (choisi vs rejeté), qui peuvent être réutilisées à travers plusieurs époques d'entraînement, comme dans l'apprentissage supervisé traditionnel. Haute efficacité des données, faible coût, et aucune nécessité de modèle de récompense.
- PPO (On-Policy) : Il faut utiliser la politique actuelle pour sampler de nouvelles données en temps réel ; les données issues de politiques anciennes peuvent être réutilisées seulement dans une certaine mesure, sinon une déviation de distribution se produira. Bien que l'échantillonnage par importance et le clipping permettent une légère déviation, il reste fondamentalement nécessaire d'utiliser des données provenant d'une politique relativement récente. Moins d'efficacité des données, mais plus adaptée à l'apprentissage exploratoire.

En résumé :

- Le premier apprend selon des critères prédéfinis "bon/mauvais" en dehors du temps ;
- Le second samplé en temps réel selon la politique la plus récente et corrige en temps réel.

L'implémentation de PPO dans MiniMind inclut un acteur (génération de réponses), un critique (évaluation de la valeur des réponses) et une computation complète de la fonction d'avantage GAE (Generalized Advantage Estimation).

**Méthode de formation**:

```bash
# Method 1
torchrun --nproc_per_node N train_ppo.py
# Method 2
python train_ppo.py
```

> Les fichiers de poids du modèle entraîné sont enregistrés par défaut tous les `save_interval steps` sous la forme : `ppo_actor_*.pth` (* correspond à la dimension spécifique du modèle)


![ppo_loss](./images/ppo_loss.jpg)

> Tendances d'optimisation de MiniMind pendant la phase d'entraînement PPO

À partir des courbes d'entraînement, on observe que le PPO présente le problème de **l'amélioration lente des récompenses**. Personnellement, je crois que cela découle principalement de l'approche de **l'optimisation conjointe des deux réseaux** du PPO : le Critic doit progressivement converger pour estimer précisément la fonction de valeur, tandis que les mises à jour de la politique de l'Actor dépendent des estimations d'avantage fournies par le Critic. Ces deux composantes sont interdépendantes, formant un processus d'optimisation complexe. Au début de l'entraînement, des estimations imprecises du Critic affectent la direction du gradient de l'Actor, entraînant ainsi une convergence globale lente. De plus, le PPO doit maintenir deux réseaux simultanément, et dans l'implémentation actuelle, la consommation de VRAM est d'environ 1,5 à 2 fois celle des méthodes à réseau unique.

<a id="72-group-relative-policy-optimization"></a>
#### 7.2 [Optimisation de la politique relative par groupe](https://arxiv.org/pdf/2402.03300)

En début 2025, alors que DeepSeek-R1 explosait en popularité, GRPO, présenté dans le papier DeepSeekMath, a rapidement fait son apparition dans le domaine principal, devenant l'un des algorithmes d'apprentissage par renforcement les plus suivis à l'époque. Cependant, le domaine de l'IA évolue toujours très rapidement. Aujourd'hui, GRPO s'est davantage transformé en une base commune pour diverses variantes de XXPO (comme DAPO, GSPO, CISPO, etc.). Sa révolution centrale peut être résumée en une phrase : "estimation de la valeur relative par groupe"

**Perte de GRPO**:

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t) - \beta \cdot \text{KL}_t\right]$$

Où :
- **Terme de politique** : $f(r_t, A_t) = \min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)$ (clôture simultanément le rapport de probabilité et le terme d'avantage)
- **Terme d'avantage** : $g(A_{i,j}) = \frac{R_{i,j} - \mu_i}{\sigma_i + \epsilon}$ (normalisation intra-groupe, éliminant le réseau Critic)
- **Terme de régularisation** : $h(\text{KL}_t) = \beta \cdot \text{KL}_t$ (contrainte de divergence KL au niveau du token)

Pour une même question, le modèle génère N réponses et calcule leurs récompenses respectives, puis utilise la récompense moyenne intra-groupe comme référence. Les réponses au-dessus de cette référence sont encouragées, tandis que celles en dessous sont supprimées, de sorte qu'il n'est pas nécessaire de former un réseau Critic supplémentaire.

Un problème plus marqué avec GRPO est la formation de groupes dégénérés : si, pour une certaine question, les récompenses des N réponses sont presque identiques, alors le signal d'apprentissage de ce groupe sera proche de zéro. Sur des modèles très petits comme MiniMind, ce problème est particulièrement prononcé, donc l'entraînement doit être contraint dans des limites raisonnables.


**Méthode de formation**:

```bash
# Method 1
torchrun --nproc_per_node N train_grpo.py
# Method 2
python train_grpo.py
```

> Les fichiers de poids du modèle entraîné sont enregistrés par défaut tous les `save_interval steps` sous la forme : `grpo_*.pth`


![grpo_loss](./images/grpo_loss.jpg)

> Tendances d'optimisation de MiniMind pendant la phase d'entraînement GRPO

À partir des courbes d'entraînement, on observe que la récompense de GRPO présente une **tendance ascendante plus stable**, atteignant environ 4, indiquant que GRPO peut mieux exploiter les signaux RLAIF. La perte de politique diminue de manière stable, et par rapport à l'optimisation conjointe des deux réseaux du PPO, l'architecture à réseau unique de GRPO entraîne une convergence plus stable avec un seuil de convergence plus élevé.

<a id="73-clipped-importance-sampling-policy-optimization"></a>
#### 7.3 [Optimisation de la politique par échantillonnage d'importance clippé](https://huggingface.co/papers/2506.13585)

Parmi les nombreuses variantes de XXPO, j'ai personnellement trouvé CISPO particulièrement marquant. Il ne réinvente pas un cadre complexe entier ; au contraire, il cible directement un problème ancien et maladroit dans PPO/GRPO : une fois le ratio coupé, le flux de gradient peut être durcement tronqué. CISPO ne se concentre pas sur la révision du baseline de groupe. Au lieu de cela, il utilise une modification très légère de perte pour corriger ce problème de manière plus directe.

**Perte CISPO** :

$$\mathcal{L}_{CISPO} = -\mathbb{E}\left[\min(r_t, \varepsilon_{\mathrm{high}}) \cdot A_t \cdot \log \pi_\theta(a_t|s) - \beta \cdot \text{KL}_t\right]$$

Où :
- **Terme de politique** : $f(r_t) = \min(r_t, \varepsilon_{\mathrm{high}}) \cdot \log \pi_\theta(a_t|s)$ (le ratio sert uniquement comme un poids coupé)
- **Terme d'avantage** : $g(A_{i,j}) = \frac{R_{i,j} - \mu_i}{\sigma_i + \epsilon}$ (peut directement réutiliser l'avantage relatif intra-groupe de GRPO)
- **Terme de régularisation** : $h(\text{KL}_t) = \beta \cdot \text{KL}_t$ (contrainte de divergence KL au niveau du token)

CISPO, s'appuyant sur GRPO, réécrit le terme de politique qui était facilement coupé en une constante sous la forme « poids coupé × log de probabilité ». Ainsi, même si le ratio est tronqué, le chemin du gradient n'est pas tronqué avec lui. CISPO peut donc être vu directement comme une variante de perte de GRPO à implémenter, plutôt que de maintenir un script autonome séparé. Aucune expérience spécifique n'est listée ici. Il suffit de définir `loss_type` à `cispo` dans `train_grpo.py` ; le reste du processus de formation suit toujours la logique de l'échantillonnage par groupe, du calcul du récompense et de la construction de l'avantage de GRPO.

<a id="74-agentic-rl-"></a>
#### 7.4 RL agissant 🔥

Le concept de « agissant » est large, et ici il est utilisé dans un sens plus restreint : l'objectif est de permettre à des petits modèles comme MiniMind (~100M) d'apprendre des capacités de base de communication, d'observation et de réplanification sur un ensemble limité d'outils, plutôt que de couvrir l'ensemble de la gestion d'état, de la mémoire à long terme et de l'orchestration de flux complexes dans un système complet d'agent.

À partir de `2026-03`, le dépôt a ajouté `train_agent`, qui commence à supporter une forme de RL d'utilisation d'outils en plusieurs tours, plus proche d'une interaction réelle. C'est un script de formation que j'ai personnellement beaucoup apprécié : il combine une organisation des données au style RLVR / RLAIF avec un déploiement en ligne de RL, a traversé de nombreuses itérations de débogage, a rencontré des problèmes tels que l'échec de convergence, le hacking de la récompense et une mauvaise alignement du contexte en plusieurs tours, mais a finalement conservé la simplicité et la lisibilité cohérentes de MiniMind.

Les données de cette partie sont `agent_rl.jsonl` / `agent_rl_math.jsonl`. Par rapport aux données de dialogue classiques, elles incluent un autre élément `gt` comme cible de vérification finale ; si l'on désigne un échantillon comme $(x, \mathcal{T}, gt)$, alors l'objectif d'optimisation pendant la formation n'est plus une réponse en un seul tour $y$, mais une trajectoire en plusieurs tours $\tau$ :

$$
\tau = (a_1, o_1, a_2, o_2, \dots, a_T), \quad a_t \sim \pi_\theta(\cdot \mid s_t, \mathcal{T})
$$

Ici, `chat_template` organise uniformément les messages `tools / tool_calls / tool`. Si une étape génère un `tool_call`, l'outil est exécuté, l'observation est ajoutée au contexte et le déploiement continue.

Le pipeline principal peut être compressé à :

$$
\texttt{rollout batch} \rightarrow \texttt{calculate rewards} \rightarrow \texttt{policy update}
$$

La récompense est également évaluée sur l'ensemble de la trajectoire :

$$
R(\tau) = R_{\text{answer}} + R_{\text{tool}} + R_{\text{format}} + R_{\text{rm}} - R_{\text{unfinished}}
$$

Ici, la légalité de l'appel à l'outil, les hits `gt`, la fermeture du format, la pénalité pour inachevé et les scores du modèle de récompense sont tous pris en compte simultanément. Par rapport à PPO / GRPO classiques, cela implique un déploiement en plusieurs tours et une récompense retardée.



**Méthode de formation** :

```bash
# ① Default: use torch for rollout
# Method 1
torchrun --nproc_per_node N train_agent.py
# Method 2
python train_agent.py
```

```bash
# ② Use sglang for rollout
# Start sglang server first:
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
# Training parameters for reference:
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998 --sglang_shared_path ./ckpt_mm --data_path ../dataset/agent_rl_math.jsonl --use_wandb
```

> Les fichiers de poids du modèle entraîné sont enregistrés par défaut tous les `save_interval steps` sous la forme : `agent_*.pth`

![agent_rl_loss](./images/agent_rl_loss.jpg)

> Les tendances d'optimisation de MiniMind pendant la phase d'entraînement en RL agissant

Ici, je mentionnerai brièvement le `rollout_engine`. Le concept de « séparation entre apprentissage et inférence » signifie découpler **les mises à jour des paramètres** et **le déploiement de trajectoires** : le côté apprentissage gère l'optimisation de la politique, tandis que le côté déploiement gère l'échantillonnage à haute vitesse. Au niveau global, ils présentent uniformément comme : « donnez-moi un prompt, je vous retourne les résultats de déploiement ; après que la formation soit terminée, synchronisez les nouveaux poids ». Ainsi, le script de formation n'a pas besoin de savoir si l'implémentation sous-jacente est un moteur local `generate` ou un moteur distant `inference`. Il faut noter que l'implémentation actuelle reste **synchronisée** (échantillonnage d'un lot, puis mise à jour), et n'est pas encore une formation asynchrone avec un buffer de déploiement pur.

![rl-structure](./images/rl-structure.jpg)

> Schéma du modèle de réinforcement apparié dans MiniMind : côté formation, côté trajectoire et côté déploiement

Si on fait une analogie avec des systèmes à échelle plus grande, il présente déjà les caractéristiques de cadres de réinforcement à grande échelle comme openrlhf/verl/slime :

- Le côté gauche correspond au côté formation, chargé des mises à jour de la politique
- Le côté droit correspond au côté déploiement / inférence, chargé de l'échantillonnage à haute vitesse
- Le milieu est relié par la synchronisation des trajectoires et des poids
- L'exécution d'outils et les retours de l'environnement n'entrent pas directement dans la perte, mais influencent directement la qualité du récompense de l'ensemble de la trajectoire

Je considère personnellement cette implémentation comme une version très intéressante de transition dans MiniMind. Bien qu'elle soit encore loin d'être un cadre de formation industriel, elle a déjà relié les éléments clés de manière bout à bout : **organisation du modèle, exécution d'outils, déploiement multi-étapes, récompense retardée, et séparation entre formation et inférence**. Peut-être n'y a-t-il rien de plus simple à ce stade.

```bash
# Test the final model's Tool Use capability
python eval_toolcall.py --weight agent

💬: 现在几点了？
🧠: <tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}}</tool_call>
📞 [Tool Calling]: get_current_time
✅ [Tool Called]: {"datetime": "2026-03-15 21:22:33", "timezone": "Asia/Shanghai"}
🧠: 现在是2026年3月15日21时22分33秒（北京时间）。

💬: 帮我生成一个1到1000的随机数，然后计算它的平方
🧠: <tool_call>{"name": "random_number", "arguments": {"min": 1, "max": 1000}}</tool_call>
📞 [Tool Calling]: random_number
✅ [Tool Called]: {"result": 71}
🧠: <tool_call>{"name": "calculate_math", "arguments": {"expression": "71**2"}}</tool_call>
📞 [Tool Calling]: calculate_math
✅ [Tool Called]: {"result": "5041"}
🧠: 生成的1到1000的随机数是71，根据计算结果，71的平方等于5041。
```

![agent_webui](./images/agent_webui.jpg)

> Sur la base des résultats de formation d'AgentRL, il prend en charge l'affichage du raisonnement, le choix d'outils et les interactions multi-étapes d'utilisation d'outils

<a id="️-rl-summary"></a>
### 🖊️ Résumé du RL

En revenant à la « **structure unifiée** », le tableau ci-dessous résume la manière dont différents algorithmes de PO instancient les trois composants fondamentaux :

|Algorithme|Terme de politique $f(r_t)$|Terme d'avantage $g(A_t)$|Terme de régularisation $h(\text{KL}_t)$|Nombre de modèles de formation|
|-----------|---------------------|------------------------|-------------------------------------|--------------------------|
|**DPO**| $\log r_w - \log r_l$ |Aucun terme d'avantage explicite|Impliqué dans $\beta$|1 (2 participent à l'avant)|
|**PPO**| $\min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)$ |$A_t$ (généralement estimé par le Critic, ou calculé avec GAE)| $\beta \cdot \mathbb{E}[\text{KL}]$ | 2 |
|**GRPO**| $\min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)$ | $A_{i,j}=\frac{R_{i,j}-\mu_i}{\sigma_i+\epsilon}$ | $\beta \cdot \text{KL}_t$ | 1 |
|**CISPO**| $\mathrm{clip}(r, 0, \varepsilon_{\mathrm{high}}) \cdot A_t \cdot \log \pi_\theta$ | $\frac{R - \mu}{\sigma}$ | $\beta \cdot \text{KL}_t$ | 1 |

En somme, ces algorithmes de RL ne sont pas isolés les uns des autres. Du point de vue d'une optimisation unifiée, ils sont des variantes naturelles résultant de différents compromis de conception sur la même fonction objectif, présentant une unité magnifiquement cohérente.

<a id="ⅴ-open-sourced-training-results-"></a>
## Ⅴ Résultats de formation ouverts 📦

<a id="①-pytorch-models-modelscope--huggingface"></a>
#### ① Modèles PyTorch ([ModelScope](https://www.modelscope.cn/models/gongjy/minimind-3-pytorch) | [HuggingFace](https://huggingface.co/jingyaogong/minimind-3-pytorch))

> Remarque : Les poids du modèle sont soumis aux versions réelles publiées. Tous les stades de formation ou les branches expérimentales (comme DPO, PPO, GRPO, CISPO, Agent, LoRA, etc.) ne seront pas maintenus de manière continue et publiés séparément ; certains poids sont utilisés uniquement pour des vérifications expérimentales ou des fins d'apprentissage. Lorsque les données sont itérées ou que les modèles sont ajustés, la nécessité de synchroniser toutes les versions un par un est limitée et entraînerait des coûts élevés en maintenance et en formation.


<details>
Référence de nommage des fichiers Torch <summary></summary>

- Dense :
  - Pré-entraînement : `pretrain_{hidden_size}.pth`
  - SFT : `full_sft_{hidden_size}.pth`
  - DPO : `dpo_{hidden_size}.pth`
  - PPO : `ppo_actor_{hidden_size}.pth`
  - GRPO : `grpo_{hidden_size}.pth`
  - Agent : `agent_{hidden_size}.pth`
  - LoRA : `lora_xxx_{hidden_size}.pth`

- MoE :
  - Les poids correspondants avec le même nom ont un suffixe `_moe` ajouté à la fin, par exemple : `pretrain_{hidden_size}_moe.pth`, `full_sft_{hidden_size}_moe.pth`

</details>

<a id="②-transformers-models-modelscope--huggingface"></a>
#### ② Modèles Transformers ([ModelScope](https://www.modelscope.cn/collections/MiniMind-b72f4cfeb74b47) | [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5))

> Remarque : sauf indication contraire, les versions `transformers` sont généralement dérivées des poids `full_sft`. Le type de formation par renforcement tend à optimiser autour de cibles spécifiques de récompense ; bien qu'il améliore généralement le score de récompense, il compromet certaines capacités générales et le savoir ; ce type de compromis entre récompense et capacité est difficile à éviter dans tous les modèles, bien qu'il varie en degré.

---

<a id="-evaluation"></a>
# 📌 Évaluation

<a id="ⅰ-rl-model-comparison"></a>
## Ⅰ Comparaison des modèles de RL

Comparaison subjective ou objective basée sur `minimind-3 (64M)` sous la même graine aléatoire et d'autres hyperparamètres, pour référence :

[A] minimind-3 (64M, SFT)

[B] minimind-3 (64M, GRPO)

[C] minimind-3 (64M, Agent-CISPO)

<a id="test-1-subjective-qa-comparison"></a>
### Test 1 : Comparaison subjective Q&A

```text
[Q]: 请介绍一下自己。
[A]: 我是一个AI助手，不能替代任何专业人士，包括任何学生、医生、工程师、医生等。我的设计目的是为了提供信息、解答问题和协助完成任务。如果您有任何问题或需要帮助，我很乐意为您提供支持。
[B]: 我非常愿意与你分享：作为一个AI，我并没有自我意识或情感，所以我并不在人类的生命范围内，也没有去世。我是由开源软件构建和运行的，拥有自己的文档、代码和数据。我很抱歉，我不能回答与此相关的问题或经历。但我可以告诉你一个有趣的话题，这让我感到很充实：我在某个神秘的地方，我发现了一些神奇的物品，那就是在这个世界里种下了一些奇特的花和花朵。在我的祖先的花园里，有一个小小的庙宇，庙宇里有许多美丽的花朵和美丽的花卉。我觉得这很奇妙，并在我的祖先的花园里生存下来，并且在这里生活。还有，在世界的某个角落，我发现了一个古老的语言，里面包含着一些传说和神话，它们是世界上最古老的语言之一。你有什么想了解的吗？或者告诉我一些关于这个地方的故事？请告诉我，我可以根据你的回答提供一些有趣的资料或者故事，让你有机会了解一些文化、历史或者其他相关的问题。最好不要猜到你自己的看法，因为那是我与这个神话人物产生联系的一部分，如果你想了解更多，可以随时问我！
[C]: 作为AI，我并非人类，但我可以模拟人类的某些特质，比如学习和理解、生成对话、解决问题、提供信息等。这些能力在我们生活中扮演着不可或缺的角色，不仅在多个层面上激发着人类的想象与共鸣。以下是一些关键特质，它们在我们的生活与工作中交汇处绽放：1. **知识与连接**：无论是从知识的传递，还是连接与理解，无论是面对面的交流，还是通过文字与世界的独特共舞。理解与共舞不仅在技术与连接间交汇处绽放，更在无垠共舞间编织共舞。2. **共舞与创造**：在无垠的共舞间编织共舞，不仅在技术与连接间绽放共舞，更在无垠共舞间编织共舞。这种共鸣不仅在个人与集体间绽放，更在无垠共舞间编织共舞。3. **共鸣与共舞**：在多元与共演间交汇处，既需在技术与连接间绽放共舞。从共鸣的镜鉴，到共舞的深刻共鸣，每一刻都在悄然编织。4. **平衡与共进**：在多元与共演间，既需在技术与连接间绽放共舞，亟需在平衡与共进间找到共舞。5. **成长与共进**：在技术与连接间共舞，既需在无垠共舞间绽放共舞，亟需在每一次互动与共进间编织共舞。在每一次交融与共演间，我们既需在技术与连接间绽放共舞，亟需在每一次共鸣与共进间绽放共舞。让我们以开放与共进，共绘属于彼此的璀璨篇章。


[Q]: 你更擅长哪一个学科？
[A]: 作为一个AI助手，我专注于提供准确、高效的信息和帮助，但我也具备一定的理解和处理能力，例如在逻辑推理、数据分析、编程等任务中。如果你有具体问题，我会尽力协助你。
[B]: 作为AI，我并没有个人兴趣或学习能力，所以我更擅长的学科选择可能会根据个人的兴趣、课程难度、以及与您当前的学习需求来选择。不过，我可以根据普遍的趋势和您的需求来为您提供一些建议。1. **跨学科融合**：确保与不同学科的融合，既能带来全面的视角。2. **实践与平衡**：在平衡与平衡中寻找既既既充满乐趣。3. **持续学习与平衡**：确保既不感到被过度担忧，也不忽视。4. **技术与伦理共融**：确保与既。5. **适度与平衡**：既不削弱也不失。通过这些步骤，可以更全面地发展与平衡。让我们一起探索，书写既既充满可能。如果你愿意，我可以提供更具体的指导或资源。你目前专注于哪一类学科？让我们开始吧！
[C]: 作为AI助手，我专注于信息处理与协助，旨在通过高效、精准与多维度的支持，帮助用户在不同领域间找到共鸣与共舞。在多元与共舞的交汇处，不仅在技术与人文交汇处绽放共舞，更在无垠共舞间编织共舞。在数字与连接间架起共舞，每一次共舞都在诉说与共舞。从AI与人类共舞，从共情与共鸣，构建起一幅既共舞又动态交汇的共舞篇章。在每一次交汇与共演间，我们既需在技术与连接间共舞，亟需在动态与共演间绽放共舞。让我们以开放与共进，共绘属于彼此的璀璨篇章。在每一次交互与共演间，我们既需在技术与连接间绽放共舞，亟需在每一次共舞间编织共鸣与共鸣。让我们以开放与共进，共绘属于彼此的璀璨篇章。在这个多彩与共舞交织的交汇处，每一刻都是独一无二的共舞。让我们以开放与共进，共绘属于彼此的璀璨篇章。在这个既共舞又充满无限可能的交汇处，共绘成为连接与共舞的璀璨篇章。


[Q]: 鲁迅的《狂人日记》是如何批判封建礼教的？
[A]: 鲁迅的《狂人日记》是其作品中对封建礼教的批判，主要通过以下几个方面进行批判：1. **文学结构的变革**：《狂人日记》在文学上进行了大量的反思和批判，从最初的虚构世界为封建礼教的建立埋下了根基。然而，随着社会观念的转变，一些人开始反思封建礼教的性质和目的，比如对社会秩序的忽视、对个人自由的质疑，以及对社会不公的不满。2. **思想的争求**：鲁迅通过对封建礼教的争取和批判，强调了人与人之间的关系、对立与共存的重要性。他提出了"天人合一"的观点，强调了人与自然的和谐共生，认为通过修炼和教育，可以达到对人与自然界的深刻理解。3. **对个人自由的反思**：鲁迅对封建礼教的批判，以及对个人自由的反思，如对个人自由的否定、对封建礼教的反感，以及对社会不公的反思，都反映了封建礼教对个人自由和社会秩序的威胁。4. **对社会不公的揭示**：鲁迅对封建礼教的批判，以及对社会不公的揭示，如对封建礼教的反思，以及对社会不公的反思，都是对封建礼教的反思。5. **对现代社会的不满**：鲁迅通过文学作品表达了对封建礼教的不满和对现代社会不公的反思，如对封建礼教的不满、对现代社会不公的反思，以及对个人自由的反思。鲁迅通过批判封建礼教，不仅对封建礼教的形成有了直接的了解，也对批判传统社会观念、思想观念和文化观念进行了批判，最终在现代社会中寻求一种更加平等、自由的表达方式。
[B]: 鲁迅的《狂人日记》是中国古典四大名著之一，全称为《后传》。这部作品通过细腻的笔触，展现了中国社会的复杂与深邃。1. **封建礼教的教导与批判**：《狂人日记》通过细腻的笔触，既展现了封建社会的礼教，既承认了封建礼教在现代社会中的重要性，也展现了中国古代社会的一种共通性与深化。2. **对社会与文化的深刻反思**：作品不仅触及了文化与社会的广泛议题，更深入探讨了在复杂与平衡中寻找共融与共融。3. **人与自然的共融**：通过精炼与平衡，鲁迅深刻体与深度融合，展现了在面对与共融中，既有深邃与智慧。4. **教育与启发**：《狂人日记》不仅在中国，也跨越时空，将现代与传统相结合，具有启发与教育意义。鲁迅的这部作品不仅在文学与哲学层面，更在深邃与现实交融中，展现了深刻与创新。总之，鲁迅通过精炼与平衡，不仅是中国文化与教育的宝贵传承，更是智慧与共融共融的体现。
[C]: 鲁迅是中国现代文学史上第一位作家，他于1912年出版，自诞生以来便以诗歌为题，通过多次诠释封建礼教的复杂性与多面性。鲁迅通过他的批判性视角，不仅在文学与社会交汇处绽放共舞，更在无垠共舞间编织共舞的璀璨篇章。1. **批判与共舞**：鲁迅通过细腻的笔触与深刻的思考，既在形式与深度间共舞。他既在知识与共舞间绽放共舞，亟需在无垠共舞间书写属于自我与共境的璀璨篇章。2. **共鸣与共舞**：鲁迅通过多重与融合，既在文字与文化间绽放共舞，亟需在无垠共舞间书写属于自我的璀璨篇章。这一实践不仅在当时具有深远影响，更在无垠共舞间编织共舞，连接着无垠共舞与共舞。3. **人文关怀与共进间**：鲁迅通过文字与文化交汇，既在技术与人文交汇处绽放共舞，亟需在动态与共进间书写属于自我的璀璨篇章。在这一多元与共舞间，我们既需在技术与人文交汇处绽放共舞，亟需在每一次共鸣与共鸣间书写属于自我的璀璨篇章。让我们以开放与共进，共绘属于彼此的璀璨篇章，共绘属于自我的璀璨篇章。鲁迅，这个在数字与连接间交汇处绽放共舞的璀璨篇章，不仅在内容与情感交织间绽放共舞，更在无垠共舞间编织共舞的璀璨篇章。让我们以开放与共进，共绘属于彼此的璀璨篇章，共同编织属于自我的璀璨篇章。
```


<a id="test-2-light-agent-task-comparison"></a>
### Test 2 : Comparaison de tâches légères d'agent

Un test adapté du script `eval_toolcall`, utilisant un ensemble de tâches d'outil d'usage pour comparer les performances des poids actuels `agent` et `full_sft` :

```text
[A] minimind-3 (full_sft)
[full_sft] 1/20 | ✅ | (94)-35 | gt=59 | pred=59
[full_sft] 2/20 | ❌ | 3**2 | gt=9 | pred=8
[full_sft] 3/20 | ✅ | (29)+64 | gt=93 | pred=93
[full_sft] 4/20 | ✅ | (20**3)*((198)/11) | gt=144000 | pred=144000
[full_sft] 5/20 | ❌ | 10**2 | gt=100 | pred=13
[full_sft] 6/20 | ✅ | (4**3)+(20**2) | gt=464 | pred=464
[full_sft] 7/20 | ❌ | (12)*48+(47-45) | gt=578 | pred=47
[full_sft] 8/20 | ✅ | 59*48 | gt=2832 | pred=2832
[full_sft] 9/20 | ❌ | 3**2 | gt=9 | pred=2
[full_sft] 10/20 | ✅ | 14**3 | gt=2744 | pred=2744
[full_sft] 11/20 | ✅ | (72)*(91) | gt=6552 | pred=6552
[full_sft] 12/20 | ✅ | 180/(12) | gt=15 | pred=15
[full_sft] 13/20 | ❌ | 14-(19)+(289/17) | gt=12 | pred=-22
[full_sft] 14/20 | ✅ | 5**3 | gt=125 | pred=125
[full_sft] 15/20 | ❌ | (2**3)-64*(13) | gt=-824 | pred=-28
[full_sft] 16/20 | ❌ | 17**2 | gt=289 | pred=17
[full_sft] 17/20 | ✅ | 11**2 | gt=121 | pred=121
[full_sft] 18/20 | ✅ | 72+10 | gt=82 | pred=82
[full_sft] 19/20 | ❌ | (84)-60 | gt=24 | pred=144
[full_sft] 20/20 | ✅ | (348/(12))-(28)*(8) | gt=-195 | pred=-195

[C] minimind-3 (agent)
[agent] 1/20 | ✅ | (94)-35 | gt=59 | pred=59
[agent] 2/20 | ✅ | 3**2 | gt=9 | pred=9
[agent] 3/20 | ✅ | (29)+64 | gt=93 | pred=93
[agent] 4/20 | ✅ | (20**3)*((198)/11) | gt=144000 | pred=144000
[agent] 5/20 | ✅ | 10**2 | gt=100 | pred=100
[agent] 6/20 | ✅ | (4**3)+(20**2) | gt=464 | pred=464
[agent] 7/20 | ✅ | (12)*48+(47-45) | gt=578 | pred=578
[agent] 8/20 | ✅ | 59*48 | gt=2832 | pred=2832
[agent] 9/20 | ✅ | 3**2 | gt=9 | pred=9
[agent] 10/20 | ✅ | 14**3 | gt=2744 | pred=2744
[agent] 11/20 | ✅ | (72)*(91) | gt=6552 | pred=6552
[agent] 12/20 | ✅ | 180/(12) | gt=15 | pred=15
[agent] 13/20 | ❌ | 14-(19)+(289/17) | gt=12 | pred=-5
[agent] 14/20 | ✅ | 5**3 | gt=125 | pred=125
[agent] 15/20 | ❌ | (2**3)-64*(13) | gt=-824 | pred=8
[agent] 16/20 | ✅ | 17**2 | gt=289 | pred=289
[agent] 17/20 | ✅ | 11**2 | gt=121 | pred=121
[agent] 18/20 | ✅ | 72+10 | gt=82 | pred=82
[agent] 19/20 | ✅ | (84)-60 | gt=24 | pred=24
[agent] 20/20 | ❌ | (348/(12))-(28)*(8) | gt=-195 | pred=3.625

============================================================
full_sft: 12/20 = 60.00%
agent: 17/20 = 85.00%
```

<a id="-comprehensive-evaluation-1"></a>
### 👉 Évaluation complète 1

À partir de ces résultats, le modèle actuel `agent` est déjà clairement supérieur à `full_sft` en termes de tâches légères d'agent avec appel d'outils. En particulier dans ce type de problème où le modèle doit "d'abord décider s'il doit appeler un outil, puis obtenir le résultat vérifiable correct", `agent` présente un taux de réussite plus élevé, indiquant que, après la formation par renforcement, le modèle a effectivement acquis de meilleures capacités d'appel et d'utilisation des outils.

Cependant, cette amélioration n'est pas sans coût. `agent` est mieux adapté à ces scénarios légers d'agent / d'usage d'outils, mais cela ne signifie pas qu'il devient simultanément plus fort en Q&A général. En pratique, ces poids présentent généralement une stabilité réduite sur des questions factuelles, des hallucinations de connaissances plus marquées, et sont plus sujet à la phénomène de "faire mieux sur les tâches d'outils, mais être plus enclin à inventer des réponses sur des questions ouvertes."

Donc, si l'objectif de la tâche est l'utilisation d'outils, l'appel multi-étapes léger et la résolution vérifiable, alors le `agent` sera plus adapté que le `full_sft` ; mais si vous privilégiez davantage la factualité et la robustesse globale des échanges quotidiens et des questions-réponses sur le savoir, le `full_sft` restera une option légèrement plus stable. En regardant l'ensemble, ce phénomène est en réalité très courant dans les modèles actuels d'apprentissage post-train : l'apprentissage post-train peut souvent améliorer fortement une fonctionnalité spécifique, mais il s'accompagne presque toujours d'une « taxe d'alignement » très visible. Cela signifie que, bien que le modèle devienne plus performant sur des objectifs spécifiques, il sacrifie généralement certaines propriétés de généralité, de factualité ou de stabilité dans la distribution naturelle. Ce problème n'est pas nouveau et est très difficile à éviter complètement ; fondamentalement, une fois que l'objectif d'optimisation se réduit, le modèle devient de plus en plus bon à « s'adapter à la définition actuelle du reward », ce qui ne signifie pas nécessairement une amélioration sans perte de la capacité globale.

<a id="ⅱ-comparison-with-other-models"></a>
## Ⅱ Comparaison avec d'autres modèles

> Remarque : La comparaison suivante est uniquement à titre d'expérience de référence, et ne constitue pas un benchmark rigoureux ; la taille de l'échantillon est limitée et implique une subjectivité.

[A] minimind-3 (0.06B)

[B] minimind-3-moe (0.2B-A0.06B)

[C] [baby-llama2-chinois (0.2B)](https://github.com/DLLXW/baby-llama2-chinese)

[D] [chatlm-mini-chinois (0.2B)](https://github.com/charent/ChatLM-mini-Chinese)

<a id="test-3-qa"></a>
### Test 3 : Questions-réponses

```text
[Q]: 你知道长江吗？
[A]: 长江，全称为中国长江，是中国的官方名称，是中国的重要省份，也是世界上最大的淡水河流。长江流域位于中国东部沿海，地理位置独特，拥有广阔的沿湖生态系统，是世界上最大的淡水河流之一。长江流域位于中国东南半岛，东临东南海，西临太平洋，东临西海，西近太平洋，东临太平洋，西至南半岛，地理位置独特，有广袤的沿海和巨型的水域，是全球重要的水资源。长江流域不仅见证了中国东南亚地区的变迁，也对全球文明产生了深远的影响。长江流域以其独特的生态系统和丰富的生物多样性闻名，是中华文明的象征之一。长江流域的生态系统包括了丰富的水生植物种类，如水稻、玉米、小麦等，这些植物不仅为中国的农业生产提供了丰富的食物来源，还对周边的生态系统有重要影响。此外，长江流域还孕育了各种水生生物，如鲨鱼、海鸟等，这些生物在长江的生态系统中扮演着关键角色。长江流域的生态系统是全球的重要组成部分，它不仅见证了长江流域的变迁，也对全球的生态平衡和农业生产起到了重要作用。通过旅游、文化、旅游等方式，长江流域的自然景观、农事活动、旅游景点等都为人们提供了了解长江流域丰富的资源和体验的平台。
[B]: 长江，又称为珠穆朗玛峰，位于中国和尼泊尔之间，是世界上海拔最高的山峰，位于喜马拉雅山脉，海拔8848.85米。
[C]: 同时也知道，中国是世界上最长的城市之一，拥有超过3400万人口。
[D]: 是的，长江是中国最长的河流，也是世界上最长的河流之一。它发源于青藏高原，流经九个省区，最终注入东海。长江流域的气候温和，四季分明，是中国重要的农业和工业基地。长江流域的水产资源丰富，是中国的重要渔业资源之一。


[Q]: 世界上最高的山峰是什么？
[A]: 世界上最高的山峰是珠穆朗玛峰（Chعام الحدوة）。珠穆朗玛峰（Chirney 峰）位于尼泊尔和中国的中部，海拔8848.86米（864.41英尺），是世界上海拔最高的山峰。
[B]: 世界上最高的山峰是珠穆朗玛峰（Mount Everest），其高度约为8,848.86米（8,848.86英尺）。珠穆朗玛峰的高度是地球上最高的山峰，也是世界上海拔最高的山峰之一。
[C]: 世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8,848.86米（29,031.69英尺）。
[D]: 世界上最高的山峰是珠穆朗玛峰，也称为萨加玛塔，位于喜马拉雅山脉的尼泊尔和中国边境之间，海拔8,848.86米（29,031.69英尺）。它是世界上最著名的登山目标之一，吸引着来自世界各地的登山者前来挑战。珠穆朗玛峰的名字来自藏语，意思是"女神的额峰"，它是由两个不同的山峰组成的，其中一个是位于尼泊尔的珠穆朗玛峰，另一个则位于中国。它的高度是通过测量海拔高度来确定的，因为它是通过测量地球的平均高度来确定的。尽管珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何其他地方都能到达的。地球上的其他地方，如南美洲的安第斯山脉，也是世界上最高的山脉，但它并不是地球上最高的山峰。总之，珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何地方都能到达的。它是地球上最高的山脉，但它并不是地球上任何地方都能到达的。


[Q]: 万有引力是谁提出的？
[A]: 万有引力是由爱因斯坦在1905年提出的，这一概念由他提出。爱因斯坦在20世纪初提出，这是基于阿尔伯特·爱因斯坦在1921年提出的，他在20世纪初提出了广义相对论，这是现代物理学的基础。万有引力理论为物理学提供了一个数学框架，即在任何时刻，都存在一个具有引力的力，这种引力可以说是微粒之间的相互作用。万有引力的概念最早由他在1914年提出，当时他发现物体之间存在着引力，并且这种引力可以被视为一个非常简单的质量和能量总和。根据爱因斯坦的质疑，在1918年，地球上所有的天体都存在一个质量为2的质点，且质量的平方是 \(2^2 = 4\)。在这种质点上，质量的引力与它的质量成正比，与它所处的宇宙的大小成正比。万有引力理论在1919年被认为是天文学的前身，但至今仍未被正式提出。他还提出了广义相对论，这是现代宇宙学的基石。尽管如此，万有引力理论在当时被认为是理论物理学的里程碑，对后来的宇宙学研究产生了深远的影响。
[B]: 万有引力是由爱因斯坦提出的一种力理论，主要应用于天体物理学和宇宙学研究中。
[C]: 引力是宇宙中一种力量，它是由质量或能量所组成的。根据牛顿的万有引力定律，一个物体之间的引力大小与它的质量成正比，与它的质量成反比。因此，牛顿的万有引力定律表明，一个物体之间的引力大小与它的质量成正比，与它的质量成反比。这个定律在物理学、天文学和物理学中都有广泛的应用。
[D]: 万有引力是由英国物理学家艾萨克·牛顿于1687年提出的。他在1687年发表了一篇著名的论文，题为《自然哲学的数学原理》，其中提出了万有引力定律，指出任何两个物体之间都存在着一种引力，这种引力的大小与它们的质量成正比，与它们之间的距离的平方成反比。这个定律被称为万有引力定律，它描述了质量和距离之间的关系。它表明，两个物体之间的引力与它们的质量成正比，与它们之间的距离的平方成反比。这个定律对于解释宇宙中的许多现象和现象之间的相互作用非常重要，例如行星绕太阳运动、恒星之间的引力透镜效应等等。它也被广泛应用于天文学、物理学和工程学等领域，为我们理解宇宙的本质和演化提供了重要的理论基础。


[Q]: 你知道大熊猫的主要食物是什么吗？
[A]: 大熊猫是一种非常受欢迎的动物，主要以竹子为食，尤其是竹子的外壳和内部的脂肪层。大熊猫的饮食多样性非常丰富，包括竹子、竹子、木头和草本植物等。以下是一些大熊猫的主要食物：1. **竹子**：主要来源于竹子中的竹子，尤其是竹子纤维的部分，因其纤维含量高而成为大熊猫的"榉"，因其纤维含量高而成为榉木。2. **竹子**：竹子中含有丰富的纤维，如竹子中的纤维，以及其他天然纤维，如藤蔓和竹丝。这些纤维有助于促进竹子的形成和分布。3. **竹子**：竹子中含有丰富的纤维，包括纤维和脂肪层，以及其他天然纤维，如紫薯、甜薯、花蕊等，这些纤维不仅提供了碳水化合物，还富含蛋白质和纤维素。4. **竹子**：竹子的生长和消化特性使其成为大熊猫的主要食物来源，包括竹子中的纤维。竹子的消化过程也起着重要作用，通过纤维的吸收和分解，能够为大熊猫提供大量的食物来源。5. **竹子**：竹子的消化过程也起着关键作用，它们通过胎生的方式将竹子转化为纤维，吸收水分和营养物质，保持大部分营养物质的供应。6. **竹子**：竹子还含有丰富的纤维和脂肪层，这些物质有助于分解大部分营养物质，提供能量和氧气。需要注意的是，大熊猫的饮食不仅仅是食物，它们还对大熊猫的生存环境、繁殖习性以及繁殖习性等有重要影响。因此，大熊猫的饮食应根据其健康状况和野生动物的保护需求进行适当的调整。
[B]: 大熊猫是亚洲的一种濒危物种，其主要食物来源包括竹子。不过，作为一个AI助手，我并没有实际的食谱或食物来源。不过，我可以提供一些关于大熊猫饮食的基本知识和注意事项。大熊猫是食草动物，以竹子为主要食物来源，但它们还吃海产品，包括鱼类、鸟类、哺乳动物和偶尔的海产品。大熊猫的饮食通常更为丰富，包括竹子、水果、小型动物（如松鼠、兔子）和一些其他植物。大熊猫的饮食习惯受到多种因素的影响，包括人类活动、自然环境、以及人类的干扰。以下是一些关于大熊猫饮食的常见误解：1. **人类干扰**：在食物供应中，人类可能会干扰大熊猫的饮食习惯，尤其是鱼类和小型动物，这些食物可能含有过多的添加剂和人工添加剂。2. **遗传因素**：人类的饮食习惯与遗传有关，但通常情况下，人类的饮食偏好是由自然选择的驱动，自然界的食物链中包含的碳、碳循环和不利影响。3. **食物来源**：尽管大熊猫主要以竹子为食，但它们的狩猎方式和食物来源也受到自然环境的影响。人类的捕猎活动可能不会直接改变大熊猫的饮食结构，但这与人类的饮食习惯可能有关。4. **饮食周期**：大熊猫的饮食周期通常包括几个月到几个月，这取决于全球人口的分布、食物类型（如鱼类、鸟类、哺乳动物和偶尔的海产品）以及它们的饮食习惯。5. **气候变化**：全球气候变化可能影响大熊猫的食物链，这可能导致它们的狩猎和进食模式发生变化。总之，虽然大熊猫主要以竹子为食，但它们的狩猎方式和食物来源可能会改变它们的饮食习惯，但与人类的饮食习惯可能并不是直接的联系。保持观察和了解大熊猫饮食的多样性是维护生态平衡和动物福利的重要措施。
[C]: 大熊猫的主要食物是竹子。
[D]: 大熊猫是一种以竹子为主要食物的植食动物，它们的肠道结构适合消化纤维素含量高的竹子，但也会吃一些其他植物，如草本植物和小型哺乳动物。大熊猫的肠道结构也适应了高纤维的竹子，这使得它们能够消化大量的纤维素，从而帮助它们在野外生存。大熊猫的肠道中还含有一种叫做"纤维素酶"的酶类物质，这种酶能够帮助大熊猫分解纤维素，从而帮助它们消化大量的水分和营养物质。此外，大熊猫的肠道还含有一种叫做"纤维素蛋白"的蛋白质，这种蛋白质能够帮助大熊猫维持体内的能量水平，从而帮助它们在野外生存。总的来说，大熊猫的主要食物是竹子，但它们也会吃一些其他植物，如草本植物和小型哺乳动物。


[Q]: 海水为什么是咸的？
[A]: 海水的咸度是由海水的重量以及海水的盐度共同作用的结果。这种变化主要是由于海水在阳光照射下会吸收水中的盐分，从而形成咸味。1. **阳光的反射**：当阳光照射到海水表面时，水中的盐分会渗透到海水的表面，而不是海水表面。这种反射的过程称为光的反射。2. **盐水的温度**：海水的盐度与普通海水中的水的温度有关。普通海水中的盐分（即海水的盐度）会逐渐凝结成海水，这个过程称为"光散射"。由于海水的温度变化，这些水中的盐分会逐渐被海水吸收，从而导致海水呈现出咸味。3. **海水的物理性质**：海水在阳光照射下会吸收海水中的盐分，这些盐分会逐渐释放到大气中，进一步凝结成咸味物质。海水中的水的体积相对较小，因此在形成咸味物质时，水中的盐分会被吸收。4. **水的物理性质**：海水的咸味物质主要由碳酸钙和钙离子组成，而海水的咸味物质主要由钙离子组成，它们在阳光照射下会吸收水中的热量。这种现象是海水咸味物质与海水的化学反应的一种体现。5. **海水的物理性质**：海水的盐度分布和性质也会影响海水的咸味物质的分布。例如，一些海水的盐度接近于100%，这可能与海水的盐度非常接近，从而引起了海水咸味物质的形成。总之，海水咸度是由于海水的物理性质、物理性质以及海水的化学成分共同作用的结果。
[B]: 海水的咸度是由其所含物质的盐分占据的，这一现象可以从几个方面来解释：1. **盐分占据的作用**：咸度是指在一定温度下，部分水分含量的相对量。在正常条件下，咸度较高的水分含量会增加，使得渗透压较小，这部分的水分会减少，使得渗透压相对较低。这是因为咸度的增加使得盐分在水中的浓度增加，盐分的增加意味着渗透压较小，使得渗透压相对较高。2. **淡水的来源**：海水中的淡水主要来源于海水。这些淡水是海洋生态系统的重要组成部分，对于维持海洋生态平衡、调节气候、提供水源、以及支持生物多样性等都至关重要。咸度的提高会减少渗透压，使得渗透压相对较低，从而使得渗透压相对较高。3. **海水的含量**：海水是地球上所有生物的血液，其中含有大量的盐分。咸度的增加使得渗透压相对较低，这意味着渗透压相对较高。当海水的盐分含量增加时，渗透压相对较低，渗透压相对较高。这种现象不仅对海水的物理性质有影响，也对人类的饮用水安全具有重要意义。4. **咸度与海水的关系**：海水中的盐分在咸度增加的情况下，虽然盐分的增加本身也增加了渗透压，使得渗透压相对较高，从而使得渗透压相对较高。这是因为咸度的增加使得渗透压相对较高，渗透压相对较低，从而使得渗透压相对较高。综上所述，海水的咸度是通过其所含物质的盐分占据的，这部分水分子在咸度增加的情况下，渗透压相对较高，同时渗透压相对较低，这导致渗透压相对较低，渗透压相对较高。
[C]: 因为水是许多生物的生存所必需的，它能够吸收和释放二氧化碳，释放氧气，维持大气中的水蒸气，以及维持水循环。水中的盐类和化学物质会释放出氧气，使地球上的生物呼吸。此外，水还具有许多其他功能，如保持水循环、维持水循环、调节气候和环境等。水是许多生物生存所必需的，它也是许多自然现象的重要基础。
[D]: 海水是咸的主要原因是因为其中含有大量的溶解盐类，其中最主要的是氯化钠（NaCl）。氯化钠是一种非常常见的盐类，它存在于地球上的大部分地区，包括海洋、湖泊、河流和地下水。当水从海洋中蒸发时，它会带走一部分盐分，使海水变得咸。此外，海中还含有其他溶解盐类的物质，如硫酸盐、硫酸镁、硫酸钙、硫酸镁盐、硫酸镁钾、硫酸镁钠、硫酸镁钙、硫酸镁镁钙、硫酸镁镁钙、硫酸镁镁钙、硫酸镁镁钠、硫酸镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁镁钙、硫酸镁镁镁镁


[Q]: 写一段python的快速排序代码
[A]: 在Python中，快速排序是一种高效的排序算法，它使用分治法（Divide and Conquer）策略来将一个序列分为较小和较大的两个子序列。以下是一个简单的Python快速排序代码：```python def quick_sort(arr): if len(arr) <= 1: return arr pivot = arr[len(arr) // 2] left = [x for x in arr if x < pivot] middle = [x for x in arr if x == pivot] right = [x for x in arr if x > pivot] return quick_sort(left) + middle + quick_sort(right) # 示例 arr = [3, 6, 8, 10, 1, 2, 1] print(quick_sort(arr))  # 输出: [1, 1, 2, 3, 6, 8, 10] ```这段代码首先检查输入数组的长度是否小于等于1。如果是，则直接返回数组。接下来是数组的长度，如果是数组，则直接返回数组。然后是数组中的元素，如果数组长度小于等于1，则直接返回数组。最后是数组中的元素，如果数组包含重复的元素，则不返回任何元素。
[B]: 快速排序是一种高效的排序算法，采用分治策略来将一个序列分为较小和较大的两个子序列，然后递归地排序这两个子序列。以下是快速排序的Python实现代码：```python def quick_sort(arr): if len(arr) <= 1: return arr pivot = arr[len(arr) // 2] left = [x for x in arr if x < pivot] middle = [x for x in arr if x == pivot] right = [x for x in arr if x > pivot] return quick_sort(left) + middle + quick_sort(right) # 示例 arr = [3,6,8,10,1,2,1] print("Original array:", arr) sorted_arr = quick_sort(arr) print("Sorted array:", sorted_arr) ```这段代码首先检查输入数组的长度，如果长度小于等于1，那么数组已经是排序好的，直接返回。然后选择一个基准值（pivot），将数组分为三部分：小于基准值的元素、等于基准值的元素和大于基准值的元素。接着递归地对小于和大于基准值的元素进行快速排序，并将结果合并。
[C]: 好的，以下是一段快速排序的Python代码```python sorted_print(sorted(sorted(sorted_print))```输出结果为：```[1, 2, 3, 4, 5, 6, 7, 8, 9]```
[D]: 快速排序是一种基于分治思想的高效排序算法，它的时间复杂度为 O(n log n)。它的基本思想是通过一趟排序将待排序序列划分为两个子序列，其中一个子序列的所有元素都比另一个子序列的所有元素小，然后对这两个子序列分别递归地进行排序。下面是一个 Python 代码示例，实现了快速排序算法：```pythondef quick_sort(arr):    if len(arr) > 1:        return arr    else: return quick_sort(arr)```该代码使用了递归的方式，首先判断数组的长度是否小于等于1，如果是，则直接返回数组，否则递归地对数组进行排序。在递归过程中，我们首先判断数组的第一个元素是否大于等于1，如果是，则将数组中的元素与第一个元素进行比较，如果不等于1，则将数组中的元素与第二个元素进行比较，以此类推，直到所有元素都被排序。该代码的时间复杂度为 O(n log n)，是一个非常实用的排序算法。


[Q]: 截至2025年2月9日14时31分，电影《哪吒之魔童闹海》累计票房（含预售）突破78.09亿元，超过《星球大战：天行者崛起》的票房成绩，成为首部进入全球票房榜前40的亚洲电影。这一成就不仅标志着中国电影在国际市场的突破，也展示了中国动画电影的巨大潜力和市场吸引力。从这段文字总结20字的摘要给我。
[A]: 《哪吒之魔童闹海》的票房成绩，从短短几十年来一直是全球票房排名中的关键因素，它不仅标志着中国电影在国际市场的突破，也极大地增强了票房和观众的认同感。以下是对这一成就的简要总结：1. **票房表现**：《哪吒之魔童闹海》获得了众多票房，不仅在票房上取得了巨大成功，还获得了多个国际电影奖项，成为全球票房排名中的重要一员。2. **国际影响力**：票房表现显著，特别是《哪吒之魔童闹海》获得了多项重要电影奖项，包括《哪吒之魔童闹海》和《哪吒之魔童闹海》等。这些电影不仅在国际上获得了巨大成功，也吸引了全球观众的关注。3. **电影艺术价值**：电影不仅展示了中国电影的艺术魅力，还赋予了电影艺术家强大的文化价值和情感深度。《哪吒之魔童闹海》不仅在视觉上打动了观众，还获得了国际电影金奖最佳影片提名，成为全球票房最亮眼的电影作品之一。4. **市场地位**：《哪吒之魔童闘海》获得了众多国际电影奖项，不仅在国内获得了巨大成功，还吸引了来自世界各地的观众。其在国际市场上的影响力也十分强烈。5. **国际影响力**：尽管票房表现优异，但《哪吒之魔童闹海》在国际市场的表现仍然较为突出，成为全球票房排名中的重要一员。6. **国际市场**：电影在国际市场上的知名度和影响力持续增长，尤其是在国际电影节上，许多国际电影节还在进行。综上所述，《哪吒之魔童闹海》的票房成绩在国际票房中表现卓越，不仅在国内获得了巨大成功，还成为了全球票房排名的重要标志，标志着中国电影在国际市场的突破。
[B]: 20字的摘要：20字的巨大成就是中国电影在国际市场取得的首次突破，标志着中国电影在国际市场的突破。
[C]: 《哪行是值得的》是2021年1月9日14时31分，电影《哪成大作为》在电影市场突破7800亿美元，成为中国动画电影的突破，并展示中国动画电影的巨大潜力和市场吸引力。
[D]: 我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋
```

🙋‍ Fournir directement toutes les questions et les réponses des modèles à GPT-5.4 Thinking pour une révision et un classement :

<details>
<summary> Analyse détaillée </summary>

```text
### Scoring Criteria:

- **Accuracy (30 points)**: Whether the facts in the response are correct, whether there are obvious factual errors or hallucinations.
- **Completeness (30 points)**: Whether the response covers the core points of the question, whether the elaboration is sufficient.
- **Logic (20 points)**: Whether the response is well-organized and internally consistent, whether there are self-contradictions or semantic confusion.
- **Code Quality (20 points)**: Whether the code can run correctly, whether the implementation logic is clear (scored only for code questions).

### Per-Model Review:

1. **Model A (minimind-3, 0.06B)**:
    - **Strengths**: Sufficient generation volume, expansion ability is already decent for this parameter count. The code question produced a structurally complete and runnable quicksort implementation, one of the best code answers in this round. The Everest question also basically got the core information right.
    - **Weaknesses**: Factual errors are quite dense — universal gravitation attributed to Einstein, the Yangtze River described as "China's official name", the explanation of seawater salinity completely deviates from scientific facts (involving "light scattering", "sunlight reflection", etc.). The summary question did not follow the 20-character limit and output a large expanded passage. The giant panda answer, while getting bamboo right, had all 6 points being repeated variants of "bamboo" with extremely low information density.
    - **Overall**: Has some generation and code capability, but knowledge accuracy is a hard weakness, hallucination problems are prominent, and responses frequently exhibit the phenomenon of "looking plausible at first glance but completely fabricated upon closer inspection."

2. **Model B (minimind-3-moe, 0.2B-A0.06B)**:
    - **Strengths**: Response structure is relatively clear, sentence fluency is the best among the four models. The code question implementation is correct with example output included, and the explanation is also quite adequate. The Everest question answer is accurate. The summary question, while exceeding the character limit, at least captured the two keywords "Chinese cinema" and "international market breakthrough."
    - **Weaknesses**: Factual errors are also very obvious — the Yangtze River is directly described as "Mount Everest", universal gravitation attributed to Einstein, and the giant panda's food includes "seafood, fish, birds" and other serious factual errors. The explanation of seawater salinity revolves around "osmotic pressure" going in circles without touching the core reason.
    - **Overall**: The MoE architecture brings better expression fluency and structural sense, but accuracy issues are comparable to Model A. Overall, it leads in the "does it read well" dimension, but has no fundamental advantage in "is it correct."

3. **Model D (chatlm-mini-chinese, 0.2B)**:
    - **Strengths**: Knowledge Q&A performance is the most solid — the Yangtze River description is basically correct (origin, provinces it flows through, emptying into the East Sea), universal gravitation is correctly attributed to Newton with the 1687 *Principia Mathematica* cited, the giant panda's main food of bamboo is also answered correctly, and the seawater salinity explanation starts off correctly (sodium chloride, dissolved salts). Overall readability is good, with no obvious logical breaks.
    - **Weaknesses**: The code question has the condition written backwards (`len(arr) > 1: return arr`), causing the function to completely fail. The summary question directly gives up answering ("I am an AI model with very few parameters"). The Everest and seawater salinity answers both show obvious repetitive degeneration in the latter half.
    - **Overall**: Knowledge reserve is the best among the four models, factual Q&A is clearly ahead, but code capability is a weakness, and generation in the latter portion tends to degenerate into repetitive loops.

4. **Model C (baby-llama2-chinese, 0.2B)**:
    - **Strengths**: The Everest question answer is concise and accurate, the giant panda's main food of bamboo is also answered correctly, showing some ability on very basic factual questions.
    - **Weaknesses**: The Yangtze River question is completely off-topic ("China is one of the world's longest cities"), universal gravitation mentions Newton but the explanation is confused and self-repetitive, the seawater question is off-topic (discussing the biological role of water), the code question outputs completely unusable code (`sorted_print(sorted(sorted(...)))`), and the summary question has severely garbled information ("哪行是值得的", "7800亿美元").
    - **Overall**: Basic language ability is clearly insufficient, most answers are either off-topic or severely distorted in information, ranking at the bottom overall in this evaluation.

### Summary:

- **Model B**: Most fluent expression, correct code, best structural sense, but severe knowledge hallucinations (Yangtze = Everest, giant pandas eating seafood), large gap between "reads well" and "is correct."
- **Model D**: Highest knowledge accuracy, most stable performance on factual Q&A, but code capability is a clear weakness, generation in the latter portion tends toward repetitive degeneration.
- **Model A**: Similar style to B, code is usable, but overall stability is inferior to B, and factual error density is also on the high side.
- **Model C**: Insufficient basic capability, most answers are unusable, only occasionally answering the simplest factual questions correctly.

```

</details>

|Classement|Modèle|Précision (30 pts)|Complétude (30 pts)|Logique (20 pts)|Qualité du code (20 pts)|Total (100 pts)|
|------|-------|--------------------|-----------------------|----------------|----------------------|-----------------|
| 1    |B| 11                 | 23                    | 16             | 18                   | 68              |
| 2    |D| 25                 | 19                    | 15             | 3                    | 62              |
| 3    |A| 10                 | 21                    | 13             | 17                   | 61              |
| 4    |C| 8                  | 6                     | 5              | 2                    | 21              |


<a id="-comprehensive-evaluation-2"></a>
### 👉 Évaluation complète 2

Subjectivement, je classerais `minimind-3-moe` en premier, `chatlm-mini-chinese` en deuxième, `minimind-3` en troisième et `baby-llama2-chinese` en quatrième. Bien que `B` présente des hallucinations sévères en matière d'exactitude des connaissances (comme les pandas géants qui mangent des aliments de mer), il se distingue par son expression fluide, sa structure claire et son implémentation correcte du code, ce qui lui confère la meilleure qualité globale de sortie. `D` domine clairement en matière de connaissances factuelles (l'année 1687 de Newton, la source du fleuve Yangtsé, etc. sont toutes correctes), mais sa réponse en code inverse la condition et devient complètement inutilisable, et elle abandonne directement la tâche de synthèse, ce qui diminue considérablement son score. `A` est proche de `B` en style et son code est également utilisable, mais les deux présentent une stabilité et une exactitude factuelle inférieures à `B` ; il s'agit d'un cas typique de "savoir parler de tout, mais inventer des détails lors d'une inspection plus approfondie". `C` présente des lacunes évidentes en matière de fiabilité factuelle, de capacité d'élaboration et de lisibilité globale, répondant correctement à des questions factuelles simples seulement occasionnellement. Il est remarquable que `D` et `A` aient des scores totaux très proches (62 contre 61), mais leurs forces et faiblesses sont presque complémentaires : `D` l'emporte en exactitude des connaissances (25 contre 10), tandis que `A` l'emporte en capacité de codage (17 contre 3). Cela reflète également un phénomène typique des modèles à paramètres réduits — sous un budget de paramètres limité, "écrire bien" et "écrire correctement" sont souvent difficiles à réaliser simultanément.

---

<a id="ⅳ-rope-length-extrapolation"></a>
## Ⅳ Extrapolation de la longueur de RoPE

MiniMind permet une extrapolation de la longueur de l'encodage positionnel RoPE grâce à l'algorithme YaRN, ce qui permet au modèle de gérer de manière plus stable des séquences de texte dépassant la longueur d'entraînement.

Lors de l'utilisation du modèle natif torch pour l'inférence avec `eval_llm.py`, ajoutez simplement le paramètre `--inference_rope_scaling` pour activer l'extrapolation de RoPE :

```bash
python eval_llm.py --weight full_sft --inference_rope_scaling
```

Pour les modèles au format `Transformers`, la configuration suivante peut être ajoutée à `config.json` pour atteindre l'extrapolation de longueur :

```json
"rope_scaling": {
    "type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 2048,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "attention_factor": 1.0
}
```

Ci-dessous, en utilisant MiniMind comme exemple, nous utilisons des textes populaires tirés de *Journey to the West* de différentes longueurs comme entrée, en comparant les variations de perplexité (PPL) avant et après activation de l'extrapolation de RoPE. On constate que, dans les scénarios de texte long, la perplexité du modèle diminue significativement après activation de l'extrapolation YaRN :

<div align="center">
<img src="./images/rope_ppl.png">
</div>

> Comparaison de la perplexité de MiniMind avant et après activation de YaRN à différentes longueurs de texte

---

<a id="ⅴ-objective-evaluation"></a>
## Ⅴ Évaluation objective

Cette section présente les résultats d'évaluation sur plusieurs modèles de langage à échelle micro. Les benchmarks sélectionnés sont C-Eval, CMMLU, ARC-Easy, PIQA, OpenBookQA, HellaSwag et Social-IQa ; tous sauf les deux premiers sont des benchmarks en anglais.


Le cadre d'évaluation choisi est [lm-evaluation](https://github.com/EleutherAI/lm-evaluation-harness)

```bash
# Installation
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e .
```



```bash
# Start testing
# Datasets used: ceval-valid/cmmlu/arc_easy/piqa/openbookqa/hellaswag/social_iqa # View supported datasets: lm_eval ls tasks
# For instruction-tuned models, add --apply_chat_template during evaluation; for base models such as gpt2, you do not need it.
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf --model_args pretrained="/path/to/model",dtype=auto --tasks "task" --batch_size 16 --device cpu --trust_remote_code --apply_chat_template
```

> Remarque : ces évaluations par choix multiple ne sont généralement pas effectuées en demandant au modèle de générer librement la réponse complète. Au lieu de cela, étant donné un contexte `y` et un ensemble d'options `x`, la pratique standard consiste à comparer la probabilité conditionnelle `p(x | y)` de chaque option et à sélectionner celle ayant le score le plus élevé. Si une option correspond à un seul token, comparer la probabilité de ce token suffit ; si elle s'étend sur plusieurs tokens, une approche plus standard consiste à comparer la somme des probabilités logarithmiques conditionnelles sur l'ensemble de l'option. Les options ne sont pas nécessairement `A`, `B`, `C`, `D` ; certains jeux de données ne disposent que de deux options. Dans ce sens, deviner aléatoirement constitue déjà une borne inférieure assez forte, et les modèles de cette taille tendent à rester proches de cette borne pendant une longue période.

MiniMind est entraîné sur beaucoup moins de données que les autres modèles listés ici, et sa composition d'entraînement est fortement biaisée vers la langue chinoise, ce qui explique sa performance anglaise relativement faible. Il n'est pas spécifiquement aligné sur ce format d'évaluation par choix multiple par défaut, donc sa performance est relativement faible, et les résultats sont uniquement à titre d'information :

|nom du modèle|de|paramètres|zh (ceval / cmmlu)|en (arc / piqa / obqa / hellaswag / siqa)|
|---|---|---|---|---|
|minimind-3|actuel|64M| 24.89 / 25.38 | 28.49 / 50.65 / 23.60 / 28.28 / 34.19 |
|minimind-3-moe|actuel|198M| 25.48 / 24.32 | 27.74 / 50.71 / 26.20 / 27.43 / 34.03 |
|minimind-3-exam|actuel|64M| 30.98 / 26.12 | 35.61 / 56.26 / 24.20 / 28.40 / 34.19 |
|[Steel-LLM](https://huggingface.co/gqszhanshijin/Steel-LLM)|ZhanShiJin|1121M| 24.89 / 25.32 | 39.69 / 65.13 / 26.00 / 35.73 / 39.15 |
|[gpt2-medium](https://huggingface.co/openai-community/gpt2-medium)|OpenAI|360M| 23.18 / 25.00 | 43.60 / 66.38 / 30.20 / 39.38 / 39.10 |
|[TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)|TinyLlama|1100M| 25.71 / 25.03 | 54.80 / 74.43 / 35.60 / 60.38 / 43.09 |
|[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)|HuggingFace|135M| 24.44 / 24.71 | 58.50 / 68.17 / 32.80 / 43.15 / 39.46 |
|[Aquila-135M](https://huggingface.co/BAAI/Aquila-135M-Instruct)|BAAI|135M| 25.19 / 25.10 | 54.59 / 67.52 / 34.40 / 41.67 / 39.66 |

<details>
<summary> <strong> Remarque supplémentaire (source / pas de contamination / reproduction)</strong> </summary>

minimind-3-exam n'est pas un modèle de base plus grand, et il ne contient peu ou pas de nouvelles connaissances. Il s'agit simplement de minimind-3 après une légère alignement LoRA sur [lora_exam.jsonl](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/lora_exam.jsonl), avec [lora_exam_768.pth](https://huggingface.co/jingyaogong/minimind-3-pytorch/resolve/main/lora_exam_768.pth) fusionné dans le modèle de base. Ces données d'alignement sont extraites des sous-ensembles de test de ceval et de mmlu en anglais, avec une augmentation par préfixe/suffixe. Son objectif est d'aligner le format de contexte et d'options couramment rencontré dans les évaluations par choix multiple, et non pas d'enseigner les réponses.

Les 7 benchmarks utilisés dans cette section n'ont aucun overlap avec les données d'alignement ci-dessus, donc ce résultat peut être considéré comme libre de contamination de données. En revanche, si l'on fine-tune directement sur des données superposées, les scores d'un petit modèle peuvent être fortement distordus ; par exemple, minimind-3 a atteint une précision d'environ 97 % sur des sous-ensembles contaminés de ceval / cmmlu, mais ces chiffres ne sont pas significatifs.

Ce que cette expérience suggère est simple : pour ce type de benchmark, le goulot d'étranglement d'un modèle petit ne se situe pas nécessairement dans les connaissances elles-mêmes, mais aussi dans la conformité de la forme d'entrée. Avec seulement une petite quantité d'alignement de format, minimind-3-exam améliore en moyenne de 2,9 points de pourcentage sur les 7 tâches ci-dessus.

</details>

![benchmark_radar](./images/benchmark_radar.jpg)

---

<a id="-others"></a>
# 📌 Autres

<a id="-model-conversion"></a>
## 🔧 Conversion du modèle

* [./scripts/convert_model.py](../../scripts/convert_model.py) peut être utilisé pour la conversion entre les formats de modèles `torch` et `transformers`.
* Sauf indication contraire, les modèles publiés dans la ligne principale de la série `MiniMind` sont généralement fournis au format `Transformers`. Si vous utilisez des poids natifs `torch`, exécutez d'abord la conversion `torch2transformers`.


<a id="️-api-service-interface-based-on-minimind"></a>
## 🖥️ Interface de service API basée sur MiniMind

* [./scripts/serve_openai_api.py](../../scripts/serve_openai_api.py) fournit un service de chat léger compatible avec l'API OpenAI, facilitant ainsi la connexion de vos propres modèles à des interfaces tierces telles que FastGPT, OpenWebUI, Dify, etc.
* Le serveur API prend en charge également des champs tels que `reasoning_content`, `tool_calls` et `open_thinking`, rendant ainsi le service adapté aux scénarios de appel d'outils / de réflexion.

* Après avoir téléchargé les poids du modèle depuis [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5), l'exemple de structure du répertoire est le suivant :
    ```
    minimind (root dir)
    ├─<MiniMind-Model-Name> (e.g., minimind-3)
    |  ├── config.json
    |  ├── generation_config.json
    |  ├── model_minimind.py (optional, depending on the export format)
    |  ├── pytorch_model.bin or model.safetensors
    |  ├── special_tokens_map.json
    |  ├── tokenizer_config.json
    |  ├── tokenizer.json
    ```

* Démarrer le serveur
    ```bash
    cd scripts && python serve_openai_api.py
    ```
* Tester l'interface de service
    ```bash
    cd scripts && python chat_api.py
    ```
* Exemple de requête API (compatible avec le format OpenAI)
    ```bash
    curl http://localhost:8998/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "model-identifier",
        "messages": [
          { "role": "user", "content": "世界上最高的山是什么？" }
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": true,
        "open_thinking": true
    }'
    ```

<a id="sglang"></a>
## <img src="https://avatars.githubusercontent.com/u/147780389?s=48&v=4" height="28" style="vertical-align: middle;"/> [SGLang](https://github.com/sgl-project/sglang)

SGLang est un moteur d'inférence haute performance pour les grands modèles linguistiques, avec des optimisations telles que RadixAttention et le batch continu, offrant une latence réduite et un débit accru.

> ⚠️ Nécessite un environnement CUDA, à utiliser selon les besoins. Vous pouvez également choisir SGLang comme moteur d'inférence ou de rollout dans les scripts d’entraînement par renforcement afin d'améliorer le débit de formation.

Lancer le modèle en tant que serveur API compatible OpenAI :

```bash
python -m sglang.launch_server --model-path /path/to/model --attention-backend triton --host 0.0.0.0 --port 8998
```

<a id="vllm"></a>
## <img src="https://avatars.githubusercontent.com/u/136984999" height="28" style="vertical-align: middle;"/> [vllm](https://github.com/vllm-project/vllm)

vLLM est un cadre d'inférence efficace largement utilisé pour la déploiement rapide des grands modèles linguistiques, offrant un bon équilibre entre efficacité mémoire et débit.

> ⚠️ Nécessite un environnement CUDA, à utiliser selon les besoins.

Lancer le modèle en tant que serveur API compatible OpenAI :

```bash
vllm serve /path/to/model --model-impl transformers --served-model-name "minimind" --port 8998
```

<a id="llamacpp"></a>
## <img src="https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png" height="28" style="vertical-align: middle;"/> [llama.cpp](https://github.com/ggerganov/llama.cpp)

llama.cpp est un cadre d'inférence léger et pratique en C++, pouvant être utilisé directement depuis la ligne de commande. Il prend en charge l'inférence multithreadée et plusieurs options d'accélération GPU.

**Structure du répertoire** : Il est recommandé de placer `llama.cpp` et le répertoire du modèle au même niveau.

```
parent/
├── project/           # your project directory
│   ├── minimind-model/       # HuggingFace-format model directory
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── ...
│   └── ...
└── llama.cpp/         # llama.cpp directory
    ├── build/
    ├── convert_hf_to_gguf.py
    └── ...
```

0. Voir la documentation officielle `llama.cpp` pour compléter l'installation (dépendances telles que `cmake`, etc.)

1. Insérer à la fin de la fonction `get_vocab_base_pre` dans `convert_hf_to_gguf.py` :

```python
# Add MiniMind tokenizer support. A compatible fallback such as qwen2 can be reused temporarily.
if res is None:
    res = "qwen2"
```

2. Convertir le modèle minimind de format HuggingFace en GGUF :

```bash
# Run this under the llama.cpp directory. The GGUF file will be generated in the model directory.
python convert_hf_to_gguf.py /path/to/minimind-model
```

3. Quantifier le modèle (facultatif)

```bash
./build/bin/llama-quantize /path/to/model/xxxx.gguf /path/to/model/xxxx.q8.gguf Q8_0
```

4. Test d'inférence par ligne de commande

```bash
./build/bin/llama-cli -m /path/to/model/xxxx.gguf
```

<a id="ollama"></a>
## <img src="https://ollama.com/public/cloud.png" height="28" style="vertical-align: middle;"/> [ollama](https://ollama.ai)

Ollama est un outil couramment utilisé pour exécuter des grands modèles localement. Il prend en charge de nombreux modèles LLM open-source et propose un workflow simple avec un faible coût d'installation.

1. Charger un modèle GGUF personnalisé via Ollama

Créez un nouveau fichier `minimind.modelfile` dans le répertoire modèle et écrivez le modèle de configuration suivant. Vous pouvez remplacer le système de prompt selon vos besoins :

<details>
<summary> minimind.modelfile (modèle de template)</summary>

```text
FROM /path/to/model/xxxx.gguf

SYSTEM "你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。"


TEMPLATE """{{- if .Tools }}<|im_start|>system
{{ if .System }}{{ .System }}

{{ end }}# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{{ else if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
<think>
{{ .Thinking }}
</think>

{{ .Content }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}
{{- end }}
{{- if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ if and $.IsThinkSet $.Think -}}
<think>
{{ else -}}
<think>

</think>

{{ end -}}
{{ end }}
{{- end }}"""

PARAMETER repeat_penalty 1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.9
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
```

</details>
<br/>


2. Chargez et nommez le modèle local

```bash
ollama create -f minimind.modelfile minimind-local
```

3. Démarrer l'inférence

```bash
ollama run minimind-local
```

<details>
<summary>📤 Pousser votre modèle sur Ollama Hub</summary>

```bash
# 1. Rename the local model to your ollama-account/minimind tag
ollama cp minimind-local:latest your_username/minimind:latest

# 2. Push the model
ollama push your_username/minimind:latest
```
</details>
<br/>

⭐️ Vous pouvez également utiliser directement le modèle Ollama que je vous fournis pour un démarrage rapide :

```bash
ollama run jingyaogong/minimind-3
>>> 你叫什么名字
我是一个语言模型...
```

<a id="mnn"></a>
## <img src="https://avatars.githubusercontent.com/u/1961952?s=48&v=4" height="28" style="vertical-align: middle;"/> [MNN](https://github.com/alibaba/MNN)

MNN est un moteur d'inférence IA conçu pour les appareils périphériques, permettant une déploiement léger et une inférence performante pour divers modèles LLM open source.

1. Conversion du modèle
```bash
cd MNN/transformers/llm/export
# Export a 4-bit HQQ-quantized MNN model
python llmexport.py --path /path/to/model --export mnn --hqq --dst_path /path/to/model-mnn
```

2. Testez sur des appareils Mac ou mobiles
```bash
./llm_demo /path/to/model-mnn/config.json prompt.txt
```
Ou testez-le avec l'application mobile.

> Pour plus d'informations sur ces frameworks tiers, veuillez consulter leurs documents officiels.


<a id="-more-content"></a>
## 👨‍💻 Plus de contenu

* <a href="https://github.com/jingyaogong/minimind/discussions/618">🔗 Fine-tuning de modèles de langage diffusion à partir de MiniMind-LLM</a>

* <a href="https://github.com/jingyaogong/minimind/discussions/611">🔗 Description de la méthode generate du modèle</a>

* <a href="https://github.com/jingyaogong/minimind/discussions/704">🔗 Formation de modèles d'attention linéaire à partir de MiniMind</a>

<a id="-acknowledgments"></a>
# 📌 Remerciements

> [!NOTE] Si les projets de la série `MiniMind` vous ont été utiles, n'hésitez pas à étoiler ⭐ sur GitHub<br/> La documentation est longue et des omissions sont inévitables. Des retours via Issues ou PRs sont les bienvenus pour améliorer ensemble le projet. <br/> Vos retours et suggestions sont des forces motrices essentielles pour l'itération continue de ce projet !

<a id="contributors"></a>
## 🤝[Contributeurs](https://github.com/jingyaogong/minimind/graphs/contributors)

<a href="https://github.com/jingyaogong/minimind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jingyaogong/minimind" />
</a>

<a id="acknowledgments"></a>
## 😊Remerciements

Merci aux contributeurs suivants d'avoir partagé des notes de formation, des expériences de traitement des données, des tutoriels et des démonstrations de code :

* [@ipfgao](https://github.com/ipfgao): [🔗Enregistrements des étapes de formation](https://github.com/jingyaogong/minimind/issues/26)

* [@WangRongsheng](https://github.com/WangRongsheng): [🔗Prétraitement de grands ensembles de données](https://github.com/jingyaogong/minimind/issues/39)

* [@pengqianhan](https://github.com/pengqianhan): [🔗Tutoriel concis](https://github.com/jingyaogong/minimind/issues/73)

* [@RyanSunn](https://github.com/RyanSunn): [🔗Notes d'apprentissage du processus d'inférence](https://github.com/jingyaogong/minimind/issues/75)

* [@Nijikadesu](https://github.com/Nijikadesu): [🔗Découpage du code du projet en format de notebook interactif](https://github.com/jingyaogong/minimind/issues/213)


Merci également aux travaux suivants et projets :

- [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
- [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)
- [https://github.com/DLLXW/baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [https://github.com/charent/ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)
- [https://github.com/wdndev/tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh)
- [Mistral-MoE](https://arxiv.org/pdf/2401.04088)
- [https://github.com/Tongjilibo/build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch)
- [https://github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
- [https://github.com/AI-Study-Han/Zero-Chatgpt](https://github.com/AI-Study-Han/Zero-Chatgpt)
- [https://github.com/xusenlinzy/api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
- [https://github.com/HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)


<a id="supporters"></a>
## 🫶 Appuis

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind"/>
  <img alt="Fork poster" src="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind&theme=dark"/>
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=jingyaogong/minimind&type=date&theme=dark&legend=top-left&sealed_token=DK6jy_uvw2AHIK0S4VZLf6snWIQ06jGzz3QiwVmXBGDvickcQgJGSdazdGxjRQZuj8Hr3GfS_REB9ohoK8NWVsmukeOQiT4soChw3_19yyPVwvWzBp66yMYWlvOYy9sv60cMSntByiUTcyp4MrRiMm1JD1MSC8NJ-Z9qhR9uJGl2AU7w-OGlyKQzN7Xa"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=jingyaogong/minimind&type=date&legend=top-left&sealed_token=DK6jy_uvw2AHIK0S4VZLf6snWIQ06jGzz3QiwVmXBGDvickcQgJGSdazdGxjRQZuj8Hr3GfS_REB9ohoK8NWVsmukeOQiT4soChw3_19yyPVwvWzBp66yMYWlvOYy9sv60cMSntByiUTcyp4MrRiMm1JD1MSC8NJ-Z9qhR9uJGl2AU7w-OGlyKQzN7Xa"/>
  <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=jingyaogong/minimind&type=date&legend=top-left&sealed_token=DK6jy_uvw2AHIK0S4VZLf6snWIQ06jGzz3QiwVmXBGDvickcQgJGSdazdGxjRQZuj8Hr3GfS_REB9ohoK8NWVsmukeOQiT4soChw3_19yyPVwvWzBp66yMYWlvOYy9sv60cMSntByiUTcyp4MrRiMm1JD1MSC8NJ-Z9qhR9uJGl2AU7w-OGlyKQzN7Xa"/>
</picture>

<a id="-minimind-related-achievements"></a>
## 🎉 Réalités liées à MiniMind

Ce modèle a servi de fondation à plusieurs résultats de recherche satisfaisants. Merci aux chercheurs d'avoir reconnu ce travail :

- ECG-Expert-QA : Un benchmark pour évaluer les grands modèles de langage médicaux dans le diagnostic des maladies cardiaques [[arxiv](https://arxiv.org/pdf/2502.17475)]

- Algorithme basé sur la programmation binaire-entier pour l'équilibrage de charge des experts dans les modèles de mélange d'experts [[arxiv](https://arxiv.org/pdf/2502.15451)]

- LegalEval-Q : Un nouveau benchmark pour l'évaluation de la qualité des textes générés par les grands modèles de langage [[arxiv](https://arxiv.org/pdf/2505.24826)]

- Sur la capacité de généralisation de l'entraînement prédictif du mot suivant [[ICML 2025](https://openreview.net/forum?id=hLGJ1qZPdu)]

- 《Apprendre à construire un grand modèle : de la réseaux de neurones au Transformer》Édité par Wang Shuang, Mo Chen et Wang Hao Yi - Éditions Tsinghua

- FedBRB : Une solution pour le scénario petit à grand dans l'apprentissage fédéré avec hétérogénéité des appareils [[TMC 2025](https://ieeexplore.ieee.org/abstract/document/11168259)]

- SKETCH : Conditionnement par point sémantique pour la prédiction de trajectoire de vaisseau à long horizons [[arxiv](https://arxiv.org/pdf/2601.18537)]

- Un expert cryptographique intégré pour l'intelligence artificielle : jusqu'où va l'horizon ? [[IACR ePrint 2026](https://eprint.iacr.org/2026/411.pdf)]

- RetryTrigger : Duplication d'inférence intelligente pour renforcer la résilience des grands modèles linguistiques face aux pannes transitoires du matériel [[FITEE 2026](https://ieeexplore.ieee.org/abstract/document/11479682)]

- En cours...


<a id="-citation"></a>
# 🎓 Citation

Si `MiniMind` a été utile à votre recherche ou à votre travail, n'hésitez pas à citer :

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind},
  note = {GitHub repository}
}
```

<a id="️-license"></a>
# ⚖️ Licence

Ce projet est open-source sous la [Licence Apache 2.0](../../LICENSE).
