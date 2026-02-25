# **Project Overview**

This repository provides the implemented code for the **Research Project**, as part of the **MEng in Computer Science course program at IMT Atlantique** and also the **Data2Laws research initiative**.

More specifically, it explores the use of **Natural Language Inference (NLI)** to analyze collaboration between users by **classifying interactions into a useful taxonomy**, providing **structured insights into community dynamics**.

# **Context**

The **Data2Laws project** investigates large-scale online collectives and their mechanisms for **growth, sustainability, and collaborative governance**. Understanding the dynamics of **user interactions** is crucial for analyzing how these communities **evolve and maintain cohesion**.

However, extracting meaningful insights about collaboration from these interactions is challenging due to:

* **The unstructured nature of textual messages**.
* **Ambiguity in user intents and contributions**.
* **The need to classify interactions according to collaborative behaviors**.

This project addresses these challenges by leveraging the use of **Natural Language Inference (NLI)**. The approach is to **classify user interactions into a taxonomy of collaboration using NLI models**, enabling **structured analysis of how users engage and contribute collectively**.

This approach builds on recent work [1], which showed that **NLI is efficient for classifying textual requirements tasks** due to key advantages:

* **Label verbalization**: describing classes in natural language to specify categories without relying on fixed or numeric labels, and allowing for **more interpretable predictions**.
  
* **Zero-shot learning capability**: leveraging general language understanding to match statements with verbalized labels. This reduces the need for **extensive labeled datasets** and enables **rapid adaptation to new classification tasks**, including novel collaborative behaviors.

This project evaluates how these advantages apply to analyzing **interactions in code review datasets**.

# **Approach**

**Dataset & Model**: We used 10,045 manually annotated code review comments grouped into five main categories [2]. The model used is MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33 based on DeBERTa-v3 and trained on 33 NLI datasets [3].

**Experiments**: Three complementary aspects were explored in this project:

1. **Zero‑shot learning performance**  
   Models were evaluated without any task‑specific training to measure how well general NLI capabilities allow them to recognize and categorize collaborative interaction types.

2. **Impact of label verbalization**  
   The effect of **verbalizing taxonomy labels** in natural language was examined to see if such verbalization improves classification clarity and model performance.

3. **Fine‑tuning performance**  
   The models were fine‑tuned on the training and validation sets derived from the interaction dataset to measure how much performance improves when models are adapted specifically to the collaborative interaction domain.

# **Installation**

```bash
# Clone the repository
git clone https://github.com/JLiOngit/Data2Laws---NLI-classification.git
cd Data2Laws---NLI-classification

# Install dependencies
pip install -r requirements.txt
```

# **Running**

## Zero-Shot Classification

No training required. Runs directly on the test set using a pre-trained NLI model. Available via `notebook.ipynb`.

## Fine-Tuning

This approach adapts a pre-trained NLI model to the task through supervised fine-tuning on the training set. Training hyperparameters are configured in `src/config.py`, and dataset paths (`TRAIN_PATH`, `TEST_PATH`) can either be set there or overridden at runtime via CLI arguments.

```bash
python -m src.main [OPTIONS]
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `deberta_large` | Model to fine-tune |
| `--label_name` | `str` | `group` | Target label column (`group` or `label`) |
| `--label_verbalization` | `bool` | `True` | Use natural language label descriptions |
| `--train_path` | `str` | *(from config)* | Path to the training `.jsonl` file |
| `--test_path` | `str` | *(from config)* | Path to the test `.jsonl` file |


# **Analysis**
The results of the experiments are described in the article [*"Transformer les interactions utilisateurs en taxonomie grâce au Natural Language Inference (NLI)"*.](https://data2laws.wp.imt.fr/2026/02/16/transformer-les-interactions-utilisateurs-en-taxonomie-grace-au-natural-language-inference-nli/) and have also been summarized in a poster, as part of the final delivery of the **Research Project course project**..

# **References**

[1] FAZELNIA, Mohamad, KOSCINSKI, Viktoria, HERZOG, Spencer, et al. Lessons from the Use of Natural Language Inference (NLI) in Requirements Engineering Tasks.

[2] PETROVA, P. A., MARKOV, S. I., and KACHANOV, V. V. Building a Dataset for Combined Classification of Source Code Reviews. Pattern Recognition and Image Analysis, 2025, vol. 35, no 3, p. 482-492.

[3] M. Laurer, W. Atteveldt, A.Casas, K. Welbers, Building Efficient Universal Classifiers with Natural Language Inference.
