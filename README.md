# Neural Networks Are More Modular Than They Appear

This repository contains the code accompanying the paper **“Neural Networks Are More Modular Than They Appear.”**

The synthetic datasets used in the paper can be found **[here](https://huggingface.co/datasets/shreyasaha25/Neural-Network-Modularity/tree/main)**.

If you encounter any issues running the code, please contact **Shreya Saha** at [ssaha@ucsd.edu](mailto:ssaha@ucsd.edu).

---

## 1. Training Disentangled Transformations

The script **train_disentangled_invertible_transformations.py** is used to train transformations that disentangle representations in pretrained vision and language models.

### Vision Models

For vision models, use:

```bash
python3 train_disentangled_invertible_transformations.py \
    --model_name swinb \
    --transformation_type invertible \
    --syntactic_uints_percentage 0.4375 \
    --semantic_uints_percentage 0.4375 \
    --training_datasets mscoco_stylized_shape_texture
```

In the paper, vision transformations are trained **only on the stylized MS-COCO dataset**

### Language Models

For language models, use:

```bash
python3 train_disentangled_invertible_transformations.py \
    data_type language \
    --model_name gpt2_xl \
    --transformation_type invertible \
    --syntactic_uints_percentage 0.4 \
    --semantic_uints_percentage 0.4 \
    --training_datasets msr_paraphrase_corpus
```

In the paper, language transformations are trained **only on the MSR Paraphrase Corpus**.

### Important Configuration

When training transformations, make sure to specify:

* `data_type`: the type of data/model being used (e.g., vision or language).
* `transformation_type`: the type of transformation being trained.
* `training_datasets`: the dataset used to train the transformation.
* `semantic_uints_percentage` and `syntactic_uints_percentage`: the fractions of model units assigned to each subspace. **Note:** the unit names `semantic` and `syntactic` are hardcoded throughout the repository and are also used for the corresponding **content** and **style** subspaces when training vision models. This naming convention is assumed throughout the codebase.
The unit percentages must correspond to **whole numbers of units** for the specific model being used. Since different models have different numbers of units, the appropriate percentages will differ across models.

---

## 2. Vision Model Inference

The following scripts are used to run inference with the learned transformations on vision models:

```text
mscoco_retrieval.py
mscoco_sanity.py
mscoco_top100_inference.py
```

These scripts perform the inference and evaluation experiments reported in the paper.

The corresponding analyses and plots are available in the notebook **Plots and Analysis**.

These analyses correspond to **Figures 3,4,5 and Table 1.**

---

## 3. Language Model Analysis

The analyses for the language-model experiments are performed directly in the **Plots and Analysis** notebook. Specifically, look under the headers **Sanity test on language models**, **Nearest Neighbor - Language (Example)**, **Result 5 - Layerwise Comparisons**, **Results 4 - Retrieval Results with Language**. 

---

## 4. Brain Data Analysis

The preprocessed **Natural Scenes Dataset (NSD)** brain data used in this paper can be obtained **[here](#)**.

The script **vision_brain_data_inferences.py** runs the ablation studies on visual brain areas using the learned visual transformations.

These experiments correspond to **Figure 8B–C** in the paper.

The corresponding figure generation and analysis can be found in notebook **Brain Data Preprocessing**.

---
