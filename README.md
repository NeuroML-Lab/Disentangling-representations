# Neural Networks Are More Modular Than They Appear

This repository contains the code accompanying the paper **[“Neural Networks Are More Modular Than They Appear.”](https://arxiv.org/abs/2610.12842)**

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

The preprocessed **Natural Scenes Dataset (NSD)** brain data used in this paper can be obtained **[here](https://osf.io/wh35d/?view_only=202cd5926f094b2693095f744ffdafbc)**.

The script **vision_brain_data_inferences.py** runs the ablation studies on visual brain areas using the learned visual transformations.

These experiments correspond to **Figure 8B–C** in the paper.

The corresponding figure generation and analysis can be found in notebook **Brain Data Preprocessing**.

---

## 5. RNN Modularity: Yang et al. (2019) and Mante et al. (2013)

All code for the recurrent-network experiments is in **rnn_modularity_yang_mante/**. Run every command below **from inside that folder**, because the scripts use relative paths.

The trained RNNs, activations and results are **not included in this repository**. You create them yourself by running the steps below. The scripts write their outputs to these folders:

| Folder | Created by | Contents |
|---|---|---|
| `mante_paper/seed<N>/` | `mante_modularity_v2.py` | Trained Mante RNNs, activations, learned decompositions and ablation results (one folder per seed) |
| `mante_paper/` | `tdr_analysis.py`, `plot_mante_fig.py` | TDR results and the Mante figure |
| `yang_data/` | `yang_modularity.py`, `yang_ftv_lesion.py`, `plot_yang_final.py` | Trained Yang RNNs, analysis results, FTV lesion results and the Yang figures |

Both sets of experiments use the same approach. We find a subspace for each task in a trained RNN at three levels of modularity (**unit**, **orthogonal** and **linear**), ablate each subspace and measure the effect on both tasks. A **double dissociation** means that ablating subspace A mainly hurts task A and ablating subspace B mainly hurts task B. The **Double Dissociation Index (DDI)** measures how strong it is.

### 5.1 Mante et al. (2013): Context-Dependent Integration

The network receives noisy **motion** and **color** coherence inputs and a context cue, and must integrate only the cued input (fixation 200 ms, stimulus 800 ms, dt = 20 ms). The question is whether motion and color are processed in separate units, separate orthogonal subspaces or separate linear (non-orthogonal) subspaces.

The steps must be run in order, because each one reads the files written by the step before it. The full pipeline takes about 3–4 hours on a GPU.

**Step 1: Train the RNNs (10 seeds).**

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python mante_modularity_v2.py train \
        --n_units 256 --n_epochs 5000 \
        --out_dir ./mante_paper/seed${seed} --seed $seed --n_extract 5000
done
```

For each seed this creates `mante_paper/seed<N>/` and writes:
* `rnn_model.pt` and `train_history.json`: the trained RNN and its training curve.
* `activations.pt`: hidden states from `--n_extract` trials. Step 2 needs this file.

**Step 2: Learn the decompositions and run the ablations.**

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python mante_modularity_v2.py analyze \
        --out_dir ./mante_paper/seed${seed} \
        --n_units 256 \
        --levels unit,orthogonal,linear \
        --dim_A 2,5,10,15,20,50,100 \
        --kappa 1.0,1.5,2.0,3.0,5.0 \
        --n_epochs_decomp 2000 \
        --n_causal 5000
done
```

Each seed takes about 20 minutes. This adds the following files to each `seed<N>/` folder:
* `decomp_{unit,orthogonal,linear_k<kappa>}_d<dim>.pt`: one learned decomposition for each level and dimension.
* `modularity_results.json`: offline and causal (ablation) results for each dimension and level.

Key arguments:
* `--dim_A`: the subspace dimensions to sweep, as a comma-separated list.
* `--kappa`: the condition-number bound for the **linear** level, as a comma-separated list. Larger values allow subspaces that are less orthogonal.
* `--n_causal`: the number of trials used to measure accuracy after ablation.

**Step 3: Aggregate across seeds.**

```bash
python aggregate_mante.py --data_dir ./mante_paper --n_seeds 10
```

This reads each `seed<N>/modularity_results.json` and prints unit-level selectivity and the causal double dissociation for each dimension and level, as mean ± std across seeds. It does not write any files.

**Step 4: Targeted Dimensionality Reduction (TDR) baseline.**

```bash
python tdr_analysis.py --data_dir ./mante_paper --n_seeds 10 --dim_A 50
```

This repeats the regression-axis analysis from Mante et al. (motion, color, context and choice axes). It measures how orthogonal the axes are, ablates along them and compares the result with ablating the learned `orthogonal` and `linear_k2.0` decompositions at `--dim_A`. Those decompositions must exist from Step 2, so `--dim_A` must be one of the dimensions you swept and `2.0` must be one of the kappa values. The output is saved to `mante_paper/tdr_results.json`.

**Step 5: Make the figure.**

```bash
python plot_mante_fig.py --data_dir ./mante_paper --n_seeds 10
```

This saves `mante_paper/fig_mante_main.{pdf,png}`, which shows the double dissociation at each level, DDI against dimension, and accuracy after ablation against dimension.

For a quick test, train with one seed and fewer epochs (for example `--n_epochs 2000`) and analyze with fewer dimensions (for example `--dim_A 10,50 --kappa 2.0`). Then pass `--n_seeds 1` to Steps 3–5.

### 5.2 Yang et al. (2019): Multitask RNN

These experiments use the multitask RNNs of Yang et al., trained on 11 tasks (Go, Anti, DM1, DM2, Ctx_DM1, Ctx_DM2, MultSen_DM, Dly_DM1, DMS, DMC, DNMC) with **softplus** and **tanh** activations.

Task pairs analyzed (`KEY_PAIRS` in `yang_ftv_lesion.py`):

| Relationship | Task A | Task B |
|---|---|---|
| disjoint | Anti | DM1 |
| inclusive | DM1 | Dly_DM1 |
| mixed | DM1 | Ctx_DM1 |
| disjoint_equal | Ctx_DM1 | Ctx_DM2 |
| disjoint_mixed | DMC | DNMC |

**Step 1: Train the RNNs and run the modularity analysis.** This is done by `yang_modularity.py`. For each activation (softplus and tanh), it must write these files to `yang_data/`:
* `<activation>_seed0_model.pt` and `<activation>_seed0_history.json`: the trained RNN and its training curve.
* `analysis_<activation>.json`: FTV and decomposition (unit, Sinkhorn, orthogonal and linear) ablation results for every task pair.

> **Note:** `yang_modularity.py` is not yet in this repository. Steps 2 and 3 cannot be run until it is added.

**Step 2: FTV lesioning (replicates Yang et al. Fig. 2e and 5b).** For each pair, units are grouped by Fractional Task Variance (FTV) at thresholds 0.3, 0.5 and 0.7. Each group is lesioned and performance on both tasks is measured.

```bash
python yang_ftv_lesion.py --model_path ./yang_data/softplus_seed0_model.pt --activation softplus
python yang_ftv_lesion.py --model_path ./yang_data/tanh_seed0_model.pt --activation tanh
```

This reads the models from Step 1 and saves `yang_data/ftv_lesion_{softplus,tanh}.json`.

**Step 3: Make the figures.**

```bash
python plot_yang_final.py
```

This reads `yang_data/analysis_{softplus,tanh}.json` from Step 1 and saves three figures to `yang_data/`:
* `fig_yang_modularity`: the main figure.
* `fig_yang_allpairs`: results for every task pair.
* `fig_yang_ddi_levels`: DDI for Anti vs DM1 at each decomposition level (Unit/FTV, Sinkhorn, Orthogonal, Linear).

The script flips the A/B labels where needed so that the "Anti subspace" bar always means the subspace whose ablation hurts Anti more.

### Notes

* `tdr_analysis.py` imports from `mante_modularity`, but the file in this folder is `mante_modularity_v2.py`. Before running the TDR step, either change the import in `tdr_analysis.py` or copy the file to `mante_modularity.py`. `mante_modularity_v2.py` provides all the classes it needs (`ContextRNN`, `TrialConfig`, `generate_trials`, `OrthogonalModularity`, `LinearModularity`).
* `yang_ftv_lesion.py` imports from `yang_modularity`, which is not yet in this folder (see Section 5.2, Step 1).

---
