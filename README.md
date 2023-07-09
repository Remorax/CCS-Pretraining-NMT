# Contextual Code-Switching for Pretraining in mNMT
This repository contains the code for our EACL 2023 (Findings) publication "[Exploring Contextual Code-Switching for Pretraining in Multilingual Neural Machine Translation](https://aclanthology.org/2023.findings-eacl.72/)". 

In this work, we show how to leverage massive multilingual NMT models (like mBART50) to pretrain small, high-performing models with minimal data and compute requirements. Pretraining is done using a superior code-switched noising algorithm called Contextual Code-Switching (CCS) - which uses these massive models to generate contextual, many-to-many substitutions for constructing synthetic code-switchd pretraining corpora. We observe that our models perform comparably or better than massive models, depending on the amount of data provided.

## Installation

Run the following command:

```conda env create -f environment.yml ```

### The mCOLT repository

We use the mCOLT fairseq module from https://github.com/PANXiao1994/mRASP2/ and make some minor changes to make it suitable for our work. We include the same for reproducibility. Thanks to Pan et al. for providing the original module!

## Pipeline

### Preprocessing

The first step is to generate translations and alignments using a base NMT model and a word-aligner respectively. We primarily use mBART50 and awesome-align in our work. We also use `from-scratch` model as an alternative. The scripts we use for all of these models are in the `preprocessing` dir.

### CCS + Training

Once the translations and alignments are generated, we run: 

`bash preprocess.sh`  

`preprocess.sh` calls `ccs.py` which will code-switch the corpus for you. On completion, it calls `train.sh` to train the model using the generated code-switched corpus. 

Once training is complete, `train.sh` also contains commands to evaluate the trained models on the test sets.

#### Note for SLURM users

SLURM users can run:

`sbatch preprocess.sh`

It is recommended to use CPU nodes for `preprocess.sh` and GPU nodes for `train.sh`.

### Fine-Tuning 

The scripts to run the fine-tuning experiments are in `finetuning`. Run `bash preprocess.sh`, just as before. After code-switching, it will call `pretrain.sh` to pretrain the model on monolingual code-switched corpora. Then, `pretrain.sh` calls `mlft.sh` (for Multilingual Fine-Tuning) and `blft.sh` (for Bilingual Fine-Tuning) to fine-tune on real (unnoised) corpora.

## Baselines

We include scripts for all baselines mentioned in this paper in `baselines`. Namely, it includes scrips to train and evaluate Aligned Augmentation (Pan et al., 2021) and Knowledge Distillation (Hinton et al., 2015) as well as the massively multilingual models mBART50 (Tang et al., 2021) and mRASP2 (Pan et al., 2021).

## Acknowledgements

The Aligned Augmentation scripts were built using the instructions and the scripts provided in the  [mRASP2](https://github.com/PANXiao1994/mRASP2/) repository. For mBART50 and mRASP2, we use the checkpoints available [here](https://github.com/facebookresearch/fairseq/tree/main/examples/multilingual#mbart50-models) and [here](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2021/mrasp2/12e12d_last.pt). All due credits to the concerned authors for their excellent works.
