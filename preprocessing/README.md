## Preprocessing

This folder contains the scripts we use for generating translations using [mBART50](https://github.com/facebookresearch/fairseq/tree/main/examples/multilingual#mbart50-models) and word alignments using [awesome-align](https://github.com/neulab/awesome-align). These scripts were built using the example scripts provided in the linked repos, so all due credits to Tang et al., 2021 and Duou et al., 2021. 

`launch.sh` in `mBART50-translations` is the script to be run - it calls `translate.sh` to generate the relevant translations. `translate.sh`, in turn, calls the `awesome-align` scripts to extract alignments for the generated translations.

We also include scripts to train models from scratch as the base models (in the `from-scratch/train` folder), and to generate translations (`from-scratch/translate.sh`). In the latter, again run `launch.sh` so that it can call translate.sh. 

### Note: Generating translations using mBART50 

Due to some version incompatibilities in the [fairseq repository](https://github.com/facebookresearch/fairseq), mBART50 translations can only be generated from a specific commit in the repo. Please see [issue #3474](https://github.com/facebookresearch/fairseq/issues/3474) for a link to the commit and also steps on how to install the same and make it work.