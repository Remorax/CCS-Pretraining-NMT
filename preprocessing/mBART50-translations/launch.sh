SPM_MODEL=~/pretrained_models/mbart50.ft.1n/sentencepiece.bpe.model
MODEL=~/pretrained_models/mbart50.ft.1n/model.pt
DICT_SRC=~/pretrained_models/mbart50.ft.1n/dict.txt
DICT_TGT=~/pretrained_models/mbart50.ft.1n/dict.txt
lang_list=~/pretrained_models/mbart50.ft.1n/langs.txt
all_langs=(et_EE fi_FI en_XX)
FT_ALIGN_DIR="~/CCS/sample"

TRAIN_PREFIX=train
OUTPUT_PREFIX="mBART"
src_long=en_XX
tgts=(fi_FI et_EE)

tgt_long=fi_FI
src="${src_long:0:2}"
tgt="${tgt_long:0:2}"

dataset_prefix="~/CCS/data/${src}-${tgt}-uralic"
bash translate.sh $src_long $tgt_long $dataset_prefix $SPM_MODEL $TRAIN_PREFIX $MODEL $DICT_SRC $DICT_TGT $OUTPUT_PREFIX

for tgt_long in ${tgts[@]};
do
    src="${src_long:0:2}"
    tgt="${tgt_long:0:2}"
    dataset_prefix="~/CCS/data/${src}-${tgt}-uralic"

    translate_tgts=()
    for value in "${all_langs[@]}"
    do
        [[ $value != $src_long ]] && translate_tgts+=($value)
    done

    for pivot_tgt in ${translate_tgts[@]};
    do
        sbatch --export=src_long=$src_long,tgt_long=$pivot_tgt,DATASET_DIR=$dataset_prefix,SPM_MODEL=$SPM_MODEL,TRAIN_PREFIX=$TRAIN_PREFIX,MODEL=$MODEL,DICT_SRC=$DICT_SRC,DICT_TGT=$DICT_TGT,OUTPUT_PREFIX=$OUTPUT_PREFIX,lang_list=$lang_list,FT_ALIGN_DIR=$FT_ALIGN_DIR translate.sh
    done
done

SPM_MODEL=~/pretrained_models/mbart50.ft.nn/sentencepiece.bpe.model
MODEL=~/pretrained_models/mbart50.ft.nn/model.pt
DICT_SRC=~/pretrained_models/mbart50.ft.nn/dict.txt
DICT_TGT=~/pretrained_models/mbart50.ft.nn/dict.txt
lang_list=~/pretrained_models/mbart50.ft.1n/langs.txt

tgt_long=en_XX
srcs=(fi_FI et_EE)

for src_long in ${srcs[@]};
do
    src="${src_long:0:2}"
    tgt="${tgt_long:0:2}"
    dataset_prefix="~/CCS/data/${tgt}-${src}-uralic"

    translate_tgts=()
    for value in "${all_langs[@]}"
    do
        [[ $value != $src_long ]] && translate_tgts+=($value)
    done

    for pivot_tgt in ${translate_tgts[@]};
    do
        if [[ $src == "en" ]]
        then
            SPM_MODEL=~/pretrained_models/mbart50.ft.1n/sentencepiece.bpe.model
            MODEL=~/pretrained_models/mbart50.ft.1n/model.pt
            DICT_SRC=~/pretrained_models/mbart50.ft.1n/dict.txt
            DICT_TGT=~/pretrained_models/mbart50.ft.1n/dict.txt
            lang_list=~/pretrained_models/mbart50.ft.1n/langs.txt
        elif [[ $pivot_tgt == "en_XX" ]]
        then
            SPM_MODEL=~/pretrained_models/mbart50.ft.n1/sentencepiece.bpe.model
            MODEL=~/pretrained_models/mbart50.ft.n1/model.pt
            DICT_SRC=~/pretrained_models/mbart50.ft.n1/dict.txt
            DICT_TGT=~/pretrained_models/mbart50.ft.n1/dict.txt
            lang_list=~/pretrained_models/mbart50.ft.n1/langs.txt
        else
            SPM_MODEL=~/pretrained_models/mbart50.ft.nn/sentencepiece.bpe.model
            MODEL=~/pretrained_models/mbart50.ft.nn/model.pt
            DICT_SRC=~/pretrained_models/mbart50.ft.nn/dict.txt
            DICT_TGT=~/pretrained_models/mbart50.ft.nn/dict.txt
            lang_list=~/pretrained_models/mbart50.ft.1n/langs.txt
        fi
        sbatch --export=src_long=$src_long,tgt_long=$pivot_tgt,DATASET_DIR=$dataset_prefix,SPM_MODEL=$SPM_MODEL,TRAIN_PREFIX=$TRAIN_PREFIX,MODEL=$MODEL,DICT_SRC=$DICT_SRC,DICT_TGT=$DICT_TGT,OUTPUT_PREFIX=$OUTPUT_PREFIX,lang_list=$lang_list,FT_ALIGN_DIR=$FT_ALIGN_DIR translate.sh
    done
done

SPM_MODEL=~/pretrained_models/mbart50.ft.nn/sentencepiece.bpe.model
MODEL=~/pretrained_models/mbart50.ft.nn/model.pt
DICT_SRC=~/pretrained_models/mbart50.ft.nn/dict.txt
DICT_TGT=~/pretrained_models/mbart50.ft.nn/dict.txt
lang_list=~/pretrained_models/mbart50.ft.1n/langs.txt
TRAIN_PREFIX=mono
# all_langs=(et_EE fi_FI)
tmps=(en_XX)
for src_long in ${tmps[@]};
do
    # echo $src_long
    src="${src_long:0:2}"
    dataset_prefix="~/CCS/data/mono-${src}"

    translate_tgts=()
    for value in "${all_langs[@]}"
    do
        [[ $value != $src_long ]] && translate_tgts+=($value)
    done
    # translate_tgts=(et_EE)
    for pivot_tgt in ${translate_tgts[@]};
    do
        echo $pivot_tgt
        # continue

        if [[ $src == "en" ]]
        then
            SPM_MODEL=~/pretrained_models/mbart50.ft.1n/sentencepiece.bpe.model
            MODEL=~/pretrained_models/mbart50.ft.1n/model.pt
            DICT_SRC=~/pretrained_models/mbart50.ft.1n/dict.txt
            DICT_TGT=~/pretrained_models/mbart50.ft.1n/dict.txt
            lang_list=~/pretrained_models/mbart50.ft.1n/langs.txt
        elif [[ $pivot_tgt == "en_XX" ]]
        then
            SPM_MODEL=~/pretrained_models/mbart50.ft.n1/sentencepiece.bpe.model
            MODEL=~/pretrained_models/mbart50.ft.n1/model.pt
            DICT_SRC=~/pretrained_models/mbart50.ft.n1/dict.txt
            DICT_TGT=~/pretrained_models/mbart50.ft.n1/dict.txt
            lang_list=~/pretrained_models/mbart50.ft.n1/langs.txt
        else
            SPM_MODEL=~/pretrained_models/mbart50.ft.nn/sentencepiece.bpe.model
            MODEL=~/pretrained_models/mbart50.ft.nn/model.pt
            DICT_SRC=~/pretrained_models/mbart50.ft.nn/dict.txt
            DICT_TGT=~/pretrained_models/mbart50.ft.nn/dict.txt
            lang_list=~/pretrained_models/mbart50.ft.1n/langs.txt
        fi
        sbatch --export=src_long=$src_long,tgt_long=$pivot_tgt,DATASET_DIR=$dataset_prefix,SPM_MODEL=$SPM_MODEL,TRAIN_PREFIX=$TRAIN_PREFIX,MODEL=$MODEL,DICT_SRC=$DICT_SRC,DICT_TGT=$DICT_TGT,OUTPUT_PREFIX=$OUTPUT_PREFIX,lang_list=$lang_list,FT_ALIGN_DIR=$FT_ALIGN_DIR translate.sh
    done
done
