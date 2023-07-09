all_langs=(et_EE fi_FI en_XX)

TRAIN_PREFIX=train
OUTPUT_PREFIX="multilingual"


SPM_MODEL=~/CAS/data/multilingual-uralic/spm/multilingual-uralic.model
MODEL=~/CAS/checkpoints/multilingual-uralic/checkpoint_best.pt
DICT_SRC=~/CAS/data/multilingual-uralic/dict.src.txt
DICT_TGT=~/CAS/data/multilingual-uralic/dict.tgt.txt
FT_ALIGN_DIR~/CAS/models/alignment/uralic-trained

src_long=en_XX
tgts=(fi_FI et_EE)
for tgt_long in ${tgts[@]};
do
    src="${src_long:0:2}"
    tgt="${tgt_long:0:2}"
    dataset_prefix="/home/v/vivek/mCASP/word-alignment/data/${src}-${tgt}-uralic"

    translate_tgts=()
    for value in "${all_langs[@]}"
    do
        [[ $value != $src_long ]] && translate_tgts+=($value)
    done

    for pivot_tgt in ${translate_tgts[@]};
    do
        sbatch --export=src_long=$src_long,tgt_long=$pivot_tgt,DATASET_DIR=$dataset_prefix,SPM_MODEL=$SPM_MODEL,TRAIN_PREFIX=$TRAIN_PREFIX,MODEL=$MODEL,DICT_SRC=$DICT_SRC,DICT_TGT=$DICT_TGT,OUTPUT_PREFIX=$OUTPUT_PREFIX,FT_ALIGN_DIR=$FT_ALIGN_DIR translate.sh
    done
done

tgt_long=en_XX
srcs=(fi_FI et_EE)
for src_long in ${srcs[@]};
do
    src="${src_long:0:2}"
    tgt="${tgt_long:0:2}"
    dataset_prefix="/home/v/vivek/mCASP/word-alignment/data/${tgt}-${src}-uralic"

    translate_tgts=()
    for value in "${all_langs[@]}"
    do
        [[ $value != $src_long ]] && translate_tgts+=($value)
    done

    for pivot_tgt in ${translate_tgts[@]};
    do
        sbatch --export=src_long=$src_long,tgt_long=$pivot_tgt,DATASET_DIR=$dataset_prefix,SPM_MODEL=$SPM_MODEL,TRAIN_PREFIX=$TRAIN_PREFIX,MODEL=$MODEL,DICT_SRC=$DICT_SRC,DICT_TGT=$DICT_TGT,OUTPUT_PREFIX=$OUTPUT_PREFIX,FT_ALIGN_DIR=$FT_ALIGN_DIR translate.sh
    done
done

TRAIN_PREFIX=mono
for src_long in ${all_langs[@]};
do
    src="${src_long:0:2}"
    dataset_prefix="/home/v/vivek/mCASP/word-alignment/data/mono-${src}"

    translate_tgts=()
    for value in "${all_langs[@]}"
    do
        [[ $value != $src_long ]] && translate_tgts+=($value)
    done
    for pivot_tgt in ${translate_tgts[@]};
    do
        sbatch --export=src_long=$src_long,tgt_long=$pivot_tgt,DATASET_DIR=$dataset_prefix,SPM_MODEL=$SPM_MODEL,TRAIN_PREFIX=$TRAIN_PREFIX,MODEL=$MODEL,DICT_SRC=$DICT_SRC,DICT_TGT=$DICT_TGT,OUTPUT_PREFIX=$OUTPUT_PREFIX,FT_ALIGN_DIR=$FT_ALIGN_DIR translate.sh
    done
done