function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

set -e

# Variables to change

# Replace with any prefix you want to use for your experiments (purely for naming purposes)
prefix="sample-kd" 

# Replace with the prefixes in the generated translations and alignments
translation_prefix="mBART" 
alignments_prefix="ft"

# Replace with the languages for which you have parallel and mono datasets respectively
parallel_langs=(fi_FI et_EE)
mono_langs=(en_XX fi_FI et_EE)
tot_lines=54000000 # Max corpus size

# More or less fixed
base_dir="~/CAS" # Replace with the base directory of the project 
preprocessing_dir="~/CAS/baselines/knowledge-distillation" 
SCRIPTS_DIR="~/scripts" # Replace with directory containing Moses tokenization (and detokenization) scripts
datasets_dir="~/CAS/data" # Replace with directory containing all datasets

SPM=~/sentencepiece/build/src # Sentencepiece path
FAIRSEQ=~/fairseq/fairseq_cli # Fairseq path

# Derived
data_dir="${base_dir}/data/${prefix}"
checkpoints_dir="${base_dir}/checkpoints/${prefix}"
results_dir="${base_dir}/results/${prefix}"
dict_dir="${data_dir}/dict"
spm_dir="${data_dir}/spm"
spm_corpus="${data_dir}/spm_corpus"
MODEL=${spm_dir}/${prefix}.model

TRAIN_SRC="${data_dir}/train.src"
TRAIN_TGT="${data_dir}/train.tgt"
TRAIN_SRC_DS="${data_dir}/train.downsampled.src"
TRAIN_TGT_DS="${data_dir}/train.downsampled.tgt"

MONO_SRC="${data_dir}/mono/train.src"
MONO_TGT="${data_dir}/mono/train.tgt"
MONO_JOINED="${data_dir}/mono/joined"
MONO_JOINED_SHUF="${data_dir}/mono/joined.shuf"
MONO_SRC_DS="${data_dir}/mono/train.downsampled.src"
MONO_TGT_DT="${data_dir}/mono/train.downsampled.tgt"
DEV_SRC="${data_dir}/dev.src"
DEV_TGT="${data_dir}/dev.tgt"

rm -rf ${data_dir} ${data_dir}/mono ${checkpoints_dir} ${results_dir} ${dict_dir} ${spm_dir} 
mkdir -p ${data_dir} ${data_dir}/mono ${checkpoints_dir} ${results_dir} ${dict_dir} ${spm_dir}

rm -rf ${spm_corpus} ${TRAIN_SRC} ${TRAIN_TGT} ${MONO_SRC} ${MONO_TGT} ${DEV_SRC} ${DEV_TGT}
touch $spm_corpus ${TRAIN_SRC} ${TRAIN_TGT} ${MONO_SRC} ${MONO_TGT} ${DEV_SRC} ${DEV_TGT}

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"

for TGT_LANG in "${train_langs[@]}"
do
    lang="${TGT_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
    TGT_LANG_CAPS="${lang^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    SRC_FILE="${PARALLEL_DIR}/train.${SRC_LANG}"

    lines=($(wc -l < $SRC_FILE))
    langs_length=$((${#mono_langs[@]}-1))
    mono_lines=$((lines/langs_length + 1))

    low=1
    for tgt_lang in "${mono_langs[@]}"
    do
        if [[ $SRC_LANG == $tgt_lang ]]; then
            continue
        fi
        tgt_short=`echo ${tgt_lang} | cut -c 1-2`
        TGT_CAPS="${tgt_short^^}"

        TRANSLATED_FILE="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.${tgt_lang}"
        high=$((low + mono_lines - 1))
        
        sed -n "${low},${high}p" $TRANSLATED_FILE >> ${spm_corpus}
        sed -n "${low},${high}p" $SRC_FILE >> ${spm_corpus}

        low=$((high + 1))
    done
done

TGT_LANG="en_XX"
TGT_LANG_SHORT="${TGT_LANG:0:2}"

for SRC_LANG in "${train_langs[@]}"
do
    lang="${SRC_LANG:0:2}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"
    SRC_LANG_CAPS="${lang^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    SRC_FILE="${PARALLEL_DIR}/train.${SRC_LANG}"

    lines=($(wc -l < $SRC_FILE))
    langs_length=$((${#mono_langs[@]}-1))
    mono_lines=$((lines/langs_length + 1))

    low=1
    for tgt_lang in "${mono_langs[@]}"
    do
        if [[ $SRC_LANG == $tgt_lang ]]; then
            continue
        fi
        tgt_short=`echo ${tgt_lang} | cut -c 1-2`
        TGT_CAPS="${tgt_short^^}"

        TRANSLATED_FILE="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.${tgt_lang}"
        high=$((low + mono_lines - 1))
        
        sed -n "${low},${high}p" $TRANSLATED_FILE >> ${spm_corpus}
        sed -n "${low},${high}p" $SRC_FILE >> ${spm_corpus}

        low=$((high + 1))
    done
done

for SRC_LANG in "${mono_langs[@]}"
do
    SRC_LANG_SHORT="${SRC_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"

    # Variables to change
    MONO_DIR="${datasets_dir}/mono-${SRC_LANG_SHORT}"
    SRC_FILE="${MONO_DIR}/mono.${SRC_LANG}"

    lines=($(wc -l < $SRC_FILE))
    langs_length=$((${#mono_langs[@]}-1))
    mono_lines=$((lines/langs_length + 1))

    low=1
    for tgt_lang in "${mono_langs[@]}"
    do
        if [[ $SRC_LANG == $tgt_lang ]]; then
            continue
        fi
        tgt_short=`echo ${tgt_lang} | cut -c 1-2`
        TGT_CAPS="${tgt_short^^}"

        TRANSLATED_FILE="${MONO_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.${tgt_lang}"
        high=$((low + mono_lines - 1))
        
        sed -n "${low},${high}p" $TRANSLATED_FILE >> ${spm_corpus}
        sed -n "${low},${high}p" $SRC_FILE >> ${spm_corpus}

        low=$((high + 1))
    done
done

shuf $spm_corpus > "${spm_corpus}.shuf"
rm $spm_corpus

echo "Training an spm model"

${SPM}/spm_train --input="${spm_corpus}.shuf" --train_extremely_large_corpus=true --model_prefix="${data_dir}/spm/${prefix}" --vocab_size=32000 --character_coverage=1.0  --model_type=unigram --input_sentence_size=20000000 --shuffle_input_sentence=true

${SPM}/spm_encode --model="${MODEL}" < ${spm_corpus}.shuf > "${spm_corpus}.spm.src"

rm ${spm_corpus}.shuf

python ${FAIRSEQ}/preprocess.py \
        --source-lang src --target-lang src \
        --trainpref "${spm_corpus}.spm" \
        --destdir ${dict_dir} --only-source \
        --thresholdtgt 0 --thresholdsrc 0 \
        --dict-only --joined-dictionary  \
        --workers 128
rm "${spm_corpus}.spm.src"

for lang_large in "${mono_langs[@]}"
do
    lang="${lang_large:0:2}"
    lang_caps="${lang^^}"
    echo -e "LANG_TOK_${lang_caps} 1" >> "${dict_dir}/dict.src.txt"
done

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"
for TGT_LANG in "${train_langs[@]}"
do
    lang="${TGT_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
    TGT_LANG_CAPS="${lang^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    SRC_FILE="${PARALLEL_DIR}/train.${SRC_LANG}"

    lines=($(wc -l < $SRC_FILE))
    langs_length=$((${#mono_langs[@]}-1))
    mono_lines=$((lines/langs_length + 1))

    low=1
    for tgt_lang in "${mono_langs[@]}"
    do
        if [[ $SRC_LANG == $tgt_lang ]]; then
            continue
        fi
        tgt_short=`echo ${tgt_lang} | cut -c 1-2`
        TGT_CAPS="${tgt_short^^}"

        TRANSLATED_FILE="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.${tgt_lang}"
        high=$((low + mono_lines - 1))
        # Diversifying target so that kd baseline learns non-English centric directions as well (empirically found to be stronger than English-centric baselines in all cases, despite potential 'noise')
        sed -n "${low},${high}p" $TRANSLATED_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v SRC=$TGT_CAPS '{print "LANG_TOK_"SRC" " $0 } ' | tail -n +285 >> ${TRAIN_TGT}
        sed -n "${low},${high}p" $SRC_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v TGT=$SRC_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' | tail -n +285 >> ${TRAIN_SRC}

        sed -n "${low},${high}p" $TRANSLATED_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v SRC=$TGT_CAPS '{print "LANG_TOK_"SRC" " $0 } ' | head -n +285 >> ${DEV_TGT}
        sed -n "${low},${high}p" $SRC_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v TGT=$SRC_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' | head -n +285 >> ${DEV_SRC}
        
        low=$((high + 1))
    done
done

TGT_LANG="en_XX"
TGT_LANG_SHORT="${TGT_LANG:0:2}"
for SRC_LANG in "${train_langs[@]}"
do
    lang="${SRC_LANG:0:2}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"
    SRC_LANG_CAPS="${lang^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    SRC_FILE="${PARALLEL_DIR}/train.${SRC_LANG}"

    lines=($(wc -l < $SRC_FILE))
    langs_length=$((${#mono_langs[@]}-1))
    mono_lines=$((lines/langs_length + 1))

    low=1
    for tgt_lang in "${mono_langs[@]}"
    do
        if [[ $SRC_LANG == $tgt_lang ]]; then
            continue
        fi
        tgt_short=`echo ${tgt_lang} | cut -c 1-2`
        TGT_CAPS="${tgt_short^^}"

        TRANSLATED_FILE="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.${tgt_lang}"
        high=$((low + mono_lines - 1))
        # Diversifying target so that kd baseline learns non-English centric directions as well (empirically found to be stronger than English-centric baselines in all cases, despite potential 'noise')
        sed -n "${low},${high}p" $TRANSLATED_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v SRC=$TGT_CAPS '{print "LANG_TOK_"SRC" " $0 } ' | tail -n +285 >> ${TRAIN_TGT}
        sed -n "${low},${high}p" $SRC_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v TGT=$SRC_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' | tail -n +285 >> ${TRAIN_SRC}

        sed -n "${low},${high}p" $TRANSLATED_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v SRC=$TGT_CAPS '{print "LANG_TOK_"SRC" " $0 } ' | head -n +285 >> ${DEV_TGT}
        sed -n "${low},${high}p" $SRC_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v TGT=$SRC_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' | head -n +285 >> ${DEV_SRC}
        
        low=$((high + 1))
    done
done


for SRC_LANG in "${mono_langs[@]}"
do
    SRC_LANG_SHORT="${SRC_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"

    # Variables to change
    MONO_DIR="${datasets_dir}/mono-${SRC_LANG_SHORT}"
    SRC_FILE="${MONO_DIR}/mono.${SRC_LANG}"
    
    lines=($(wc -l < $SRC_FILE))
    langs_length=$((${#mono_langs[@]}-1))
    mono_lines=$((lines/langs_length + 1))

    low=1
    for tgt_lang in "${mono_langs[@]}"
    do
        if [[ $SRC_LANG == $tgt_lang ]]; then
            continue
        fi
        tgt_short=`echo ${tgt_lang} | cut -c 1-2`
        TGT_CAPS="${tgt_short^^}"

        TRANSLATED_FILE="${MONO_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.${tgt_lang}"
        high=$((low + mono_lines - 1))
        
        sed -n "${low},${high}p" $TRANSLATED_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v SRC=$TGT_CAPS '{print "LANG_TOK_"SRC" " $0 } ' | tail -n +285 >> ${TRAIN_TGT}
        sed -n "${low},${high}p" $SRC_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v TGT=$SRC_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } '  | tail -n +285 >> ${TRAIN_SRC}

        sed -n "${low},${high}p" $TRANSLATED_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v SRC=$TGT_CAPS '{print "LANG_TOK_"SRC" " $0 } ' | head -n +285 >> ${DEV_TGT}
        sed -n "${low},${high}p" $SRC_FILE | ${SPM}/spm_encode --model=${MODEL} | awk -v TGT=$SRC_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } '  | head -n +285 >> ${DEV_SRC}

        low=$((high + 1))
    done
done


:|paste -d ' ||| ' ${TRAIN_SRC} - - - - ${TRAIN_TGT}  > ${MONO_JOINED}
shuf -n $tot_lines ${MONO_JOINED} > ${MONO_JOINED_SHUF}
awk -F' \\|\\|\\| ' '{print $1}' ${MONO_JOINED_SHUF} > ${TRAIN_SRC_DS}
awk -F' \\|\\|\\| ' '{print $2}' ${MONO_JOINED_SHUF} > ${TRAIN_TGT_DS}
rm ${MONO_JOINED_SHUF} ${MONO_JOINED}
# cat ${MONO_SRC} >> ${TRAIN_SRC}
# cat ${MONO_TGT} >> ${TRAIN_TGT}

python ${FAIRSEQ}/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref ${data_dir}/train.downsampled \
    --validpref ${data_dir}/dev \
    --srcdict ${dict_dir}/dict.src.txt \
    --tgtdict ${dict_dir}/dict.src.txt \
    --destdir ${data_dir} \
    --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
    --amp --workers 128

# python ${FAIRSEQ}/preprocess.py \
#     --source-lang src --target-lang tgt \
#     --trainpref ${data_dir}/mono/train.downsampled \
#     --srcdict ${dict_dir}/dict.src.txt \
#     --tgtdict ${dict_dir}/dict.src.txt \
#     --destdir ${data_dir}/mono \
#     --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
#     --amp --workers 128

sbatch train.sh
