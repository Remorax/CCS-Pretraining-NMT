function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

set -e

# Variables to change

# Replace with any prefix you want to use for your experiments (purely for naming purposes)
prefix="sample-ft" 

# Replace with the prefixes in the generated translations and alignments
translation_prefix="mBART" 
alignments_prefix="ft"

# Replace with the languages for which you have parallel and mono datasets respectively
parallel_langs=(fi_FI et_EE)
mono_langs=(en_XX fi_FI et_EE)

# Replacement ratio
CAS_ratio=0.75

# Paths
base_dir="~/CAS" # Replace with the base directory of the project 
SCRIPTS_DIR="~/scripts" # Replace with directory containing Moses tokenization (and detokenization) scripts
datasets_dir="~/CAS/data" # Replace with directory containing all datasets

# More or less fixed
SPM=~/sentencepiece/build/src # Sentencepiece path
FAIRSEQ=~/fairseq/fairseq_cli # Fairseq path

# Folders to create during preprocessing
data_dir="${base_dir}/data/${prefix}"
checkpoints_dir="${base_dir}/checkpoints/${prefix}"
results_dir="${base_dir}/results/${prefix}"
dict_dir="${data_dir}/dict"

rm -rf ${data_dir} ${checkpoints_dir} ${results_dir} ${dict_dir}
mkdir -p ${data_dir} ${checkpoints_dir} ${results_dir} ${dict_dir}

echo "CASing En file for En->X direction..."
SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"

for TGT_LANG in "${parallel_langs[@]}"
do
    lang="${TGT_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
    TGT_LANG_CAPS="${lang^^}"

    # Replace with paths to parallel datasets (in our case we stored datasets in directories called en-fi-uralic, en-et-uralic etc.)
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    # Source file (must be the tokenized file used for computing Moses word alignments)
    SRC_FILE="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${TGT_LANG}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.${SRC_LANG}"
    # Reference file
    REF_FILE="${PARALLEL_DIR}/train.${TGT_LANG}"
    PART_PREFIX="${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${lang}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.part"
    OUTPUT_PREFIX="${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${lang}/CAS.final.tok"

    echo "CASing for the En->${lang} direction..."
    rm -rf "${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${lang}"
    mkdir -p "${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${lang}"

    line_count=($((`wc -l $SRC_FILE | cut -d " " -f1`/128)))
    split -a 3 -d -l $line_count --additional-suffix=.${SRC_LANG} $SRC_FILE ${PART_PREFIX}
    split -a 3 -d -l $line_count --additional-suffix=.${TGT_LANG} $REF_FILE ${PART_PREFIX}

    for tgt_lang in "${mono_langs[@]}"
    do
        if [[ $SRC_LANG == $tgt_lang ]]; then
            continue
        fi
        tgt_short=`echo ${tgt_lang} | cut -c 1-2`
        
        translated_prefix="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok"
        translated_prefix_part="${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/translated.tok.part"
        alignment_prefix="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.alignments"
        alignment_prefix_part="${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/translated.tok.alignments.part"
        mkdir -p "${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/"

        split -a 3 -d -l $line_count --additional-suffix=.${tgt_lang} "${translated_prefix}.${tgt_lang}" "${translated_prefix_part}"
        split -a 3 -d -l $line_count $alignment_prefix "${alignment_prefix_part}"
    done
    # Uses multiprocessing to preprocess large datasets faster (replace 128 with number of CPU cores)
    for i in `seq 0 128`; do
        if [[ $i -le 9 ]]; then
            i="00${i}";
        elif  [[ $i -le 99 ]]; then
            i="0${i}";
        fi
        if [[ ! -f "${PART_PREFIX}${i}.${SRC_LANG}" ]]; then
            if [[ $i == 128 ]]; then
                continue;
            else
                echo "[ERROR] No file named ${PART_PREFIX}${i}.${SRC_LANG}"
            fi
        fi
        translated_files=""
        alignment_files=""
        for tgt_lang in "${mono_langs[@]}"
        do
            if [[ $SRC_LANG == $tgt_lang ]]; then
                continue
            fi
            tgt_short=`echo ${tgt_lang} | cut -c 1-2`
            translated_files="${translated_files},${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/translated.tok.part${i}.${tgt_lang}"
            alignment_files="${alignment_files},${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/translated.tok.alignments.part${i}"
        done
        python ${base_dir}/cas.py \
            --source_file ${PART_PREFIX}${i}.${SRC_LANG} \
            --reference_file ${PART_PREFIX}${i}.${TGT_LANG} \
            --translation_files ${translated_files:1} \
            --alignment_files ${alignment_files:1} \
            --output_prefix ${OUTPUT_PREFIX}.part${i} \
            --replacement_ratio ${CAS_ratio} \
            --blcs &
    done
    wait
    rm -f ${OUTPUT_PREFIX}.tok.${SRC_LANG} ${OUTPUT_PREFIX}.${TGT_LANG}
    touch ${OUTPUT_PREFIX}.tok.${SRC_LANG} ${OUTPUT_PREFIX}.${TGT_LANG}
    for i in `seq 0 128`; do
        if [[ $i -le 9 ]]; then
            i="00${i}";
        elif  [[ $i -le 99 ]]; then
            i="0${i}";
        fi
        if [[ ! -f ${PART_PREFIX}${i}.${SRC_LANG} ]]; then
            if [[ $i == 128 ]]; then
                continue;
            else
                echo "[ERROR] No file named ${PART_PREFIX}${i}.${SRC_LANG}"
            fi
        fi
        cat ${OUTPUT_PREFIX}.part${i}.${SRC_LANG} >>  ${OUTPUT_PREFIX}.tok.${SRC_LANG}
        cat ${OUTPUT_PREFIX}.part${i}.${TGT_LANG} >>  ${OUTPUT_PREFIX}.${TGT_LANG}
        rm ${OUTPUT_PREFIX}.part${i}.${SRC_LANG} ${OUTPUT_PREFIX}.part${i}.${TGT_LANG} ${PART_PREFIX}${i}.${SRC_LANG} ${PART_PREFIX}${i}.${TGT_LANG}
        for tgt_lang in "${mono_langs[@]}"
        do
            if [[ $SRC_LANG == $tgt_lang ]]; then
                continue
            fi
            tgt_short=`echo ${tgt_lang} | cut -c 1-2`
            translated_prefix="${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/translated.tok"
            alignment_prefix="${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/translated.tok.alignments"
            rm "${translated_prefix}.part${i}.${tgt_lang}" "${alignment_prefix}.part${i}"
        done
    done
    echo "Done"

    perl ${SCRIPTS_DIR}/moses_detokenizer.perl -l ${SRC_LANG_SHORT} < ${OUTPUT_PREFIX}.tok.${SRC_LANG} > ${OUTPUT_PREFIX}.${SRC_LANG}
    rm ${OUTPUT_PREFIX}.tok.${SRC_LANG}
done

TGT_LANG="en_XX"
TGT_LANG_SHORT="${TGT_LANG:0:2}"
echo "CASing X file for X->En direction..."

for SRC_LANG in "${parallel_langs[@]}"
do
    lang="${SRC_LANG:0:2}"
    SRC_LANG_CAPS="${lang^^}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    SRC_FILE="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${TGT_LANG}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.${SRC_LANG}"
    REF_FILE="${PARALLEL_DIR}/train.${TGT_LANG}"
    PART_PREFIX="${PARALLEL_DIR}/CAS/${prefix}/${lang}-${TGT_LANG_SHORT}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.part"
    OUTPUT_PREFIX="${PARALLEL_DIR}/CAS/${prefix}/${lang}-${TGT_LANG_SHORT}/CAS.final.tok"

    echo "CASing for the ${lang}->En direction..."
    rm -rf "${PARALLEL_DIR}/CAS/${prefix}/${lang}-${TGT_LANG_SHORT}"
    mkdir -p "${PARALLEL_DIR}/CAS/${prefix}/${lang}-${TGT_LANG_SHORT}"

    line_count=($((`wc -l $SRC_FILE | cut -d " " -f1`/128)))
    split -a 3 -d -l $line_count --additional-suffix=.${SRC_LANG} $SRC_FILE ${PART_PREFIX}
    split -a 3 -d -l $line_count --additional-suffix=.${TGT_LANG} $REF_FILE ${PART_PREFIX}

    for tgt_lang in "${mono_langs[@]}"
    do
        if [[ $SRC_LANG == $tgt_lang ]]; then
            continue
        fi
        tgt_short=`echo ${tgt_lang} | cut -c 1-2`
        translated_prefix="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok"
        translated_prefix_part="${PARALLEL_DIR}/CAS/${prefix}/${lang}-${tgt_short}/translated.tok.part"
        alignment_prefix="${PARALLEL_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.alignments"
        alignment_prefix_part="${PARALLEL_DIR}/CAS/${prefix}/${lang}-${tgt_short}/translated.tok.alignments.part"

        mkdir -p ${PARALLEL_DIR}/CAS/${prefix}/${lang}-${tgt_short}/
        split -a 3 -d -l $line_count --additional-suffix=.${tgt_lang} "${translated_prefix}.${tgt_lang}" "${translated_prefix_part}"
        split -a 3 -d -l $line_count $alignment_prefix "${alignment_prefix_part}"
    done

    for i in `seq 0 128`; do
        if [[ $i -le 9 ]]; then
            i="00${i}";
        elif  [[ $i -le 99 ]]; then
            i="0${i}";
        fi
        if [[ ! -f "${PART_PREFIX}${i}.${SRC_LANG}" ]]; then
            if [[ $i == 128 ]]; then
                continue;
            else
                echo "[ERROR] No file named ${PART_PREFIX}${i}.${SRC_LANG}"
            fi
        fi
        translated_files=""
        alignment_files=""
        for tgt_lang in "${mono_langs[@]}"
        do
            if [[ $SRC_LANG == $tgt_lang ]]; then
                continue
            fi
            tgt_short=`echo ${tgt_lang} | cut -c 1-2`
            translated_files="${translated_files},${PARALLEL_DIR}/CAS/${prefix}/${lang}-${tgt_short}/translated.tok.part${i}.${tgt_lang}"
            alignment_files="${alignment_files},${PARALLEL_DIR}/CAS/${prefix}/${lang}-${tgt_short}/translated.tok.alignments.part${i}"
        done
        python ${base_dir}/cas.py \
            --source_file ${PART_PREFIX}${i}.${SRC_LANG} \
            --reference_file ${PART_PREFIX}${i}.${TGT_LANG} \
            --translation_files ${translated_files:1} \
            --alignment_files ${alignment_files:1} \
            --output_prefix ${OUTPUT_PREFIX}.part${i} \
            --replacement_ratio ${CAS_ratio} \
            --blcs &
    done
    wait
    rm -f ${OUTPUT_PREFIX}.tok.${SRC_LANG} ${OUTPUT_PREFIX}.${TGT_LANG}
    touch ${OUTPUT_PREFIX}.tok.${SRC_LANG} ${OUTPUT_PREFIX}.${TGT_LANG}
    for i in `seq 0 128`; do
        if [[ $i -le 9 ]]; then
            i="00${i}";
        elif  [[ $i -le 99 ]]; then
            i="0${i}";
        fi
        if [[ ! -f ${PART_PREFIX}${i}.${SRC_LANG} ]]; then
            if [[ $i == 128 ]]; then
                continue;
            else
                echo "[ERROR] No file named ${PART_PREFIX}${i}.${SRC_LANG}"
            fi
        fi
        cat ${OUTPUT_PREFIX}.part${i}.${SRC_LANG} >>  ${OUTPUT_PREFIX}.tok.${SRC_LANG}
        cat ${OUTPUT_PREFIX}.part${i}.${TGT_LANG} >>  ${OUTPUT_PREFIX}.${TGT_LANG}
        rm ${OUTPUT_PREFIX}.part${i}.${SRC_LANG} ${OUTPUT_PREFIX}.part${i}.${TGT_LANG} ${PART_PREFIX}${i}.${SRC_LANG} ${PART_PREFIX}${i}.${TGT_LANG}
        for tgt_lang in "${mono_langs[@]}"
        do
            if [[ $SRC_LANG == $tgt_lang ]]; then
                continue
            fi
            tgt_short=`echo ${tgt_lang} | cut -c 1-2`
            translated_prefix="${PARALLEL_DIR}/CAS/${prefix}/${lang}-${tgt_short}/translated.tok"
            alignment_prefix="${PARALLEL_DIR}/CAS/${prefix}/${lang}-${tgt_short}/translated.tok.alignments"
            rm "${translated_prefix}.part${i}.${tgt_lang}" "${alignment_prefix}.part${i}"
        done
    done
    echo "Done"

    perl ${SCRIPTS_DIR}/moses_detokenizer.perl -l ${lang} < ${OUTPUT_PREFIX}.tok.${SRC_LANG} > ${OUTPUT_PREFIX}.${SRC_LANG}
    rm ${OUTPUT_PREFIX}.tok.${SRC_LANG}
done


for SRC_LANG in "${mono_langs[@]}"
do
    SRC_LANG_SHORT="${SRC_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"

    # Replace with paths to monolingual datasets (in our case we stored datasets in dirs called mono-en, mono-fr etc)
    MONO_DIR="${datasets_dir}/mono-${SRC_LANG_SHORT}"

    OUTPUT_PREFIX="${MONO_DIR}/CAS/${prefix}/mono.CAS.tok"

    for tgt_lang in "${mono_langs[@]}"
    do
        if [[ $SRC_LANG == $tgt_lang ]]; then
            continue
        fi
        tgt_short=`echo ${tgt_lang} | cut -c 1-2`

        PART_PREFIX="${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.train.tok.part"
        SRC_FILE="${MONO_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_${translation_prefix}.detok.${alignments_prefix}.tok.${SRC_LANG}"
        REF_FILE="${MONO_DIR}/mono.${SRC_LANG}"

        echo "CASing ${SRC_LANG_SHORT} mono corpora in the ${SRC_LANG_SHORT}->${tgt_short} direction..."
        rm -rf "${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/"
        mkdir -p "${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/"

        line_count=($((`wc -l $SRC_FILE | cut -d " " -f1`/128)))
        split -a 3 -d -l $line_count --additional-suffix=.mono${SRC_LANG_SHORT}noised $SRC_FILE ${PART_PREFIX}
        split -a 3 -d -l $line_count --additional-suffix=.mono${SRC_LANG_SHORT}target $REF_FILE ${PART_PREFIX}

        translated_prefix="${MONO_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_mBART.detok.${alignments_prefix}.tok"
        translated_prefix_part="${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.translated_mBART.tok.part"
        alignment_prefix="${MONO_DIR}/preprocessed/cas/${SRC_LANG}-${tgt_lang}_${translation_prefix}/translated_mBART.detok.${alignments_prefix}.alignments"
        alignment_prefix_part="${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.translated_mBART.alignments.part"

        mkdir -p ${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/
        split -a 3 -d -l $line_count --additional-suffix=.${tgt_lang} "${translated_prefix}.${tgt_lang}" "${translated_prefix_part}"
        split -a 3 -d -l $line_count $alignment_prefix "${alignment_prefix_part}"
    done

    for i in `seq 0 128`; do
        if [[ $i -le 9 ]]; then
            i="00${i}";
        elif  [[ $i -le 99 ]]; then
            i="0${i}";
        fi

        translated_files=""
        alignment_files=""
        no128=false
        for tgt_lang in "${mono_langs[@]}"
        do
            if [[ $SRC_LANG == $tgt_lang ]]; then
                continue
            fi
            tgt_short=`echo ${tgt_lang} | cut -c 1-2`
            PART_PREFIX="${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.train.tok.part"

            if [[ ! -f "${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}noised" ]]; then
                if [[ $i == 128 ]]; then
                    no128=true
                    continue;
                else
                    no128=false
                    echo "[ERROR] No file named ${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}noised"
                fi
            fi
            translated_files="${translated_files},${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.translated_mBART.tok.part${i}.${tgt_lang}"
            alignment_files="${alignment_files},${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.translated_mBART.alignments.part${i}"
        done
        echo "Translated files: ${translated_files}"
        echo "Alignment files: ${alignment_files}"
        echo "Source file (mono) is ${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}noised"
        echo "Source file (mono) is ${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}target"
        if [ $no128 = true ]; then continue; fi
        no128=false
        python ${base_dir}/cas.py \
            --source_file ${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}noised \
            --reference_file ${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}target \
            --translation_files ${translated_files:1} \
            --alignment_files ${alignment_files:1} \
            --output_prefix ${OUTPUT_PREFIX}.part${i} \
            --replacement_ratio ${CAS_ratio} &
    done
    wait
    rm -f ${OUTPUT_PREFIX}.tok.mono${SRC_LANG_SHORT}noised ${OUTPUT_PREFIX}.mono${SRC_LANG_SHORT}target
    touch ${OUTPUT_PREFIX}.tok.mono${SRC_LANG_SHORT}noised ${OUTPUT_PREFIX}.mono${SRC_LANG_SHORT}target
    for i in `seq 0 128`; do
        if [[ $i -le 9 ]]; then
            i="00${i}";
        elif  [[ $i -le 99 ]]; then
            i="0${i}";
        fi
        if [[ ($i == 128) && (! -f ${OUTPUT_PREFIX}.part${i}.mono${SRC_LANG_SHORT}noised) ]]; then
            continue
        fi
        cat ${OUTPUT_PREFIX}.part${i}.mono${SRC_LANG_SHORT}noised >>  ${OUTPUT_PREFIX}.tok.mono${SRC_LANG_SHORT}noised
        cat ${OUTPUT_PREFIX}.part${i}.mono${SRC_LANG_SHORT}target >>  ${OUTPUT_PREFIX}.mono${SRC_LANG_SHORT}target
        rm ${OUTPUT_PREFIX}.part${i}.mono${SRC_LANG_SHORT}noised ${OUTPUT_PREFIX}.part${i}.mono${SRC_LANG_SHORT}target

        for tgt_lang in "${mono_langs[@]}"
        do
            if [[ $SRC_LANG == $tgt_lang ]]; then
                continue
            fi
            tgt_short=`echo ${tgt_lang} | cut -c 1-2`
            PART_PREFIX="${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.train.tok.part"

            if [[ ! -f "${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}noised" ]]; then
                if [[ $i == 128 ]]; then
                    continue;
                else
                    echo "[ERROR] No file named ${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}noised"
                fi
            fi
            translated_files="${translated_files},${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.translated_mBART.tok.part${i}.${tgt_lang}"
            alignment_files="${alignment_files},${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.translated_mBART.alignments.part${i}"
            rm "${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.translated_mBART.tok.part${i}.${tgt_lang}" "${MONO_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${tgt_short}/mono.translated_mBART.alignments.part${i}" ${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}noised ${PART_PREFIX}${i}.mono${SRC_LANG_SHORT}target
        done
    done
    echo "Done"
    perl ${SCRIPTS_DIR}/moses_detokenizer.perl -l ${SRC_LANG_SHORT} < ${OUTPUT_PREFIX}.tok.mono${SRC_LANG_SHORT}noised > ${OUTPUT_PREFIX}.mono${SRC_LANG_SHORT}noised
    rm ${OUTPUT_PREFIX}.tok.mono${SRC_LANG_SHORT}noised
done

spm_dir="${data_dir}/spm"
spm_corpus="${data_dir}/spm_corpus"
dict_corpus="${dict_dir}/dict_corpus"
MODEL=${spm_dir}/${prefix}.model

TRAIN_SRC="${data_dir}/train.src"
TRAIN_TGT="${data_dir}/train.tgt"
MONO_SRC="${data_dir}/mono/train.src"
MONO_TGT="${data_dir}/mono/train.tgt"
DEV_SRC="${data_dir}/dev.src"
DEV_TGT="${data_dir}/dev.tgt"

rm -rf ${spm_dir} ${data_dir}/mono/
mkdir -p ${spm_dir} ${data_dir}/mono/

rm -rf ${spm_corpus} ${dict_corpus} ${TRAIN_SRC} ${TRAIN_TGT} ${MONO_SRC} ${MONO_TGT} ${DEV_SRC} ${DEV_TGT}
touch $spm_corpus  ${dict_corpus} ${TRAIN_SRC} ${TRAIN_TGT} ${MONO_SRC} ${MONO_TGT} ${DEV_SRC} ${DEV_TGT}
SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"

for TGT_LANG in "${parallel_langs[@]}"
do
    lang="${TGT_LANG:0:2}"
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    OUTPUT_PREFIX="${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${lang}/CAS.final.tok"

    cat ${OUTPUT_PREFIX}.${SRC_LANG} ${OUTPUT_PREFIX}.${TGT_LANG} >> $spm_corpus
    cat ${OUTPUT_PREFIX}.${SRC_LANG} ${OUTPUT_PREFIX}.${TGT_LANG} >> $dict_corpus
done

TGT_LANG="en_XX"
TGT_LANG_SHORT="${TGT_LANG:0:2}"

for SRC_LANG in "${parallel_langs[@]}"
do
    lang="${SRC_LANG:0:2}"
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    OUTPUT_PREFIX="${PARALLEL_DIR}/CAS/${prefix}/${lang}-${TGT_LANG_SHORT}/CAS.final.tok"

    cat ${OUTPUT_PREFIX}.${SRC_LANG} ${OUTPUT_PREFIX}.${TGT_LANG} >> $spm_corpus
    cat ${OUTPUT_PREFIX}.${SRC_LANG} ${OUTPUT_PREFIX}.${TGT_LANG} >> $dict_corpus
done

for SRC_LANG in "${mono_langs[@]}"
do
    SRC_LANG_SHORT="${SRC_LANG:0:2}"
    MONO_DIR="${datasets_dir}/mono-${SRC_LANG_SHORT}"
    OUTPUT_PREFIX="${MONO_DIR}/CAS/${prefix}/mono.CAS.tok"

    cat ${OUTPUT_PREFIX}.mono${SRC_LANG_SHORT}target >> $spm_corpus
    cat ${OUTPUT_PREFIX}.mono${SRC_LANG_SHORT}noised ${OUTPUT_PREFIX}.mono${SRC_LANG_SHORT}target >> $dict_corpus
done

shuf $spm_corpus > "${spm_corpus}.shuf"
rm $spm_corpus

echo "Training an spm model"

${SPM}/spm_train --input="${spm_corpus}.shuf" --train_extremely_large_corpus=true --model_prefix="${data_dir}/spm/${prefix}" --vocab_size=32000 --character_coverage=1.0  --model_type=unigram --input_sentence_size=20000000 --shuffle_input_sentence=true

${SPM}/spm_encode --model="${MODEL}" < ${dict_corpus} > "${dict_corpus}.spm.src"

rm ${spm_corpus}.shuf

python ${FAIRSEQ}/preprocess.py \
        --source-lang src --target-lang src \
        --trainpref "${dict_corpus}.spm" \
        --destdir ${dict_dir} --only-source \
        --thresholdtgt 0 --thresholdsrc 0 \
        --dict-only --joined-dictionary  \
        --workers 128
rm "${dict_corpus}.spm.src"

for lang_large in "${mono_langs[@]}"
do
    lang="${lang_large:0:2}"
    lang_caps="${lang^^}"
    echo -e "LANG_TOK_${lang_caps} 1" >> "${dict_dir}/dict.src.txt"
done

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"
SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
for TGT_LANG in "${parallel_langs[@]}"
do
    TGT_LANG_SHORT="${TGT_LANG:0:2}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"
    PARALLEL_DIR="${datasets_dir}/en-${TGT_LANG_SHORT}-uralic"

    TRAIN_PREFIX="${PARALLEL_DIR}/CAS/${prefix}/${SRC_LANG_SHORT}-${TGT_LANG_SHORT}/CAS.final.tok"
    TRAIN_PREFIX_REV="${PARALLEL_DIR}/CAS/${prefix}/${TGT_LANG_SHORT}-${SRC_LANG_SHORT}/CAS.final.tok"
    DEV_PREFIX="${PARALLEL_DIR}/dev"
    DEV_FINAL="${PARALLEL_DIR}/preprocessed/${prefix}/dev"
    mkdir -p ${PARALLEL_DIR}/preprocessed/${prefix}

    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX}.${SRC_LANG} > ${TRAIN_PREFIX}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX}.${TGT_LANG} > ${TRAIN_PREFIX}.spm.${TGT_LANG_SHORT}

    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX_REV}.${SRC_LANG} > ${TRAIN_PREFIX_REV}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX_REV}.${TGT_LANG} > ${TRAIN_PREFIX_REV}.spm.${TGT_LANG_SHORT}

    ${SPM}/spm_encode --model=${MODEL} < ${DEV_PREFIX}.${SRC_LANG} > ${DEV_FINAL}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${MODEL} < ${DEV_PREFIX}.${TGT_LANG} > ${DEV_FINAL}.spm.${TGT_LANG_SHORT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TRAIN_PREFIX}.spm.${SRC_LANG_SHORT} >> ${TRAIN_SRC}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${TRAIN_PREFIX}.spm.${TGT_LANG_SHORT} >> ${TRAIN_TGT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TRAIN_PREFIX_REV}.spm.${SRC_LANG_SHORT} >> ${TRAIN_TGT}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${TRAIN_PREFIX_REV}.spm.${TGT_LANG_SHORT} >> ${TRAIN_SRC}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${DEV_FINAL}.spm.${SRC_LANG_SHORT} >> ${DEV_SRC}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${DEV_FINAL}.spm.${TGT_LANG_SHORT} >> ${DEV_TGT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${DEV_FINAL}.spm.${SRC_LANG_SHORT} >> ${DEV_TGT}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${DEV_FINAL}.spm.${TGT_LANG_SHORT} >> ${DEV_SRC}

done

python ${FAIRSEQ}/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref ${data_dir}/train \
    --validpref ${data_dir}/dev \
    --srcdict ${dict_dir}/dict.src.txt \
    --tgtdict ${dict_dir}/dict.src.txt \
    --destdir ${data_dir} \
    --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
    --amp --workers 128

for SRC_LANG_LONG in "${mono_langs[@]}"
do
    SRC_LANG="${SRC_LANG_LONG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG^^}"

    MONO_DIR="${datasets_dir}/mono-${SRC_LANG}"
    TRAIN_PREFIX="${MONO_DIR}/CAS/${prefix}/mono.CAS.tok"

    ${SPM}/spm_encode --model=${MODEL} < "${TRAIN_PREFIX}.mono${SRC_LANG}noised" > "${TRAIN_PREFIX}.spm.mono${SRC_LANG}noised"
    ${SPM}/spm_encode --model=${MODEL} < "${TRAIN_PREFIX}.mono${SRC_LANG}target" > "${TRAIN_PREFIX}.spm.mono${SRC_LANG}target"

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' "${TRAIN_PREFIX}.spm.mono${SRC_LANG}noised" >> "${MONO_SRC}"
    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' "${TRAIN_PREFIX}.spm.mono${SRC_LANG}target" >> "${MONO_TGT}"

done


# :|paste -d ' ||| ' "${data_dir}/mono/train.src" - - - - "${data_dir}/mono/train.tgt" > "${data_dir}/mono/train"
# shuf "${data_dir}/mono/train" > "${data_dir}/mono/train.shuf"
# awk -F ' \\|\\|\\| ' '{print $1}' "${data_dir}/mono/train.shuf" > "${data_dir}/mono/train.shuf.src"
# awk -F ' \\|\\|\\| ' '{print $2}' "${data_dir}/mono/train.shuf" > "${data_dir}/mono/train.shuf.tgt"

python ${FAIRSEQ}/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref "${data_dir}/mono/train" \
    --srcdict ${dict_dir}/dict.src.txt \
    --tgtdict ${dict_dir}/dict.src.txt \
    --destdir ${data_dir}/mono \
    --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
    --amp --workers 128

cp ${data_dir}/valid* ${data_dir}/mono/
sbatch pretrain.sh