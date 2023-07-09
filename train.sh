curr_dir=`pwd`
ulimit -n 6000
set -e

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

# Variables to change
prefix="sample"
test_langs=(fi_FI et_EE)

# More or less fixed
base_dir="~/CAS" # Replace with the base directory of the project 
datasets_dir="~/CAS/data" # Replace with directory containing all datasets
SPM=~/sentencepiece/build/src # Sentencepiece path
FAIRSEQ=~/fairseq/fairseq_cli # Fairseq path
training_config="${base_dir}/configs/cas.yml"

# Derived
data_dir="${base_dir}/data/${prefix}"
checkpoints_dir="${base_dir}/checkpoints/${prefix}"
dict_dir="${data_dir}/dict"
results_dir="${base_dir}/results/${prefix}"
spm_dir="${data_dir}/spm"

if [[ -f "${results_dir}/testing.done" ]];
then
    exit 1;
fi

source ${base_dir}/scripts/load_config.sh ${training_config} ${base_dir}

mkdir -p ${checkpoints_dir} ${results_dir}

if [[ ! -f "${results_dir}/training.done" ]];
then
    python ${FAIRSEQ}/train.py ${data_dir} \
        --user-dir ${base_dir}/mcolt \
        --save-dir ${checkpoints_dir} \
        --mono-data ${data_dir}/mono \
        ${options} --do_shuf --patience 10 \
        --ddp-backend no_c10d 1>&2
fi

touch "${results_dir}/training.done"

SPM_MODEL=${spm_dir}/${prefix}.model
MODEL="${checkpoints_dir}/checkpoint_best.pt"

if [ ! -f "${dict_dir}/dict.src.txt" ] || [ ! -f "${SPM_MODEL}" ] || [ ! -f "${MODEL}" ] ; then
    echo "[ERROR]: Prefix ${prefix} does not contain spm model or dictionary or trained model. Exiting...";
    exit 1
fi

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"
SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"

for TGT_LANG in "${test_langs[@]}"
do
    TGT_LANG_SHORT="${TGT_LANG:0:2}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"
    echo "Target lang: ${TGT_LANG_SHORT}"

    # Replace with path to test file
    PARALLEL_DIR="${datasets_dir}/en-${TGT_LANG_SHORT}-wmt"
    TEST_PREFIX="${PARALLEL_DIR}/test.detok"
    TEST_FINAL="${PARALLEL_DIR}/preprocessed/${prefix}/test.detok"
    lang_pair="${SRC_LANG_SHORT}-${TGT_LANG_SHORT}"

    rm -rf ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/
    mkdir -p ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/

    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${SRC_LANG} > ${TEST_FINAL}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${TGT_LANG} > ${TEST_FINAL}.spm.${TGT_LANG_SHORT}
    
    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TEST_FINAL}.spm.${SRC_LANG_SHORT} > ${TEST_FINAL}.prefixed.spm.${SRC_LANG_SHORT}
	cp ${TEST_FINAL}.spm.${TGT_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${TGT_LANG_SHORT}

    echo "Tokenizing Done"
    python ${FAIRSEQ}/preprocess.py \
            --source-lang ${SRC_LANG_SHORT} --target-lang  ${TGT_LANG_SHORT} \
            --testpref ${TEST_FINAL}.prefixed.spm \
            --srcdict ${dict_dir}/dict.src.txt \
            --tgtdict ${dict_dir}/dict.src.txt \
            --destdir ${data_dir}/test/${lang_pair}/ \
            --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
            --amp --workers 128

    rm -rf ${TEST_FINAL}.spm.${SRC_LANG_SHORT} ${TEST_FINAL}.spm.${TGT_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${SRC_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${TGT_LANG_SHORT}
    echo "Binarizing Done"

    for i in {0..3};
    do
        mkdir -p ${results_dir}/${lang_pair}/shard_id${i}
        CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pair} \
            --max-tokens 4000 \
            --user-dir ${base_dir}/mcolt \
            --skip-invalid-size-inputs-valid-test \
            --max-source-positions 4000 \
            --max-target-positions 4000 \
            --path ${MODEL} \
            --source-lang ${SRC_LANG_SHORT} --target-lang ${TGT_LANG_SHORT} \
            --beam 5 --task translation_w_langtok \
            --lang-prefix-tok "LANG_TOK_"`echo "${TGT_LANG_SHORT} " | tr '[a-z]' '[A-Z]'` \
            --gen-subset test --dataset-impl mmap \
            --distributed-world-size 4 --distributed-rank ${i} \
            --results-path ${results_dir}/${lang_pair}/shard_id${i} &
    done
    wait

    echo "Translation done"
    cat ${results_dir}/${lang_pair}/shard_id*/*.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | cut -d" " -f2- > ${results_dir}/${lang_pair}/output.tok.txt
    ${SPM}/spm_decode --model=${SPM_MODEL} < ${results_dir}/${lang_pair}/output.tok.txt > ${results_dir}/${lang_pair}/output.detok.txt
    rm -r ${results_dir}/${lang_pair}/output.tok.txt ${data_dir}/test/${lang_pair}
    rm -r ${results_dir}/${lang_pair}/shard_id*/*.txt 

    echo "For lang pair: ${lang_pair} Model type: ${prefix}, BLEU is:"
    mkdir -p ${results_dir}/json/
    sacrebleu --tokenize spm -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT_LANG}" < ${results_dir}/${lang_pair}/output.detok.txt > ${results_dir}/json/${lang_pair}_bleu_chrf.json
    comet-score -s "${TEST_PREFIX}.${SRC_LANG}" -r "${TEST_PREFIX}.${TGT_LANG}" -t ${results_dir}/${lang_pair}/output.detok.txt --gpus 1 --to_json true > ${results_dir}/json/${lang_pair}_comet.json
done

TGT_LANG="en_XX"
TGT_LANG_SHORT="${TGT_LANG:0:2}"
TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"

for SRC_LANG in "${test_langs[@]}"
do
    SRC_LANG_SHORT="${SRC_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
    echo "Source lang: ${SRC_LANG_SHORT}"

    PARALLEL_DIR="${datasets_dir}/en-${SRC_LANG_SHORT}-wmt"
    TEST_PREFIX="${PARALLEL_DIR}/test.detok"
    TEST_FINAL="${PARALLEL_DIR}/preprocessed/${prefix}/test.detok"
    lang_pair="${SRC_LANG_SHORT}-${TGT_LANG_SHORT}"

    rm -rf ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/
    mkdir -p ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/

    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${SRC_LANG} > ${TEST_FINAL}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${TGT_LANG} > ${TEST_FINAL}.spm.${TGT_LANG_SHORT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TEST_FINAL}.spm.${SRC_LANG_SHORT} > ${TEST_FINAL}.prefixed.spm.${SRC_LANG_SHORT}
	cp ${TEST_FINAL}.spm.${TGT_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${TGT_LANG_SHORT}

    echo "Tokenizing Done"
    python ${FAIRSEQ}/preprocess.py \
            --source-lang ${SRC_LANG_SHORT} --target-lang  ${TGT_LANG_SHORT} \
            --testpref ${TEST_FINAL}.prefixed.spm \
            --srcdict ${dict_dir}/dict.src.txt \
            --tgtdict ${dict_dir}/dict.src.txt \
            --destdir ${data_dir}/test/${lang_pair}/ \
            --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
            --amp --workers 128

    rm -rf ${TEST_FINAL}.spm.${SRC_LANG_SHORT} ${TEST_FINAL}.spm.${TGT_LANG_SHORT}
    echo "Binarizing Done"

    for i in {0..3};
    do
        mkdir -p ${results_dir}/${lang_pair}/shard_id${i}
        CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pair} \
            --max-tokens 4000 \
            --user-dir ${base_dir}/mcolt \
            --skip-invalid-size-inputs-valid-test \
            --max-source-positions 4000 \
            --max-target-positions 4000 \
            --path ${MODEL} \
            --skip-invalid-size-inputs-valid-test \
            --source-lang ${SRC_LANG_SHORT} --target-lang ${TGT_LANG_SHORT} \
            --beam 5 --task translation_w_langtok \
            --lang-prefix-tok "LANG_TOK_"`echo "${TGT_LANG_SHORT} " | tr '[a-z]' '[A-Z]'` \
            --gen-subset test --dataset-impl mmap \
            --distributed-world-size 4 --distributed-rank ${i} \
            --results-path ${results_dir}/${lang_pair}/shard_id${i} &
    done
    wait

    echo "Translation done"
    cat ${results_dir}/${lang_pair}/shard_id*/*.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | cut -d" " -f2- > ${results_dir}/${lang_pair}/output.tok.txt
    ${SPM}/spm_decode --model=${SPM_MODEL} < ${results_dir}/${lang_pair}/output.tok.txt > ${results_dir}/${lang_pair}/output.detok.txt
    rm -r ${results_dir}/${lang_pair}/output.tok.txt ${data_dir}/test/${lang_pair}
    rm -r ${results_dir}/${lang_pair}/shard_id*/*.txt 

    echo "For lang pair: ${lang_pair} Model type: ${prefix}, BLEU is:"
    mkdir -p ${results_dir}/json/
    sacrebleu --tokenize spm -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT_LANG}" < ${results_dir}/${lang_pair}/output.detok.txt > ${results_dir}/json/${lang_pair}_bleu_chrf.json
    comet-score -s "${TEST_PREFIX}.${SRC_LANG}" -r "${TEST_PREFIX}.${TGT_LANG}" -t ${results_dir}/${lang_pair}/output.detok.txt --gpus 1 --to_json true > ${results_dir}/json/${lang_pair}_comet.json
done

touch "${results_dir}/testing.done"