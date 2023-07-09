curr_dir=`pwd`
ulimit -n 6000
set -e

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

src="${src_long:0:2}"
tgt="${tgt_long:0:2}"

prefix="sample-blft_${src}-${tgt}"

echo "Bilingual finetuning: ${src} -> ${tgt}"

test_langs=(fi_FI et_EE)

# More or less fixed
# More or less fixed
base_dir="~/CAS" # Replace with the base directory of the project 
datasets_dir="~/CAS/data" # Replace with directory containing all datasets
SPM=~/sentencepiece/build/src # Sentencepiece path
FAIRSEQ=~/fairseq/fairseq_cli # Fairseq path
training_config="${base_dir}/configs/transformer-config.yml"

# Derived
data_dir="${base_dir}/data/${prefix}"
checkpoints_dir="${base_dir}/checkpoints/${prefix}"
results_dir="${base_dir}/results/${prefix}"
spm_dir="${data_dir}/spm"
SPM_MODEL="${data_dir}/spm/${prefix}.model"

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
        ${options} --do_shuf --only_parallel \
        --source-lang ${src} --target-lang ${tgt} \
        --patience 10 --no-epoch-checkpoints \
	    --reset-dataloader \
        --ddp-backend no_c10d 1>&2
fi

touch "${results_dir}/training.done"

MODEL="${checkpoints_dir}/checkpoint_best.pt"

if [ ! -f "${data_dir}/dict.${src}.txt" ]; then
    echo "${data_dir}/dict.${src}.txt"
    echo "[ERROR 1]: Prefix ${prefix} does not contain dictionary. Exiting...";
    exit 1
fi

if [ ! -f "${SPM_MODEL}" ]; then
    echo "[ERROR 2 ]: Prefix ${prefix} does not contain spm model. Exiting...";
    exit 1
fi


if [ ! -f "${MODEL}" ] ; then
    echo "[ERROR 3]: Prefix ${prefix} does not contain trained model. Exiting...";
    exit 1
fi


if [ -d "${datasets_dir}/${tgt}-${src}-wmt" ]; then
    PARALLEL_DIR="${datasets_dir}/${tgt}-${src}-wmt"
elif [ -d "${datasets_dir}/${src}-${tgt}-wmt" ]; then
    PARALLEL_DIR="${datasets_dir}/${src}-${tgt}-wmt"
else
    echo "No parallel dir to be found with ${tgt}, ${src} and wmt"
    exit 1
fi
TEST_PREFIX="${PARALLEL_DIR}/test.detok"
TEST_FINAL="${PARALLEL_DIR}/preprocessed/${prefix}/test.detok"
lang_pair="${src}-${tgt}"

rm -rf ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/
mkdir -p ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/

${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${src_long} > ${TEST_FINAL}.spm.${src}
${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${tgt_long} > ${TEST_FINAL}.spm.${tgt}


echo "Tokenizing Done"
python ${FAIRSEQ}/preprocess.py \
        --source-lang ${src} --target-lang ${tgt} \
        --testpref ${TEST_FINAL}.spm \
        --srcdict ${data_dir}/dict.${src}.txt \
        --tgtdict ${data_dir}/dict.${tgt}.txt \
        --destdir ${data_dir}/test/${lang_pair}/ \
        --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
        --amp --workers 42

rm -rf ${TEST_FINAL}.spm.${src_long} ${TEST_FINAL}.spm.${tgt_long}
echo "Binarizing Done"

for i in {0..2};
do
    mkdir -p ${results_dir}/${lang_pair}/shard_id${i}
    CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pair} \
            --max-tokens 4000 \
            --user-dir ${base_dir}/mcolt \
            --skip-invalid-size-inputs-valid-test \
            --max-source-positions 4000 \
            --max-target-positions 4000 \
            --path ${MODEL} \
            --source-lang ${src} --target-lang ${tgt} \
            --beam 5 --task translation_w_langtok \
            --lang-prefix-tok "LANG_TOK_"`echo "${tgt} " | tr '[a-z]' '[A-Z]'` \
            --gen-subset test --dataset-impl mmap \
            --distributed-world-size 3 --distributed-rank ${i} \
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
sacrebleu --tokenize spm -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${tgt_long}" < ${results_dir}/${lang_pair}/output.detok.txt > ${results_dir}/json/${lang_pair}_bleu_chrf.json
comet-score -s "${TEST_PREFIX}.${src_long}" -r "${TEST_PREFIX}.${tgt_long}" -t ${results_dir}/${lang_pair}/output.detok.txt --gpus 1 --to_json true > ${results_dir}/json/${lang_pair}_comet.json
# touch "${results_dir}/testing.done"
