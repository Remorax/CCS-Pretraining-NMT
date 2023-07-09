set -e 

SPM=~/sentencepiece/build/src/spm_encode # Sentencepiece encode path
SPM_DECODE=~/sentencepiece/build/src/spm_decode # Sentencepiece decode path
FAIRSEQ=~/fairseq/fairseq_cli # Fairseq path
SPM_MODEL=~/pretrained_models/mbart50.ft.1n/sentencepiece.bpe.model
MODEL=~/pretrained_models/mbart50.ft.1n/model.pt
DICT_SRC=~/pretrained_models/mbart50.ft.1n/dict.txt
DICT_TGT=~/pretrained_models/mbart50.ft.1n/dict.txt
lang_list=~/pretrained_models/mbart50.ft.1n/langs.txt
max_tokens_test=7500    
WORLD_SIZE=3

# More or less fixed
base_dir="~/CAS"
datasets_dir="~/CAS/data"
prefix="mBART50-ft"

# Derived
data_dir="${base_dir}/data/${prefix}"
results_dir="${base_dir}/results/${prefix}"

source ~/.bashrc
source activate fairseq-mbart

TEST=test.detok
max_tokens_test=7500

SRC=en_XX
WORLD_SIZE=3

langs=(et_EE fi_FI)
for TGT in "${langs[@]}"
do
	src=`echo ${SRC} | cut -c 1-2`
	tgt=`echo ${TGT} | cut -c 1-2`
    
	PARALLEL_DIR="${datasets_dir}/en-${tgt}-wmt"
    TEST_PREFIX="${PARALLEL_DIR}/${TEST}"
	
    ${SPM} --model=${SPM_MODEL} < ${TEST_PREFIX}.${SRC} > ${TEST_PREFIX}.spm.${SRC}
    ${SPM} --model=${SPM_MODEL} < ${TEST_PREFIX}.${TGT} > ${TEST_PREFIX}.spm.${TGT}

    lang_pairs="${SRC}-${TGT}"
    rm -rf ${data_dir}/test/${lang_pairs}
    mkdir -p ${data_dir}/test/${lang_pairs}

    python ${FAIRSEQ}/preprocess.py \
        --source-lang ${SRC} --target-lang ${TGT} \
        --testpref ${TEST_PREFIX}.spm \
        --destdir ${data_dir}/test/${lang_pairs} \
        --thresholdsrc 0 --thresholdtgt 0 \
        --srcdict ${DICT_SRC} --tgtdict ${DICT_TGT} \
        --workers 42

	rm -rf ${results_dir}/test/${lang_pairs}

    for i in {0..2};
    do
        mkdir -p ${results_dir}/test/${lang_pairs}/shard_id${i}/
        CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pairs} \
                --max-tokens ${max_tokens_test} \
                --path ${MODEL} --lang-dict ${lang_list} \
                --fixed-dictionary ${DICT_SRC} \
                --source-lang ${SRC} --target-lang ${TGT} \
                --beam 5 \
                --task translation_multi_simple_epoch \
                --lang-pairs "${SRC}-${TGT}" \
                --decoder-langtok --encoder-langtok src \
                --gen-subset test --dataset-impl mmap \
                --distributed-world-size ${WORLD_SIZE} --distributed-rank ${i} \
                --results-path ${results_dir}/test/${lang_pairs}/shard_id${i} &
    done
    wait

    echo "Done Generation. Postprocessing..."
    rm -f "${results_dir}/test/${lang_pairs}/generate-test-all.txt"
    touch "${results_dir}/test/${lang_pairs}/generate-test-all.txt"

    for i in {0..2};
    do
            cat "${results_dir}/test/${lang_pairs}/shard_id${i}/generate-test.txt" >> "${results_dir}/test/${lang_pairs}/generate-test-all.txt"
    done
    rm -f "${results_dir}/test/${lang_pairs}/output.postproc.txt"
    cat "${results_dir}/test/${lang_pairs}/generate-test-all.txt" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | ${SPM_DECODE} --model=${SPM_MODEL}  > "${results_dir}/test/${lang_pairs}/output.postproc.txt"

    echo "All Done."
    mkdir -p  ${results_dir}/test/json/
    sacrebleu --tokenize spm -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT}" < "${results_dir}/test/${lang_pairs}/output.postproc.txt" > ${results_dir}/test/json/${lang_pairs}_bleu_chrf.json
    comet-score -s "${TEST_PREFIX}.${SRC}" -r "${TEST_PREFIX}.${TGT}" -t "${results_dir}/test/${lang_pairs}/output.postproc.txt" --gpus 1 --to_json true > ${results_dir}/test/json/${lang_pairs}_comet.json

done

SPM_MODEL=~/pretrained_models/mbart50.ft.n1/sentencepiece.bpe.model
MODEL=~/pretrained_models/mbart50.ft.n1/model.pt
DICT_SRC=~/pretrained_models/mbart50.ft.n1/dict.txt
DICT_TGT=~/pretrained_models/mbart50.ft.n1/dict.txt
lang_list=~/pretrained_models/mbart50.ft.n1/langs.txt
TGT=en_XX


langs=(et_EE fi_FI)
for SRC in "${langs[@]}"
do
	src=`echo ${SRC} | cut -c 1-2`
	tgt=`echo ${TGT} | cut -c 1-2`
    
	PARALLEL_DIR="${datasets_dir}/en-${src}-wmt"
    TEST_PREFIX="${PARALLEL_DIR}/${TEST}"
	
    ${SPM} --model=${SPM_MODEL} < ${TEST_PREFIX}.${SRC} > ${TEST_PREFIX}.spm.${SRC}
    ${SPM} --model=${SPM_MODEL} < ${TEST_PREFIX}.${TGT} > ${TEST_PREFIX}.spm.${TGT}

    lang_pairs="${SRC}-${TGT}"
    rm -rf ${data_dir}/test/${lang_pairs}
    mkdir -p ${data_dir}/test/${lang_pairs}

    python ${FAIRSEQ}/preprocess.py \
        --source-lang ${SRC} --target-lang ${TGT} \
        --testpref ${TEST_PREFIX}.spm \
        --destdir ${data_dir}/test/${lang_pairs} \
        --thresholdsrc 0 --thresholdtgt 0 \
        --srcdict ${DICT_SRC} --tgtdict ${DICT_TGT} \
        --workers 42

	rm -rf ${results_dir}/test/${lang_pairs}

    for i in {0..2};
    do
        mkdir -p ${results_dir}/test/${lang_pairs}/shard_id${i}/
        CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pairs} \
                --max-tokens ${max_tokens_test} \
                --path ${MODEL} --lang-dict ${lang_list} \
                --fixed-dictionary ${DICT_SRC} \
                --source-lang ${SRC} --target-lang ${TGT} \
                --beam 5 \
                --task translation_multi_simple_epoch \
                --lang-pairs "${SRC}-${TGT}" \
                --decoder-langtok --encoder-langtok src \
                --gen-subset test --dataset-impl mmap \
                --distributed-world-size ${WORLD_SIZE} --distributed-rank ${i} \
                --results-path ${results_dir}/test/${lang_pairs}/shard_id${i} &
    done
    wait

    echo "Done Generation. Postprocessing..."
    rm -f "${results_dir}/test/${lang_pairs}/generate-test-all.txt"
    touch "${results_dir}/test/${lang_pairs}/generate-test-all.txt"

    for i in {0..2};
    do
            cat "${results_dir}/test/${lang_pairs}/shard_id${i}/generate-test.txt" >> "${results_dir}/test/${lang_pairs}/generate-test-all.txt"
    done
    rm -f "${results_dir}/test/${lang_pairs}/output.postproc.txt"
    cat "${results_dir}/test/${lang_pairs}/generate-test-all.txt" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}'  | ${SPM_DECODE} --model=${SPM_MODEL} > "${results_dir}/test/${lang_pairs}/output.postproc.txt"

    echo "All Done."
    mkdir -p  ${results_dir}/test/json/
    sacrebleu --tokenize spm -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT}" < "${results_dir}/test/${lang_pairs}/output.postproc.txt" > ${results_dir}/test/json/${lang_pairs}_bleu_chrf.json
    comet-score -s "${TEST_PREFIX}.${SRC}" -r "${TEST_PREFIX}.${TGT}" -t "${results_dir}/test/${lang_pairs}/output.postproc.txt" --gpus 1 --to_json true > ${results_dir}/test/json/${lang_pairs}_comet.json

done
