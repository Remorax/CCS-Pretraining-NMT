curr_dir=`pwd`
ulimit -n 6000
set -e

# More or less fixed
SPM=~/sentencepiece/build/src/spm_encode # Sentencepiece encode path
SPM_DECODE=~/sentencepiece/build/src/spm_decode # Sentencepiece decode path
FAIRSEQ=~/fairseq/fairseq_cli # Fairseq path
base_dir="~/CAS"
SCRIPTS_DIR="~/scripts" # Moses scripts path
datasets_dir="~/CAS/data"
prefix="mRASP2"

# Derived
data_dir="${base_dir}/data/${prefix}"
checkpoints_dir="${base_dir}/checkpoints/${prefix}"
dict_dir="${data_dir}/dict"
results_dir="${base_dir}/results/${prefix}"
spm_dir="${data_dir}/spm"

TEST=test.detok
max_tokens_test=7500

SRC=en_XX
WORLD_SIZE=3

MODEL=~/pretrained_models/mRASP.12e12d/12e12d_last.pt
VOCAB_BPE=~/pretrained_models/mRASP.12e12d/bpe_vocab
CODES_BPE=~/pretrained_models/mRASP.12e12d/codes.bpe.32000

langs=(et_EE fi_FI)
for TGT in "${langs[@]}"
do
	src=`echo ${SRC} | cut -c 1-2`
	tgt=`echo ${TGT} | cut -c 1-2`
    lang_pair="${src}-${tgt}"
	PARALLEL_DIR="${datasets_dir}/en-${tgt}-wmt"
    TEST_PREFIX="${PARALLEL_DIR}/${TEST}"
	
	cat  ${TEST_PREFIX}.${SRC} | perl ${SCRIPTS_DIR}/moses_tokenizer.pl -a -q -l ${src} -no-escape -threads 42 > ${TEST_PREFIX}.tok.${SRC}
	subword-nmt apply-bpe -c ${CODES_BPE} < ${TEST_PREFIX}.tok.${SRC} > ${TEST_PREFIX}.tok.bpe.${SRC}

	awk '{print "LANG_TOK_EN " $0} ' ${TEST_PREFIX}.tok.bpe.${SRC} > ${TEST_PREFIX}.RAS.${SRC}

	echo "TGT: ${TGT}"
        
	cat  ${TEST_PREFIX}.${TGT} | perl ${SCRIPTS_DIR}/moses_tokenizer.pl -a -q -l ${tgt} -no-escape -threads 42  > ${TEST_PREFIX}.tok.${TGT}
    subword-nmt apply-bpe -c ${CODES_BPE} < ${TEST_PREFIX}.tok.${TGT} > ${TEST_PREFIX}.tok.bpe.${TGT}

    cp ${TEST_PREFIX}.tok.bpe.${TGT} ${TEST_PREFIX}.RAS.${TGT}

    lang_pairs="${SRC}-${TGT}"
    rm -rf ${data_dir}/test/${lang_pairs}
	python ${FAIRSEQ}/preprocess.py \
        	--source-lang ${SRC} --target-lang ${TGT} \
        	--srcdict ${VOCAB_BPE} \
        	--tgtdict ${VOCAB_BPE} \
        	--trainpref ${TEST_PREFIX}.RAS \
        	--destdir ${data_dir}/test/${lang_pairs} \
        	--workers 70

	rm -rf ${results_dir}/test/${lang_pair}

    curr_dir=`pwd`
    for i in {0..2};
    do
        mkdir -p ${results_dir}/test/${lang_pair}/shard_id${i}
        CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pairs} \
            --max-tokens ${max_tokens_test} \
            --user-dir ${base_dir}/mcolt \
            --skip-invalid-size-inputs-valid-test \
            --max-source-positions ${max_tokens_test} \
            --max-target-positions ${max_tokens_test} \
            --path ${MODEL} \
            --source-lang ${SRC} --target-lang ${TGT} \
            --beam 5 \
            --task translation_w_langtok \
            --lang-prefix-tok "LANG_TOK_"`echo "${tgt} " | tr '[a-z]' '[A-Z]'` \
            --gen-subset train --dataset-impl mmap \
            --distributed-world-size 3 --distributed-rank ${i} \
            --results-path "${results_dir}/test/${lang_pair}/shard_id${i}" &
    done
    wait

	echo "Done Generation. Postprocessing..."
    cat ${results_dir}/test/${lang_pair}/shard_id*/*.txt > "${results_dir}/test/${lang_pair}/generate-train.txt"
	cat "${results_dir}/test/${lang_pair}/generate-train.txt" | grep -E '[S|H|P|T]-[0-9]+'  >  "${results_dir}/test/${lang_pair}/output.postproc.final.txt"
    cd ${base_dir}
    python scripts/utils.py "${results_dir}/test/${lang_pair}/output.postproc.final.txt" ${TEST_PREFIX}.${TGT} || exit 1;
    cd ${curr_dir}
    input_file="${results_dir}/test/${lang_pair}/hypo.out.nobpe"
    output_file="${results_dir}/test/${lang_pair}/hypo.out.nobpe.final"
    cmd="cat ${input_file}"

    lang_token="LANG_TOK_"`echo "${tgt} " | tr '[a-z]' '[A-Z]'`
    if [[ $tgt == "fr" ]]; then
        cmd=$cmd" | sed -Ee 's/\"([^\"]*)\"/« \1 »/g'"
    elif [[ $tgt == "zh" ]]; then
        tokenizer="zh"
    elif [[ $tgt == "ja" ]]; then
        tokenizer="ja-mecab"
    fi
    [[ -z $tokenizer ]] && tokenizer="none"
    cmd=$cmd" | sed -e s'|${lang_token} ||g' > ${output_file}"
    eval $cmd || { echo "$cmd FAILED"; exit 1; }
    
    perl ${SCRIPTS_DIR}/moses_detokenizer.perl -l ${tgt} -threads 42 < ${output_file} > ${output_file}.detok

    mkdir -p ${results_dir}/test/json/
    
    sacrebleu --tokenize spm -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT}" < ${output_file}.detok > ${results_dir}/test/json/${lang_pair}_bleu_chrf.json
    comet-score -s "${TEST_PREFIX}.${SRC}" -r "${TEST_PREFIX}.${TGT}" -t ${output_file}.detok --gpus 1 --to_json true > ${results_dir}/test/json/${lang_pair}_comet.json

done

TGT=en_XX
for SRC in "${langs[@]}"
do
	src=`echo ${SRC} | cut -c 1-2`
	tgt=`echo ${TGT} | cut -c 1-2`
    lang_pair="${src}-${tgt}"
	PARALLEL_DIR="${datasets_dir}/en-${src}-wmt"
    TEST_PREFIX="${PARALLEL_DIR}/${TEST}"
	
	cat  ${TEST_PREFIX}.${SRC} | perl ${SCRIPTS_DIR}/moses_tokenizer.pl -a -q -l ${src} -no-escape -threads 42 > ${TEST_PREFIX}.tok.${SRC}
	subword-nmt apply-bpe -c ${CODES_BPE} < ${TEST_PREFIX}.tok.${SRC} > ${TEST_PREFIX}.tok.bpe.${SRC}

    src_caps=`echo "${src} " | tr '[a-z]' '[A-Z]'`
	awk -v src=$src_caps '{print "LANG_TOK_"src" " $0 } ' ${TEST_PREFIX}.tok.bpe.${SRC} > ${TEST_PREFIX}.RAS.${SRC}

	echo "TGT: ${TGT}"
        
	cat  ${TEST_PREFIX}.${TGT} | perl ${SCRIPTS_DIR}/moses_tokenizer.pl -a -q -l ${tgt} -no-escape -threads 42  > ${TEST_PREFIX}.tok.${TGT}
    subword-nmt apply-bpe -c ${CODES_BPE} < ${TEST_PREFIX}.tok.${TGT} > ${TEST_PREFIX}.tok.bpe.${TGT}

    cp ${TEST_PREFIX}.tok.bpe.${TGT} ${TEST_PREFIX}.RAS.${TGT}

    lang_pairs="${SRC}-${TGT}"
    rm -rf ${data_dir}/test/${lang_pairs}
	python ${FAIRSEQ}/preprocess.py \
        	--source-lang ${SRC} --target-lang ${TGT} \
        	--srcdict ${VOCAB_BPE} \
        	--tgtdict ${VOCAB_BPE} \
        	--trainpref ${TEST_PREFIX}.RAS \
        	--destdir ${data_dir}/test/${lang_pairs} \
        	--workers 70

	rm -rf ${results_dir}/test/${lang_pair}

    curr_dir=`pwd`
    for i in {0..2};
    do
        mkdir -p ${results_dir}/test/${lang_pair}/shard_id${i}
        CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pairs} \
            --max-tokens ${max_tokens_test} \
            --user-dir ${base_dir}/mcolt \
            --skip-invalid-size-inputs-valid-test \
            --max-source-positions ${max_tokens_test} \
            --max-target-positions ${max_tokens_test} \
            --path ${MODEL} \
            --source-lang ${SRC} --target-lang ${TGT} \
            --beam 5 \
            --task translation_w_langtok \
            --lang-prefix-tok "LANG_TOK_"`echo "${tgt} " | tr '[a-z]' '[A-Z]'` \
            --gen-subset train --dataset-impl mmap \
            --distributed-world-size 3 --distributed-rank ${i} \
            --results-path "${results_dir}/test/${lang_pair}/shard_id${i}" &
    done
    wait

	echo "Done Generation. Postprocessing..."
    cat ${results_dir}/test/${lang_pair}/shard_id*/*.txt > "${results_dir}/test/${lang_pair}/generate-train.txt"
	cat "${results_dir}/test/${lang_pair}/generate-train.txt" | grep -E '[S|H|P|T]-[0-9]+'  >  "${results_dir}/test/${lang_pair}/output.postproc.final.txt"
    cd ${base_dir}
    python scripts/utils.py "${results_dir}/test/${lang_pair}/output.postproc.final.txt" ${TEST_PREFIX}.${TGT} || exit 1;
    cd ${curr_dir}
    input_file="${results_dir}/test/${lang_pair}/hypo.out.nobpe"
    output_file="${results_dir}/test/${lang_pair}/hypo.out.nobpe.final"
    cmd="cat ${input_file}"

    lang_token="LANG_TOK_"`echo "${tgt} " | tr '[a-z]' '[A-Z]'`
    if [[ $tgt == "fr" ]]; then
        cmd=$cmd" | sed -Ee 's/\"([^\"]*)\"/« \1 »/g'"
    elif [[ $tgt == "zh" ]]; then
        tokenizer="zh"
    elif [[ $tgt == "ja" ]]; then
        tokenizer="ja-mecab"
    fi
    [[ -z $tokenizer ]] && tokenizer="none"
    cmd=$cmd" | sed -e s'|${lang_token} ||g' > ${output_file}"
    eval $cmd || { echo "$cmd FAILED"; exit 1; }
    
    perl ${SCRIPTS_DIR}/moses_detokenizer.perl -l ${tgt} -threads 42 < ${output_file} > ${output_file}.detok

    mkdir -p ${results_dir}/test/json/
    
    sacrebleu --tokenize spm -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT}" < ${output_file}.detok > ${results_dir}/test/json/${lang_pair}_bleu_chrf.json
    comet-score -s "${TEST_PREFIX}.${SRC}" -r "${TEST_PREFIX}.${TGT}" -t ${output_file}.detok --gpus 1 --to_json true > ${results_dir}/test/json/${lang_pair}_comet.json

done
