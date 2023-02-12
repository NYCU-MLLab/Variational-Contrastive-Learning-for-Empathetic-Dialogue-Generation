while getopts ":d:s:t:m:c:" opt
do
    case $opt in
        d)
        DATASET="$OPTARG"
        ;;
        s)
        DECODING_STRATEGY="$OPTARG"
        ;;
        t)
        TRAINED_DATASET="$OPTARG"
        ;;
        m)
        MODEL_TYPE="$OPTARG"
        ;;
        c)
        CHECKPOINT="$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done


PROJECT_PATH='.'
USER_DIR=${PROJECT_PATH}/src
DATA_DIR=${PROJECT_PATH}/data/finetune/${DATASET}
SAVE_DIR=${DATA_DIR}/infer/${TRAINED_DATASET}/${MODEL_TYPE}
# SAVE_DIR=${DATA_DIR}/infer/exp/${TRAINED_DATASET}/${MODEL_TYPE}


if [ ${TRAINED_DATASET} = "reddit" ]; then
  MODEL_DIR=${PROJECT_PATH}/models/dialogved_${MODEL_TYPE}.pt
else
  MODEL_DIR=${PROJECT_PATH}/data/finetune/${TRAINED_DATASET}/checkpoints/${MODEL_TYPE}/${CHECKPOINT}.pt
  # MODEL_DIR=${PROJECT_PATH}/data/exp/empatheticdialog/${TRAINED_DATASET}/checkpoints/${MODEL_TYPE}/${CHECKPOINT}.pt
fi

if [ ! -d  ${SAVE_DIR} ]; then
  mkdir -p ${SAVE_DIR}
fi

echo '-------- inference on dataset: '"$DATASET"'--------'

if [ "$DECODING_STRATEGY" == "greedy" ]; then
  echo '-------- decoding strategy: greedy --------'
  # inference
  BEAM=1
  LENPEN=1
  OUTPUT_FILE=${SAVE_DIR}/${DECODING_STRATEGY}_${CHECKPOINT}_output.txt
  PRED_FILE=${SAVE_DIR}/${DECODING_STRATEGY}_${CHECKPOINT}_pred.txt
  TASK=ved_translate
  fairseq-generate "${DATA_DIR}"/binary \
    --path "${MODEL_DIR}" \
    --user-dir ${USER_DIR} \
    --task ${TASK} \
    --batch-size 64 \
    --gen-subset test \
    --beam ${BEAM} \
    --num-workers 4 \
    --no-repeat-ngram-size 3 \
    --lenpen ${LENPEN} \
    2>&1 >"${OUTPUT_FILE}"
  grep ^H "${OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > "${PRED_FILE}"
elif [ "$DECODING_STRATEGY" == "beam" ]; then
  echo '-------- decoding strategy: beam search --------'
  # inference
  BEAM=5
  LENPEN=1
  OUTPUT_FILE=${SAVE_DIR}/${DECODING_STRATEGY}_${CHECKPOINT}_output.txt
  PRED_FILE=${SAVE_DIR}/${DECODING_STRATEGY}_${CHECKPOINT}_pred.txt
  TASK=ved_translate_empathetic
  fairseq-generate "${DATA_DIR}"/binary \
    --path "${MODEL_DIR}" \
    --user-dir ${USER_DIR} \
    --task ${TASK} \
    --batch-size 64 \
    --gen-subset test \
    --beam ${BEAM} \
    --num-workers 4 \
    --no-repeat-ngram-size 3 \
    --lenpen ${LENPEN} \
    2>&1 >"${OUTPUT_FILE}"
  grep ^H "${OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > "${PRED_FILE}"
elif [ "$DECODING_STRATEGY" == "sampling"  ]; then
  echo '-------- decoding strategy: sampling --------'
  LENPEN=1
  TOP_K=100
  OUTPUT_FILE=${SAVE_DIR}/${DECODING_STRATEGY}_${CHECKPOINT}_output.txt
  PRED_FILE=${SAVE_DIR}/${DECODING_STRATEGY}_${CHECKPOINT}_pred.txt
  TASK=ved_translate
  fairseq-generate "${DATA_DIR}"/binary \
  --path "${MODEL_DIR}" \
  --user-dir ${USER_DIR} \
  --task ${TASK} \
  --batch-size 64 \
  --gen-subset test \
  --num-workers 4 \
  --no-repeat-ngram-size 3 \
  --lenpen ${LENPEN} \
  --sampling \
  --sampling-topk ${TOP_K} \
  --nbest 1 \
  --beam 1 \
  2>&1 >"${OUTPUT_FILE}"
  grep ^H "${OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > "${PRED_FILE}"
else
  echo 'decoding strategy '"$DECODING_STRATEGY"' not found!'
  exit 1
fi
