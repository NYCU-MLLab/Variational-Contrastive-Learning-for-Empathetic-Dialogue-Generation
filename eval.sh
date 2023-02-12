while getopts ":d:t:m:s:c:" opt
do
    case $opt in
        d)
        DATASET="$OPTARG"
        ;;
        t)
        TRAINED_DATASET="$OPTARG"
        ;;
        m)
        MODEL_TYPE="$OPTARG"
        ;;
        s)
        DECODING_STRATEGY="$OPTARG"
        ;;
        c)
        CHECKPOINT="$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

echo '-------- evaluate on dataset: '"$DATASET"'--------'

PROJECT_PATH='.'
DATA_DIR=${PROJECT_PATH}/data/finetune/${DATASET}
PRED_FILE=${DATA_DIR}/infer/${TRAINED_DATASET}/${MODEL_TYPE}/${DECODING_STRATEGY}_${CHECKPOINT}_pred.txt
if [ "$DATASET" == "dailydialog" ]; then
  python3 utils/evaluate.py \
    -name dailydialog \
    -hyp "${PRED_FILE}" \
    -ref "${DATA_DIR}"/processed/test.tgt

elif [ "$DATASET" == "empatheticdialog"  ]; then
  python3 utils/evaluate.py \
    -name empatheticdialog \
    -hyp "${PRED_FILE}" \
    -ref "${DATA_DIR}"/processed/test.tgt
elif [ "$DATASET" == "empatheticdialog_annotated"  ]; then
  python3 utils/evaluate.py \
    -name empatheticdialog_annotated \
    -hyp "${PRED_FILE}" \
    -ref "${DATA_DIR}"/processed/dialog/test.tgt
else
  echo 'dataset '"$DATASET"' not found!'
  exit 1
fi
