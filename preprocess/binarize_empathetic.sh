# this script is used pre-process fine-tune dialogue corpus
# in this paper we consider three corpus, i.e., dailydialog, dstc7avsd, personachat
# running this script will cause all files with the same name in the `BINARY_DIR` folder will be overwritten

PROJECT_PATH=/home/Variational-Contrastive-Learning-for-Empathetic-Dialogue-Generation
#PROJECT_PATH=/remote-home/wchen/project/DialogVED

USER_DIR=${PROJECT_PATH}/src
VOCAB_PATH=${PROJECT_PATH}/vocab.txt
NUM_WORKERS=20

# pre-process (empatheticdialog_annotated)
########################################################################################################################
DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
PROCESSED_DIR=${DATA_DIR}/processed/dialog
BINARY_DIR=${DATA_DIR}/binary

"$(which fairseq-preprocess)" \
  --fp16 \
  --user-dir ${USER_DIR} \
  --task ved_translate \
  --source-lang src \
  --target-lang tgt \
  --trainpref ${PROCESSED_DIR}/train \
  --validpref ${PROCESSED_DIR}/valid \
  --testpref ${PROCESSED_DIR}/test \
  --destdir ${BINARY_DIR} \
  --srcdict ${VOCAB_PATH} \
  --tgtdict ${VOCAB_PATH} \
  --workers ${NUM_WORKERS}

DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
PROCESSED_DIR=${DATA_DIR}/processed/emotions
BINARY_DIR=${DATA_DIR}/binary/emotions
"$(which fairseq-preprocess)" \
  --fp16 \
  --only-source \
  --user-dir ${USER_DIR} \
  --task ved_translate \
  --source-lang emotion \
  --trainpref ${PROCESSED_DIR}/train \
  --validpref ${PROCESSED_DIR}/valid \
  --testpref ${PROCESSED_DIR}/test \
  --destdir ${BINARY_DIR} \
  --workers ${NUM_WORKERS}

DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
PROCESSED_DIR=${DATA_DIR}/processed/actions
BINARY_DIR=${DATA_DIR}/binary/actions
"$(which fairseq-preprocess)" \
  --fp16 \
  --only-source \
  --user-dir ${USER_DIR} \
  --task ved_translate \
  --source-lang action \
  --trainpref ${PROCESSED_DIR}/train \
  --validpref ${PROCESSED_DIR}/valid \
  --testpref ${PROCESSED_DIR}/test \
  --destdir ${BINARY_DIR} \
  --workers ${NUM_WORKERS}

