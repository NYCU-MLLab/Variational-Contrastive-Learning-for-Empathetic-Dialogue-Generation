while getopts ":p:t:d:" opt
do
    case $opt in
        p)
          PRETRAINED_MODEL_PATH="$OPTARG"
        ;;
        t)
        PRETRAINED_MODEL_TYPE="$OPTARG"
        ;;
        d)
        DATASET="$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

PROJECT_PATH='.'

if [ "$PRETRAINED_MODEL_TYPE" == "dialogved_standard" ]; then
  echo '-------- model type: dialogved standard --------'
  ARCH=ngram_transformer_prophet_vae_standard
  MODE=ved_standard
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_large" ]; then
  echo '-------- model type: dialogved large --------'
  ARCH=ngram_transformer_prophet_vae_large
  MODE=ved_large
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_seq2seq"  ]; then
  echo '-------- model type: dialogved seq2seq --------'
  ARCH=ngram_transformer_prophet_seq2seq
  MODE=seq2seq
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_standard_latent"  ]; then
  echo '-------- model type: dialogved standard latent --------'
  ARCH=ngram_transformer_prophet_vae_standard_latent
  MODE=ved_standard_latent
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_large_latent"  ]; then
  echo '-------- model type: dialogved large latent --------'
  ARCH=ngram_transformer_prophet_vae_large_latent
  MODE=ved_large_latent
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_seq2seq_latent"  ]; then
  echo '-------- model type: dialogved seq2seq latent--------'
  ARCH=ngram_transformer_prophet_seq2seq_latent
  MODE=seq2seq_latent
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_seq2seq_latent_no_ngram"  ]; then
  echo '-------- model type: dialogved seq2seq latent no ngram--------'
  ARCH=ngram_transformer_prophet_seq2seq_latent_no_ngram
  MODE=seq2seq_latent_no_ngram

elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_large_no_ngram"  ]; then
  echo '-------- model type: dialogved large no ngram--------'
  ARCH=ngram_transformer_prophet_vae_large_no_ngram
  MODE=ved_large_no_ngram
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_seq2seq_no_ngram"  ]; then
  echo '-------- model type: dialogved seq2seq no ngram--------'
  ARCH=ngram_transformer_prophet_seq2seq_no_ngram
  MODE=seq2seq_no_ngram
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_large_mean_latent_no_ngram"  ]; then
  echo '-------- model type: dialogved large mean latent no ngram-------'
  ARCH=ngram_transformer_prophet_vae_large_mean_latent_no_ngram
  MODE=ved_large_mean_latent_no_ngram
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_large_no_ngram_another_posterior"  ]; then
  echo '-------- model type: dialogved large mean latent no ngram-------'
  ARCH=ngram_transformer_prophet_vae_large_no_ngram_another_posterior
  MODE=ved_large_no_ngram_another_posterior
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_large_no_ngram_attn_pre"  ]; then
  echo '-------- model type: dialogved large mean latent no ngram-------'
  ARCH=ngram_transformer_prophet_vae_large_no_ngram_attn_pre
  MODE=ved_large_no_ngram_attn_pre
else
z
  echo 'model type '"$PRETRAINED_MODEL_TYPE"' not found!'
  exit 1
fi



if [ "$DATASET" == "dailydialog" ]; then
  echo '-------- fine-tune on dataset: dailydialog --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/dailydialog
  SAVE_DIR=${DATA_DIR}/checkpoints/${MODE}
  TB_LOGDIR=${DATA_DIR}/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 10 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
elif [ "$DATASET" == "dstc7avsd" ]; then
  echo '-------- fine-tune on dataset: dstc7avsd --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/dstc7avsd
  SAVE_DIR=${DATA_DIR}/checkpoints/${MODE}
  TB_LOGDIR=${DATA_DIR}/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0003 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 4 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 10 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
elif [ "$DATASET" == "personachat"  ]; then
  echo '-------- fine-tune on dataset: personachat --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/personachat
  SAVE_DIR=${DATA_DIR}/checkpoints/${MODE}
  TB_LOGDIR=${DATA_DIR}/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0003 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 4 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 10 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
elif [ "$DATASET" == "empatheticdialog" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog
  SAVE_DIR=${DATA_DIR}/checkpoints/${MODE}
  TB_LOGDIR=${DATA_DIR}/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0003 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 64 --max-tokens 4500 --max-sentences 8 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 10 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"  
elif [ "$DATASET" == "empatheticdialog_annotated" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${DATA_DIR}/checkpoints/${MODE}
  TB_LOGDIR=${DATA_DIR}/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --action-labels True \
    --emotion-labels True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"      


elif [ "$DATASET" == "empatheticdialog_annotated_cl_coherence" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_coherence/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_coherence/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 100 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 10 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 0.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --add-cl-loss True\
    --cl-coherence True \
    --temp-scale 0.05\
    --use-adapter True\
    --freeze-model True\
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"    
elif [ "$DATASET" == "empatheticdialog_annotated_cl_supervised" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_supervised/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_supervised/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 20 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 0.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --action-labels True \
    --emotion-labels True \
    --add-cl-loss True \
    --cl-action True \
    --cl-emotion True \
    --temp-scale 0.05\
    --use-adapter True\
    --freeze-model True\
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"    

elif [ "$DATASET" == "empatheticdialog_annotated_cl_all" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_all/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_all/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 100 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 10 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --action-labels True \
    --emotion-labels True \
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --cl-coherence True \
    --temp-scale 0.5\
    --use-adapter True\
    --freeze-model True\
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"    
elif [ "$DATASET" == "empatheticdialog_annotated_cl_style" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_style/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_style/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 16 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 10 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 0.1 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --cl-style True \
    --temp-scale 0.05\
    --use-adapter True\
    --freeze-model True\
    --action-labels True \
    --emotion-labels True \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"    
elif [ "$DATASET" == "empatheticdialog_annotated_only_adapter" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_only_adapter/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_only_adapter/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --use-adapter True \
    --use-decoder-adapter True \
    --freeze-model True\
    --action-labels True \
    --emotion-labels True \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"  

elif [ "$DATASET" == "empatheticdialog_post_change_cl_supervised" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_post_change_cl_supervised/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_post_change_cl_supervised/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 20 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --use-adapter True \
    --use-decoder-adapter True \
    --freeze-model True\
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --temp-scale 0.1\
    --unfreeze-posterior True \
    --action-labels True \
    --emotion-labels True \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"  

elif [ "$DATASET" == "empatheticdialog_post_change_cl_coherence" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_post_change_cl_coherence/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_post_change_cl_coherence/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 200 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 20 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --use-adapter True\
    --freeze-model True\
    --add-cl-loss True \
    --cl-coherence True \
    --temp-scale 0.5\
    --unfreeze-posterior True \
    --action-labels True \
    --emotion-labels True \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"  


elif [ "$DATASET" == "empatheticdialog_post_change_cl_all" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_post_change_cl_all/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_post_change_cl_all/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 200 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 20 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --use-adapter True\
    --freeze-model True\
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --cl-coherence True \
    --temp-scale 0.5\
    --unfreeze-posterior True \
    --action-labels True \
    --emotion-labels True \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"  

elif [ "$DATASET" == "empatheticdialog_post_change_only_adapter" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_post_change_only_adapter/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_post_change_only_adapter/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 25.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 100 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 20 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --use-adapter True\
    --freeze-model True\
    --unfreeze-posterior True \
    --action-labels True \
    --emotion-labels True \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"  
elif [ "$DATASET" == "empatheticdialog_annotated_cl_style_ende_adapter" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_style_ende_adapter/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_cl_style_ende_adapter/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 16 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 40 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 0.1 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --cl-style True \
    --temp-scale 0.05\
    --use-adapter True\
    --use-decoder-adapter True\
    --freeze-model True\
    --action-labels True \
    --emotion-labels True \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"    

elif [ "$DATASET" == "empatheticdialog_annotated_post_change_ende_adapter" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_post_change_ende_adapter/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_post_change_ende_adapter/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 1000 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 16 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 20 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 0.1 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --use-adapter True\
    --use-decoder-adapter True\
    --freeze-model True\
    --unfreeze-posterior True \
    --action-labels True \
    --emotion-labels True \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"   

elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --action-labels True \
    --emotion-labels True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"      
elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder_cl_style" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder_cl_style --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_style/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_style/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 25 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --cl-style True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --unfreeze-posterior True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"   
elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder_cl_supervised" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder_cl_supervised --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_supervised/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_supervised/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --unfreeze-posterior True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"   
elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder_cl_coherence" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder_cl_coherence --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_coherence/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_coherence/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.1 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-coherence True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --unfreeze-posterior True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"   
elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder_cl_all" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder_cl_all --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_all/checkpoints/${MODE}
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_all/tensorboard/${MODE}
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 25 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --cl-coherence True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --unfreeze-posterior True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"  
elif [ "$DATASET" == "dailydialog_no_smooth" ]; then
  echo '-------- fine-tune on dataset: dailydialog_no_smooth --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/dailydialog
  SAVE_DIR=${DATA_DIR}/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${DATA_DIR}/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 10 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
elif [ "$DATASET" == "empatheticdialog_annotated_no_smooth" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_no_smooth --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${DATA_DIR}/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${DATA_DIR}/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --action-labels True \
    --emotion-labels True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}" 
elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder_no_smooth" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --action-labels True \
    --emotion-labels True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}" 
elif [ "$DATASET" == "empatheticdialog_annotated_adapter_freeze_decoder_no_smooth" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_adapter_freeze_decoder_no_smooth --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_adapter_freeze_decoder/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_adapter_freeze_decoder/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --action-labels True \
    --emotion-labels True \
    --use-adapter True\
    --freeze-model True \
    --unfreeze-posterior True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}" 
elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder_cl_style_no_smooth" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder_cl_style_no_smooth --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_style_t0.5/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_style_t0.5/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --cl-style True \
    --temp-scale 0.5\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --unfreeze-posterior True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"   
elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder_cl_coherence_no_smooth" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder_cl_coherence_no_smooth --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_coherence/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_coherence/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-coherence True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --unfreeze-posterior True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"   
elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder_cl_supervised_no_smooth" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder_cl_supervised_no_smooth --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_supervised/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_supervised/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --unfreeze-posterior True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}" 
elif [ "$DATASET" == "empatheticdialog_annotated_freeze_decoder_cl_all_no_smooth" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_annotated_freeze_decoder_cl_all_no_smooth --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_all/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_all/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --cl-coherence True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --unfreeze-posterior True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"  

elif [ "$DATASET" == "empatheticdialog_freeze_decoder_cl_style_no_smooth_another_post" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_freeze_decoder_cl_style_no_smooth_another_post --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_style/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_style/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --cl-style True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"   
elif [ "$DATASET" == "empatheticdialog_freeze_decoder_cl_coherence_no_smooth_another_post" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_freeze_decoder_cl_coherence_no_smooth_another_post --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_coherence/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_coherence/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-coherence True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"   
elif [ "$DATASET" == "empatheticdialog_freeze_decoder_cl_supervised_no_smooth_another_post" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_freeze_decoder_cl_supervised_no_smooth_another_post --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_supervised/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_supervised/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}" 
elif [ "$DATASET" == "empatheticdialog_freeze_decoder_cl_all_no_smooth_another_post" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_freeze_decoder_cl_all_no_smooth_another_post --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
 SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_all/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_all/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --cl-coherence True \
    --temp-scale 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"  

elif [ "$DATASET" == "empatheticdialog_freeze_decoder_cl_emotion_no_smooth_another_post" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_freeze_decoder_cl_emotion_no_smooth_another_post --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_emotion/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated_freeze_decoder_cl_emotion/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 15 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --cl-coherence True \
    --temp-scale 0.05\
    --use-adapter True\
    --emotion-labels True \
    --freeze-model True \
    --freeze-decoder True \
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"

elif [ "$DATASET" == "empatheticdialog_freeze_decoder_no_smooth" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_freeze_decoder_no_smooth --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/exp/empatheticdialog/adapter_cl_both_scale_100/checkpoints/${MODE}_no_smooth
  TB_LOGDIR=${PROJECT_PATH}/data/exp/empatheticdialog/adapter_cl_both_scale_100/tensorboard/${MODE}_no_smooth
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 30 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 1.0 \
    --target-kl 5.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --nll-weight 1.0 \
    --class-weight 0.0 \
    --add-cl-loss True \
    --cl-emotion-weight 100.0 \
    --cl-action-weight 0.0 \
    --cl-coherence-weight 100.0 \
    --cl-emotion True \
    --cl-coherence True \
    --temp-scale-class 0.5\
    --temp-scale-instance 0.05\
    --emotion-labels True \
    --use-adapter True\
    --freeze-model True \
    --freeze-decoder True \
    --freeze-norm True\
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"


elif [ "$DATASET" == "empatheticdialog_freeze_decoder_cl_supervised_no_smooth_another_post_freeze_norm_class_cl_kl_scaling" ]; then
  echo '-------- fine-tune on dataset: empatheticdialog_freeze_decoder_cl_supervised_no_smooth_another_post_freeze_class_cl_kl_scaling --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate_empathetic
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_annotated
  SAVE_DIR=${PROJECT_PATH}/data/finetune/empatheticdialog_freeze_decoder_cl_supervised_freeze_norm_class_cl_kl_scaling/checkpoints/${MODE}_no_smooth_temp_scaling_final
  TB_LOGDIR=${PROJECT_PATH}/data/finetune/empatheticdialog_freeze_decoder_cl_supervised_freeze_norm_class_cl_kl_scaling/tensorboard/${MODE}_no_smooth_temp_scaling_final
  if [ ! -d  ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
  fi
  if [ ! -d  ${TB_LOGDIR} ]; then
    mkdir -p ${TB_LOGDIR}
  fi
  fairseq-train \
    ${DATA_DIR}/binary \
    --save-interval 1 \
    --fp16 \
    --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
    --lr 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-06 --warmup-updates 400 \
    --criterion $CRITERION --label-smoothing 0.0 \
    --update-freq 32 --max-tokens 4500 --max-sentences 16 \
    --num-workers ${NUM_WORKERS}  \
    --dropout 0.1 --attention-dropout 0.0 --activation-dropout 0.1 --weight-decay 0.01 \
    --encoder-layer-drop 0.0 \
    --save-dir ${SAVE_DIR} \
    --max-epoch 20 \
    --keep-last-epochs 10 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --kl-loss-weight 0.5 \
    --target-kl 10.0 \
    --cls-bow-loss-weight 0.0 \
    --latent-bow-loss-weight 1.0 \
    --masked-lm-loss-weight 0.0 \
    --nll-weight 1.0 \
    --cl-weight 50.0 \
    --class-weight 10.0 \
    --add-cl-loss True \
    --cl-emotion True \
    --cl-action True \
    --temp-scale-class 0.5\
    --temp-scale-instance 0.05\
    --use-adapter True\
    --action-labels True \
    --emotion-labels True \
    --freeze-model True \
    --freeze-decoder True \
    --freeze-norm True\
    --tensorboard-logdir ${TB_LOGDIR} \
    --dataset-impl mmap \
    --empty-cache-freq 64 \
    --seed 1 \
    --skip-invalid-size-inputs-valid-test \
    --distributed-no-spawn \
    --ddp-backend no_c10d \
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
else
#  echo 'dataset not found!'
  echo 'dataset '"$DATASET"' not found!'
  exit 1
fi
