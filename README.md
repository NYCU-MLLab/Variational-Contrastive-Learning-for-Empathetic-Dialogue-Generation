# Variational-Contrastive-Learning-for-Empathetic-Dialogue-Generation

### Requirements

- python==3.7
- torch==1.12.1
- tensorboardX==1.7
- pytorch_transformers
- sklearn
- nltk==3.5

```shell
sudo apt install default-jdk
curl https://install.meteor.com/ | sh

pip install -r requirements.txt
```

#### Preprocess

```shell
bash preprocess/process_empathetic.sh
```

#### Binarization

```shell
bash preprocess/binarize_empathetic.sh
```

#### Training

the script `train.sh` has three parameters, namely `p`, `t` and `d`.

- `p`: pretrained model **p**ath
- `t`: pretrained model **t**ype (`dialogved_standard`, `dialogved_large` or `dialogved_seq2seq`)
- `d`: fine-tuned **d**ataset (`dailydialog`, 'empatheticdialog_freeze_decoder_no_smooth')

```shell
bash train.sh -p /remote-home/models/dialogved_standard.pt -t dialogved_large_no_ngram_attn_pre -d empatheticdialog_annotated
```

#### Inference

the script `infer.sh` has two parameters, namely `d` and `s`.

- `d`: fine-tuned **d**ataset (`empatheticdialog_annotated`)
- `s`: decoding **s**trategy (`greedy`, `beam` or `sampling`)
- `t`: **t**rained loss (`empatheticdialog_freeze_decoder_no_smooth`)
- `m`: **m**odel tpye (`dialogved_large_no_ngram_attn_pre`)
- `c`: **c**heckpoint  (`checkpoint1`,`checkpoint_best`)

```shell
bash infer.sh -d empatheticdialog_annotated -s beam -t 
```

#### Evaluation

the script `eval.sh` has one parameter, namely `d`.

- `d`: fine-tuned **d**ataset (`empatheticdialog_annotated`)
- `t`: **t**rained loss (`empatheticdialog_freeze_decoder_no_smooth`)
- `m`: **m**odel tpye (`dialogved_large_no_ngram_attn_pre`)
- `s`: decoding **s**trategy (`greedy`, `beam` or `sampling`)
- `c`: **c**heckpoint  (`checkpoint1`,`checkpoint_best`)

```shell
bash eval.sh -d dailydialog
```
