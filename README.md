# Variational-Contrastive-Learning-for-Empathetic-Dialogue-Generation

### Requirements

- python==3.7
- torch==1.12.1
- fairseq==0.9.0
- tensorboardX==1.7
- pytorch_transformers
- scikit-learn
- nltk==3.5

```shell
apt install pip
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt install python3.7-dev

ln -sf /usr/bin/python3.7 /usr/bin/python3
apt install python3.7-distutils

sudo apt install default-jdk
curl https://install.meteor.com/ | sh

pip install -r requirements.txt
```
If you don't install cuda, you can install with following
```
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

Since some errors occur when using fairseq==0.9.0 with torch==1.12.1, we have fixed these errors in fairseq_fixed.

Please operate as following
```
rm -r /usr/local/lib/python3.7/dist-packages/fairseq
cp -a /home/Variational-Contrastive-Learning-for-Empathetic-Dialogue-Generation/fairseq_fixed /usr/local/lib/python3.7/dist-packages/fairseq
```


#### Preprocess

```shell
bash preprocess/process_empathetic.sh
```

#### Binarization

```shell
bash preprocess/binarize_empathetic.sh
```


Here is pre-trained model we use, we do some modification different from dialogVED, and finetune it on dailydialog.

[Pre-trained model](https://drive.google.com/file/d/1VqB_1x9FCJisCW3e8sHG0e3uGVrVJR4H/view?usp=sharing)




#### Training

the script `train.sh` has three parameters, namely `p`, `t` and `d`.

- `p`: pretrained model **p**ath (`/home/dailydialog_ved_large.pt`)
- `t`: pretrained model **t**ype (`dialogved_large_no_ngram_attn_pre`)
- `d`: fine-tuned **d**ataset ('empatheticdialog_freeze_decoder_no_smooth')

```shell
bash train.sh -p /home/dailydialog_ved_large.pt -t dialogved_large_no_ngram_attn_pre -d empatheticdialog_freeze_decoder_no_smooth
```

#### Inference

the script `infer.sh` has two parameters, namely `d` and `s`.

- `d`: target **d**ataset (`empatheticdialog_annotated`)
- `s`: decoding **s**trategy (`greedy`, `beam` or `sampling`)
- `t`: **t**rained loss (`empatheticdialog_freeze_decoder_no_smooth`)
- `m`: **m**odel tpye (`dialogved_large_no_ngram_attn_pre`)
- `c`: **c**heckpoint  (`checkpoint1`,`checkpoint_best`)

```shell
bash infer.sh -d empatheticdialog_annotated -s beam -t empatheticdialog_freeze_decoder_no_smooth -m ved_large_no_ngram_attn_pre_no_smooth -c checkpoint_best
```

#### Evaluation

the script `eval.sh` has one parameter, namely `d`.

- `d`: target **d**ataset (`empatheticdialog_annotated`)
- `t`: **t**rained loss (`empatheticdialog_freeze_decoder_no_smooth`)
- `m`: **m**odel tpye (`dialogved_large_no_ngram_attn_pre`)
- `s`: decoding **s**trategy (`greedy`, `beam` or `sampling`)
- `c`: **c**heckpoint  (`checkpoint1`,`checkpoint_best`)

```shell
bash eval.sh -d empatheticdialog_annotated -t empatheticdialog_freeze_decoder_no_smooth -m ved_large_no_ngram_attn_pre_no_smooth -s beam -c checkpoint_best
```
