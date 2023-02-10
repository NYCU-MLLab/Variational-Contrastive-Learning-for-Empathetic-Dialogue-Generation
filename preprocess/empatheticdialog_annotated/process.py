import os
import sys
sys.path.append('/home/Thesis/DialogVED')
# sys.path.append('/Users/lemuria_chen/PycharmProjects/DialogVED')

from utils.processor import convert_empathetic_dialog, check


# FINETUNE_PREFIX_PATH = '/remote-home/wchen/project/DialogVED/data/finetune'
FINETUNE_PREFIX_PATH = '/home/Thesis/DialogVED/data/finetune'


ORIGINAL_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'empatheticdialog_annotated/original_data/dialog')
PROCESSED_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'empatheticdialog_annotated/processed/dialog')


convert_empathetic_dialog(
    fin=os.path.join(ORIGINAL_PATH, 'dial.train'),
    src_fout=os.path.join(PROCESSED_PATH, 'train.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'train.tgt')
)
convert_empathetic_dialog(
    fin=os.path.join(ORIGINAL_PATH, 'dial.valid'),
    src_fout=os.path.join(PROCESSED_PATH, 'valid.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'valid.tgt')
)
convert_empathetic_dialog(
    fin=os.path.join(ORIGINAL_PATH, 'dial.test'),
    src_fout=os.path.join(PROCESSED_PATH, 'test.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'test.tgt')
)

check(PROCESSED_PATH, mode='train')
check(PROCESSED_PATH, mode='valid')
check(PROCESSED_PATH, mode='test')
