from transformers import BertModel, BertForPreTraining, BertTokenizer, BertForMaskedLM, AdamW
import torch.nn.functional as F
from rnnt.dataset import AudioDataset
import logging
from torch import nn
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

contents = []
all = {}
idx = 0
with open('/home/oshindo/rnn-transducer/thchs30_char_embedding.txt', 'r', encoding='utf-8') as fid:
    for line in fid:
        parts = line.strip().split()
        contents.append(parts[0])
        all["".join(parts[0])] = " ".join(parts[1:])

idx = 0
with open('/home/oshindo/rnn-transducer/thchs30_label/thchs30_train_char_table.txt', 'r', encoding='utf-8') as fid:
    for line in fid:
        parts = line.strip().split()
        if parts[0] in contents:
            logger.info('successfully get: num = %d ' % (idx))
            with open('thchs30_train_char_embedding.txt', 'a') as fid:
                fid.write("".join(parts[0]) + " " + all["".join(parts[0])] + '\n')
            idx = idx + 1





    
