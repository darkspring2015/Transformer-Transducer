from transformers import BertForPreTraining, BertTokenizer, BertForMaskedLM
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


tokenizer = BertTokenizer.from_pretrained('./wwm/pytorch')
#model = BertForPreTraining.from_pretrained('./wwm/pytorch')
model = BertForMaskedLM.from_pretrained('./wwm/pytorch')

#input_id = tokenizer("我爱北京", re)
input_id = torch.tensor([tokenizer.encode("我爱北", add_special_tokens=True)])
outputs = model(input_id, lm_labels=input_id)
scores = outputs[1]
print(scores)
'''
hidden = model.get_output_embeddings()
embedding = hidden.weight.data

idx2unit = {}
idx = 0
with open('/home/oshindo/rnn-transducer/wwm/pytorch/vocab.txt', 'r', encoding='utf-8') as fid:
    for line in fid:
        parts = line.strip().split()
        idx2unit[idx] = "".join(parts)
        idx = idx + 1

for i in range(embedding.size(0)):
    embed = [str(unit.item()) for unit in embedding[i].flatten()]
    str_embed = " ".join(embed)
    with open('thchs30_char_embedding.txt', 'a') as fid:
        fid.write(idx2unit[i] + " " + str_embed + '\n')
    logger.info('successfully get: num = %d ,char = %s' % (i, idx2unit[i]))
'''




