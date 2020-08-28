from transformers import BertModel, BertTokenizer
import logging

logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained("../wwm/data/aishell_word_table.txt")
model = BertModel.from_pretrained("../wwm/data")


# 语料
def get_contents():
    contents = []
    with open('../aishell_label/aishell_word_text.txt', 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split(' ')
            utt_id = parts[0]
            text = parts[1:]
            contents.append(text)
    return contents


seq = get_contents()
tokens = tokenizer.tokenize(seq)
ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

# 词典

# 微调


