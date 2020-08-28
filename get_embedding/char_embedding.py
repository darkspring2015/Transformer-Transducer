from gensim.models import fasttext
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import logging


def get_contents():
    contents = []
    with open('../thchs30_label/thchs30_word_text.txt', 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split(' ')
            utt_id = parts[0]
            text = parts[1:]
            text = "".join(text)
            contents.append(text)
    return contents


seq = get_contents()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#model.build_vocab(seq)
#model.train(seq, total_examp
model = Word2Vec(seq, min_count=1)
model.save('thchs30_char.model')
model.wv.save_word2vec_format("thchs30_char.txt")
