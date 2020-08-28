import codecs
import logging

def get_word_list():
    word_list = []
    with codecs.open('thchs30_label/thchs30_word_table.txt', 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split()
            #parts = parts[0]
            parts = "".join(parts)
            word_list.append(parts)
    return word_list


target = get_word_list()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
def get_embedding():
    contents = []
    i = 1

    with codecs.open('Tencent_AILab_ChineseEmbedding.txt', 'r', encoding='utf-8') as fid:

        for line in fid:
            parts = line.strip().split(' ')
            unit = parts[0]
            #embedding = [float(i) for i in parts[1:]]

            if unit in target:
                contents.append(unit + " " + " ".join(parts[1:]))
                logger.info('successfully get: num = %d ,char = %s' % (i, parts[0]))
                i = i + 1
                if i == 8874:
                    return contents
            else:
                continue
    return contents



with open('thchs30_tencent_word_embedding', 'w') as fid:
    for line in get_embedding():
        fid.write(str(line)+'\n')

