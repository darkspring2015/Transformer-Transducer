import codecs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_contents():
    contents = []
    with codecs.open('/home/oshindo/rnn-transducer/char.txt', 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split(' ')
            text = " ".join(parts[1:])
            contents.append(text)
    return contents


with open('wordvocab2.txt', 'w') as fid:
    for line in get_contents():
        fid.write(str(line)+'\n')


def get_word():
    word = []
    list = []
    i = 0
    with codecs.open('wordvocab2.txt', 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split(' ')
            for x in parts:
                if x in list:
                    continue
                else:
                    list.append(x)
                    word.append(x + " " + str(i))
                    logger.info('successfully get: num = %d ,char = %s' % (i, x))
                    i += 1
    return word


with open('aishell_tt_char_table.txt', 'w') as fid:
    for line in get_word():
        fid.write(str(line)+'\n')
