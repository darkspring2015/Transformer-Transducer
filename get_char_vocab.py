import codecs
from rnnt.utils import init_logger
import logging

logging.basicConfig(level=logging.INFO)


def get_contents():
    contents = []
    with codecs.open('/home/oshindo/rnn-transducer/char.txt', 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split(' ')
            utt_id = parts[0]
            utt_id = "".join(utt_id)
            text = parts[1:]
            text = "".join(text)
            text = " ".join(text)
            contents.append(utt_id + " " + text)
    return contents


#with open('/home/oshindo/kaldi/egs/thchs30/s5/data/mfcc/train/chartext.txt', 'w') as fid:
#    for line in get_contents():
#        fid.write(str(line)+'\n')


def get_char():
    char = []
    list = []
    i = 0
    with codecs.open('/home/oshindo/rnn-transducer/char.txt', 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split(' ')
            for x in parts[1:]:
                if x in list:
                    continue
                else:
                    list.append(x)
                    char.append(x + " " + str(i))
                    i += 1
    return char


with open('thchs30_label/thchs30_char_table.txt', 'w') as fid:
    for line in get_char():
        fid.write(str(line)+'\n')

