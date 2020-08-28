import codecs
from pypinyin import Style, pinyin
import logging

logging.basicConfig(level=logging.INFO)


with codecs.open('/home/oshindo/rnn-transducer/char.txt', 'r', encoding='utf-8') as fid:
    for line in fid:
        utt = []
        contents = []
        parts = line.strip().split(' ')
        utt_id = "".join(parts[0])
        text = "".join(parts[1:])
        pyin = pinyin(text, style=Style.TONE3)
        for unit in pyin:
            utt.append("".join(unit).strip('[]'))
        contents.append(utt_id + " " + " ".join(utt).strip('[]'))
        with open('/home/oshindo/rnn-transducer/pinyin.txt', 'a') as f:
            f.write("".join(contents) + '\n')




def get_char():
    char = []
    list = []
    i = 0
    with codecs.open('charvocab2.txt', 'r', encoding='utf-8') as fid:
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


#with open('thchs30_label/thchs30_char_table.txt', 'w') as fid:
#    for line in get_char():
#        fid.write(str(line)+'\n')

