import codecs
from rnnt.utils import init_logger


def get_word():
    word = []
    with codecs.open('sgns.context.word-character.char1-1.dynwin5.thr10.neg5.dim300.iter5', 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split(' ')
            word.append(parts[0])
    return word


#with open('wordvocab2.txt', 'w') as fid:
#    for line in get_word():
#        fid.write(str(line)+'\n')

list = get_word()
#list = ['而', '楼市', '抑制', '作用', '成交']
logger = init_logger('getvocab.log')
with codecs.open('thchs30_word_table.txt', 'r', encoding='utf-8') as fid:
    f = 0
    for line in fid:
        parts = line.strip()
        if parts in list:
            continue
        else:
            f = f + 1
            logger.info('char:%s, num:%d' % (line, f))
