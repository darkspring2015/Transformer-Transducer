import codecs
import torch

idx2unit = {}
idx = 0
with codecs.open('thchs30_label/thchs30_phone_table.txt', 'r', encoding='utf-8') as fid:
    for line in fid:
        parts = line.strip().split()
        unit = parts[0]
        idx2unit[idx] = unit
        idx = idx + 1


def decode(seqs):
    decode_seq = []
    for idx in seqs:
        if int(idx) in idx2unit:
            decode_seq.append(idx2unit[int(idx)])
        else:
            decode_seq.append('<blk>')
    return decode_seq


with codecs.open('decode.txt', 'r', encoding='utf-8') as fid:
    for line in fid:
        decode_seq = []
        parts = line.strip('[]\n').split(',')
        for idx in parts:
            if int(idx) in idx2unit:
                decode_seq.append(idx2unit[int(idx)])
        with open('dev.txt', 'a') as f:
            f.write('[' + " ".join(decode_seq) + ']' + '\n')






