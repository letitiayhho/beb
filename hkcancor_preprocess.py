import xml.etree.ElementTree as ET
import os
import pandas as pd
from random import random
from hanzidentifier import has_chinese

sentences = []
labels = []

hkcancor_path = '../corpora/hkcancor/raw'
w_hkcancorTSV = '../corpora/hkcancor/hkcancor.tsv'

for filename in os.listdir(hkcancor_path):
    file = open(os.path.join(hkcancor_path, filename))
    print("processing: " + filename)
    sentence = []

    while True:
        line = file.readline()
        if has_chinese(line):
            char = line.split('/', 1)[0]
            sentence.append(char[3:])
        elif char == "。" or char == "，":
            sentence.append(char)
        if "</sent_tag>" in line:
            labels.append(0) if random() <= 0.3 else labels.append(1)
            print(''.join(sentence))
            sentences.append(''.join(sentence))
            sentence = []

#for i in range(20):
#    print(labels[i] + '\t' + sentences[i])

#df = {'labels': labels, 'sentences': sentences}

#with open(w_hkcancorTSV, 'w') as write_tsv:
#    write_tsv.write(df.to_tsv(sep='\t', index=False)
