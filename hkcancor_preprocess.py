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
    file = open(filename)
    sentence = []

    while True:
        line = file.readline()
        if "</sent_tag>" in line:
            sentences.append(sentence)
            sentence = []
        if has_chinese(line):
            sentence.append
        for char in line:
            if has_chinese(char):
                sentence.append(char)
        



    while True:
        line = file.readline()
        if "<sent_tag>" in line:
            x = True
            sentence = []
        while x:
            if has_chinese(char)
            sentence.append(
            

            if "</sent_tag>" in line:
                x = False

    
    with open(os.path.join(hkcancor_path, filename)) as fp:
        tree = ET.parse(fp)
        root = tree.getroot()
        for sentence in root.iter('sent_tag'):
            sentences.append(sentence)
            labels.append(0) if random() <= 0.3 else labels.append(1)

for i in range(20):
    print(label[i] + '\t' + sentences[i])

#df = {'labels': labels, 'sentences': sentences}

#with open(w_hkcancorTSV, 'w') as write_tsv:
#    write_tsv.write(df.to_tsv(sep='\t', index=False)
