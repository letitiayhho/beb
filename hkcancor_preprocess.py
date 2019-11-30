import xml.etree.ElementTree
import os
import pandas as pd
from random import random

sentences = []
labels = []

hkcancor_path = '../corpora/hkcancor/raw'
w_hkcancorTSV = '../corpora/hkcancor/hkcancor.tsv'

for filename in os.listdir(hkcancor_path):
    with open(os.path.join(hkcancor_path, filename)) as fp:
        tree = ElementTree.parse(fp)
        root = tree.getroot()
        for sentence in root.iter('sent_tag'):
            sentences.append(sentence)
            labels.append(0) if random() <= 0.3 else labels.append(1)

for i in range(20):
    print(label[i] + '\t' + sentences[i])

#df = {'labels': labels, 'sentences': sentences}

#with open(w_hkcancorTSV, 'w') as write_tsv:
#    write_tsv.write(df.to_tsv(sep='\t', index=False)
