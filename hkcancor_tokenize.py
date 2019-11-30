from pickle import dump
from os import listdir
from hanzidentifier import has_chinese

file_list = listdir()
sentences = []

for file_name in file_list:
    file = open('FC-001_v2')
    while True:
        line = file.readline()
        
        
        if "<sent_tag>" in line:
            tokens.append("[CLS]")
        for char in line:
            if has_chinese(char):
                tokens.append(char)
            elif char == "。" or char == "，":
                tokens.append(char)
                tokens.append("[SEP]")
        if not line:
            break
    file.close()
print(tokens[0:100])

file = open('hkcancor_tokens.txt', 'w')
file.write('\n'.join(tokens))
file.close()
