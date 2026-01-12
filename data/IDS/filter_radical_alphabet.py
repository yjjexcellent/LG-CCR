import os

file = open('ctw_radical_alphabet_3650.txt', 'w+', encoding='utf-8')

alphabet = []
lines = open('ctw_decompose_3650.txt', 'r', encoding='utf-8').readlines()
for line in lines:
    line = line.split('\n')[0]
    seq = line.split(':')[1]
    if len(seq) > 1:
        r_list = seq.split(' ')
        for r in r_list:
            if r not in alphabet:
                alphabet.append(r)
    else:
        if seq not in alphabet:
            alphabet.append(seq)

first = True
for r in alphabet:
    if first:
        file.write(r)
        first = False
    else:
        file.write('\n' + r)
