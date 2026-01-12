import os

file1 = open('unseen_char_1000.txt', 'w+', encoding='utf-8')
file2 = open('unseen_decompose_1000.txt', 'w+', encoding='utf-8')

curr_char_list = open('char_3755.txt', 'r', encoding='utf-8').read()
alphabet_file = open('radical_alphabet_3755.txt', 'r', encoding='utf-8').readlines()
curr_alphabet_list = []
for a in alphabet_file:
    curr_alphabet_list.append(a.split('\n')[0])
total_lines = open('decompose_27533_benchmark.txt', 'r', encoding='utf-8').readlines()

first = True
for line in total_lines:
    line = line.split('\n')[0]
    char = line.split(':')[0]
    seq = line.split(':')[1]

    if char in curr_char_list:
        continue

    flag = False
    if len(seq) > 1:
        r_list = seq.split(' ')
        for r in r_list:
            if r not in curr_alphabet_list:
                flag = True
                continue
    else:
        if seq not in curr_alphabet_list:
            continue

    if flag:
        continue

    file1.write(char)

    if first:
        file2.write(line)
        first = False
    else:
        file2.write('\n' + line)

