import os

f3755 = open('ctw_char_3650.txt', 'r', encoding='utf-8')
words_list = f3755.read()

file = open('ctw_decompose_3650_rnn.txt', 'w+', encoding='utf-8')

first = True
lines = open('decompose_27533_benchmark_rnn.txt', 'r', encoding='utf-8').readlines()
for line in lines[:27533]:
    line = line.split('\n')[0]
    word = line.split(':')[0]
    if word in words_list:
        if first:
            file.write(line)
            first = False
        else:
            file.write('\n' + line)
