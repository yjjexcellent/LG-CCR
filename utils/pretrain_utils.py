import torch
import time
import shutil
import os
import copy

from cfgs.CCDT_config import config


alp2num = {}
alphabet_character = []
alphabet_character.append('PAD')
lines = open(config['alphabet_path'], 'r', encoding='utf-8').readlines()
for line in lines:
    alphabet_character.append(line.strip('\n'))
alphabet_character.append('$')
for index, char in enumerate(alphabet_character):
    alp2num[char] = index

dict_file = open(config['decompose_path'], 'r', encoding='utf-8').readlines()
char_radical_dict = {}
for line in dict_file:
    line = line.strip('\n')
    char, r_s = line.split(':')
    char_radical_dict[char] = r_s.split(' ')


def convert(label):
    r_label = []
    batch = len(label)
    for i in range(batch):
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])  # 部首序列
        r_tmp.append('$')   # 后加结束标志
        r_label.append(r_tmp)

    text_tensor = torch.zeros(batch, config['max_len']).long().cuda()   # 空位都是0
    for i in range(batch):
        tmp = r_label[i]
        for j in range(len(tmp)):
            text_tensor[i][j] = alp2num[tmp[j]]
    return text_tensor


def get_alphabet():
    return alphabet_character
