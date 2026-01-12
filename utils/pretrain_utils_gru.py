import torch
import time
import shutil
import os
import copy
import pickle

from cfgs.train_config_gru import config


stru_dict = {'⿱': 3, '⿰': 1, '⿺': 97, '⿹': 17, '⿸': 13, '⿶': 287, '⿷': 233, '⿵': 49, '⿴': 138, '⿻': 35}
j = 0

with open(config['char_dict'], 'rb') as f:
    char_dict = pickle.load(f)
    char_list = sorted(char_dict.keys())

alp2num = {}
alphabet_character = []
alphabet_character.append('PAD')
lines = open(config['alphabet_path'], 'r', encoding='utf-8').readlines()
for line in lines:
    alphabet_character.append(line.strip('\n'))
alphabet_character.append('$')
for index, char in enumerate(alphabet_character):
    alp2num[char] = index
    if char in stru_dict.keys():
        stru_dict[char] = index

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


def convert_rnn(label):
    stru_list = list(stru_dict.keys())
    batch = len(label)

    text_tensor = torch.zeros(batch, config['max_len']).long().cuda()
    tree_tensor = torch.zeros(batch, config['max_len'], 13).long().cuda()

    class TreeNode:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    def build_tree(preorder):
        if not preorder:
            return None
        root_val = preorder.pop(0)
        root = TreeNode(root_val)
        if root_val in stru_list:
            root.left = build_tree(preorder)
            root.right = build_tree(preorder)
        return root

    for i in range(batch):
        global j
        j = 0
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])
        length = len(r_tmp)
        preorder = list(r_tmp)
        root = build_tree(preorder)

        def dfs(node, parent_val, is_left):
            if not node:
                return
            if node.val in stru_list:
                dfs(node.left, node.val, True)
                dfs(node.right, node.val, False)
            else:
                global j
                text_tensor[i][j] = alp2num[node.val]
                tree_tensor[i][j][stru_list.index(parent_val)] = 1
                if is_left:
                    tree_tensor[i][j][-2] = 1
                else:
                    tree_tensor[i][j][-1] = 1
                j += 1

        if length > 1:
            dfs(root, None, None)
        else:
            text_tensor[i][0] = alp2num[r_tmp[0]]
            tree_tensor[i][0][-3] = 1

        text_tensor[i][j] = alp2num['$']

    return text_tensor, tree_tensor
