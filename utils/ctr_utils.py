import torch
import time
import shutil
import os
import copy
import pickle

from cfgs.ctr_config import config
from datasets.lmdbReaderCTR import lmdbDataset, resizeNormalize


alphabet_character_file = open(config['alpha_path'], 'r', encoding='utf-8')
alphabet_character = list(alphabet_character_file.read().strip())
alphabet_character_raw = ['START']

for item in alphabet_character:
    alphabet_character_raw.append(item)

alphabet_character_raw.append('END')
alphabet_character = alphabet_character_raw

alp2num_character = {}

for index, char in enumerate(alphabet_character):
    alp2num_character[char] = index


stru_dict = {'⿱': 3, '⿰': 1, '⿺': 97, '⿹': 17, '⿸': 13, '⿶': 287, '⿷': 233, '⿵': 49, '⿴': 138, '⿻': 35}
j = 0

r2num = {}
alphabet_radical = []
alphabet_radical.append('PAD')
lines = open(config['radical_path'], 'r', encoding='utf-8').readlines()
for line in lines:
    alphabet_radical.append(line.strip('\n'))
alphabet_radical.append('$')
for index, char in enumerate(alphabet_radical):
    r2num[char] = index
    if char in stru_dict.keys():
        stru_dict[char] = index

dict_file = open(config['decompose_path'], 'r', encoding='utf-8').readlines()
char_radical_dict = {}
for line in dict_file:
    line = line.strip('\n')
    char, r_s = line.split(':')
    char_radical_dict[char] = r_s.split(' ')


def get_data_package():
    train_dataset = []
    for dataset_root in config['train_dataset'].split(','):
        dataset = lmdbDataset(dataset_root, resizeNormalize((config['imageW'],config['imageH']), test=False))
        train_dataset.append(dataset)
    train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_total, batch_size=config['batch'], shuffle=True, num_workers=12
    )

    test_dataset = []
    for dataset_root in config['test_dataset'].split(','):
        dataset = lmdbDataset(dataset_root, resizeNormalize((config['imageW'],config['imageH']), test=True))
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=128, shuffle=False, num_workers=12
    )

    return train_dataloader, test_dataloader


def convert(label):
    r_label = []
    batch = len(label)
    for i in range(batch):
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])  # 部首序列
        r_tmp.append('$')   # 后加结束标志
        r_label.append(r_tmp)

    text_tensor = torch.zeros(batch, 20).long().cuda()   # 空位都是0
    for i in range(batch):
        tmp = r_label[i]
        for j in range(len(tmp)):
            text_tensor[i][j] = r2num[tmp[j]]
    return text_tensor


def get_alphabet():
    return alphabet_character


def get_radical_alphabet():
    return alphabet_radical


def convert_rnn(label):
    stru_list = list(stru_dict.keys())
    batch = len(label)

    text_tensor = torch.zeros(batch, 20).long().cuda()
    # tree_tensor = torch.zeros(batch, 20, 6, 13).long().cuda()
    tree_tensor = torch.zeros(batch, 20, 9, 13).long().cuda()

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
                parent_val.append(node.val)
                is_left.append(True)
                dfs(node.left, parent_val, is_left)
                is_left[-1] = False
                dfs(node.right, parent_val, is_left)
                parent_val.pop()
                is_left.pop()
            else:
                global j
                text_tensor[i][j] = r2num[node.val]
                for k in range(len(parent_val)):
                    tree_tensor[i][j][k][stru_list.index(parent_val[k])] = 1
                    if is_left[k]:
                        tree_tensor[i][j][k][-2] = 1
                    else:
                        tree_tensor[i][j][k][-1] = 1
                j += 1

        if length > 1:
            dfs(root, [], [])
        else:
            text_tensor[i][0] = r2num[r_tmp[0]]
            j += 1
            tree_tensor[i][0][0][-3] = 1

        text_tensor[i][j] = r2num['$']

    return text_tensor, tree_tensor


def converter(label):
    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            try:
                text_input[i][j + 1] = alp2num[label[i][j]]
            except:
                print(label[i])

    # _text_tensor = torch.zeros(batch, max_length, 20).long().cuda()
    # _tree_tensor = torch.zeros(batch, max_length, 20, 6, 13).long().cuda()
    _text_tensor_list = []
    _tree_tensor_list = []

    for i in range(batch):
        text_tensor, tree_tensor = convert_rnn(label[i][:-1])
        _text_tensor_list.append(text_tensor)
        _tree_tensor_list.append(tree_tensor)

        # for j in range(len(label[i]) - 1):
        #     _text_tensor[i][j] = text_tensor[j]
        #     _tree_tensor[i][j] = tree_tensor[j]
    _text_tensor = torch.cat(_text_tensor_list, dim=0)
    _tree_tensor = torch.cat(_tree_tensor_list, dim=0)
    # _text_tensor = _text_tensor.view(batch * max_length, 20)
    # _tree_tensor = _tree_tensor.view(batch * max_length, 20, 6, 13)

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i])-1):
                text_all[start + j] = alp2num['END']
            else:
                text_all[start + j] = alp2num[label[i][j]]
        start += len(label[i])

    return length, text_input, text_all, string_label, _text_tensor, _tree_tensor


def tensor2str(tensor):
    alphabet = get_alphabet()
    string = ""
    for i in tensor:
        if i == (len(alphabet)-1):
            continue
        string += alphabet[i]
    return string


def must_in_screen():
    text = os.popen('echo $STY').readlines()
    string = ''
    for line in text:
        string += line
    if len(string.strip()) == 0:
        print("run in the screen!")
        exit(0)