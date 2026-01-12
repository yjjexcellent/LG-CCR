import os
from collections import OrderedDict

stru_dict = {'⿱': 3, '⿰': 1, '⿺': 97, '⿹': 17, '⿸': 13, '⿶': 287, '⿷': 233, '⿵': 49, '⿴': 138, '⿻': 35}
stru_list = list(stru_dict.keys())

dis = {}

f = open('./decompose_3755_rnn_10stru.txt', 'r', encoding='utf-8')
for line in f.readlines():
    line = line.split('\n')[0]
    char = line.split(':')[0]
    raw_text = line.split(':')[1]
    comp_list = raw_text.split(' ')
    radical_text = []
    for comp in comp_list:
        if comp not in stru_list:
            radical_text.append(comp)
    length = len(radical_text)
    if length not in dis.keys():
        dis[length] = 1
    else:
        dis[length] += 1
    if length == 8 or length == 9:
        print(char, comp_list)

print(sorted(dis.items()))

# 矗 ['⿱', '⿱', '十', '⿴', '且', '一', '⿰', '⿱', '十', '⿴', '且', '一', '⿱', '十', '⿴', '且', '一']
# 囊 ['⿱', '⿻', '一', '中', '⿱', '冖', '⿱', '⿱', '⿰', '口', '口', '⿱', '井', '一', '𧘇']
# [(1, 138), (2, 1115), (3, 1370), (4, 727), (5, 331), (6, 62), (7, 10), (8, 1), (9, 1)]