import os
import numpy as np
import struct
from PIL import Image

# 路径为存放数据集解压后的.gnt文件
train_data_dir = 'C:\\Users\\chy\\data\\HWDB_zip\\1.0test'
test_data_dir = 'C:\\Users\\chy\\data\\HWDB_zip\\1.1test'


def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            yield image, tagcode

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode


# char_set = set()
# for _, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
#     tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
#     char_set.add(tagcode_unicode)
# char_list = list(char_set)
# char_dict = dict(zip(sorted(char_list), range(len(char_list))))
# print(len(char_dict))
# print("char_dict=", char_dict)

import pickle

# f = open('char_dict_10', 'wb')
# pickle.dump(char_dict, f)
# f.close()
with open('char_dict', 'rb') as f:
    char_dict = pickle.load(f)

train_counter = 0
test_counter = 0
for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    im = Image.fromarray(image)
# 路径为data文件夹下的子文件夹，train为存放训练集.png的文件夹
    dir_name = 'C:\\Users\\chy\\data\\HWDB\\train\\' + '%0.5d' % char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    im.convert('RGB').save(dir_name + '\\1.0_tst_' + str(train_counter) + '.png')
    print("train_counter=", train_counter)
    train_counter += 1
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    im = Image.fromarray(image)
# 路径为data文件夹下的子文件夹，test为存放测试集.png的文件夹
    dir_name = 'C:\\Users\\chy\\data\\HWDB\\train\\' + '%0.5d' % char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    im.convert('RGB').save(dir_name + '\\1.1_tst_' + str(test_counter) + '.png')
    print("test_counter=", test_counter)
    test_counter += 1

print('done')
