import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
import six

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def check_image_is_valid(image_bin):
    if image_bin is None:
        return False
    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
    if image_buf.size == 0:
        return False
    img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    return True

def create_lmdb_dataset(output_path, root, check_valid=True):
    # 创建LMDB环境
    env = lmdb.open(output_path, map_size=1820000000)  # 设置最大空间1TB
    cache = {}
    cnt = 1

    with open('', 'r', encoding='utf-8') as f:
        utf_list = sorted(list(f.read()))
    gbk_list = utf_list.copy()
    n_samples = 0
    for cls_uni in gbk_list[:]:
        print(cls_uni)
        idx = utf_list.index(cls_uni)
        str_idx = f"{idx:05}"
        cls_folder = os.path.join(root, str_idx)
        # cls_folder = os.path.join(root, cls_uni)
        if not os.path.isdir(cls_folder):
            continue
        img_files = sorted(os.listdir(cls_folder))  # 按数字顺序排序
        for img_file in img_files:
            img_path = os.path.join(cls_folder, img_file)
            with open(img_path, 'rb') as f:
                image_bin = f.read()

            if check_valid:
                if not check_image_is_valid(image_bin):
                    print(f'{img_path} is not a valid image.')
                    continue

            image_key = 'image-%09d' % cnt
            label_key = 'label-%09d' % cnt

            cache[image_key] = image_bin
            cache[label_key] = cls_uni.encode()

            if cnt % 1000 == 0:
                write_cache(env, cache)
                cache = {}

            cnt += 1
            n_samples += 1

    # 写入剩余数据
    cache['num-samples'] = str(n_samples).encode()
    write_cache(env, cache)
    print(f'Created LMDB dataset with {n_samples} samples')

if __name__ == "__main__":
    # root = 'C:\\Users\\chy\\data\\HWDB\\train'  # 修改为你的根目录路径
    root = ''
    output_path = ''  # 输出的lmdb路径
    create_lmdb_dataset(output_path, root)