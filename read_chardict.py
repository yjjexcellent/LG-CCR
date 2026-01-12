import pickle

# 打开文件以读取二进制数据
with open('char_dict', 'rb') as f:
    char_dict_loaded = pickle.load(f)

# 打印加载的char_dict
print("Loaded char_dict:", char_dict_loaded)