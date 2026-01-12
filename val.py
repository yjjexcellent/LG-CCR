import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from tqdm import trange
import pickle
import torch.nn.functional as F

from loss.loss import ClipInfoCELoss
from model.pretrain_model_spaatt import Clip_Codebook
from datasets.dataset import TrainDataset
from utils.pretrain_utils_radicalrnn import *
from cfgs.train_config_rnn import config


def main():
    with open(config['char_dict'], 'rb') as f:
        char_dict = pickle.load(f)
        char_list = sorted(char_dict.keys())

    alphabet = get_alphabet()
    model = Clip_Codebook(vision_input_resolution=config['vision_input_resolution'],
                          vision_patch_size=config['vision_patch_size'],
                          vision_width=config['vision_width'],
                          vision_layers=config['vision_layers'],
                          vision_heads=config['vision_heads'],
                          vision_output_dim=config['vision_output_dim'],
                          text_embed_dim=config['text_embed_dim'],
                          text_context_length=config['max_len'],
                          text_vocab_size=len(alphabet),
                          text_width=config['text_width'],
                          text_heads=config['text_heads'],
                          text_layers=config['text_layers'],
                          codebook_num=config['codebook_num'],
                          codebook_dim=config['codebook_dim'],
                          temperature=config['temperature'],
                          is_threshold=config['is_threshold']).cuda()

    val_num = 500
    val_char_list = char_list[:500]
    val_dataset = TrainDataset(config['val_dataset'], val_num, 'train')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2880, shuffle=False, num_workers=12)

    global global_acc_list, best_acc
    global_acc_list = []
    best_acc = -1

    tmp_text, tmp_stru = convert_rnn(val_char_list)
    tmp_vec = tmp_stru
    eos_indices = tmp_text.argmax(dim=1)
    ban_list = []
    for i, number in enumerate(eos_indices):
        if number == 1:
            ban_list.append(val_char_list[i])

    def val(model):
        model.eval()

        if val_num == 0:
            return
        text_global_features = []
        iters = (val_num - 1) // 100
        with torch.no_grad():
            for i in range(iters + 1):
                s = i * 100
                e = (i + 1) * 100
                if e > val_num:
                    e = val_num
                text_global_features_tmp = model.val_encode_text(tmp_text[s:e], tmp_vec[s:e])
                text_global_features.append(text_global_features_tmp)
            text_global_features = torch.cat(text_global_features, dim=0)

            global_correct = 0
            global_total = 0
            for image, label_idx in val_loader:
                label_unicode = [char_list[l] for l in label_idx]
                image = image.cuda()
                logits_global_per_image = model.val_encode_data(image, text_global_features)
                global_probs, global_index = logits_global_per_image.softmax(dim=-1).max(dim=-1)
                for i in range(len(label_unicode)):
                    # if label_unicode[i] in ban_list:
                    #     continue
                    if val_char_list[global_index[i]] == label_unicode[i]:
                        global_correct += 1
                    global_total += 1
            global_current_acc = global_correct / global_total
            global global_acc_list
            global_acc_list.append(global_current_acc)
            global best_acc
            if global_current_acc > best_acc:
                best_acc = global_current_acc

            print(f"global_Acc: {global_current_acc:.5f}\tBest_Acc: {best_acc:.5f}\n")

    for i in range(16, 17):
        model.load_state_dict(torch.load('', map_location='cuda:0'))
        val(model)


if __name__ == '__main__':
    main()