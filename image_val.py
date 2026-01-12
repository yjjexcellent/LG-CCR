import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from tqdm import trange
import pickle
from PIL import Image

from loss.loss import ClipInfoCELoss
from datasets.dataset import resizeNormalize
from model.pretrain_model_rnn import Clip_Codebook
from datasets.dataset import TrainDataset
from utils.pretrain_utils_rnn import *
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

    val_num = config['val_class']
    val_char_list = char_list[-val_num:]
    val_dataset = TrainDataset(config['val_dataset'], val_num, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=12)

    global local_acc_list, global_acc_list, best_acc
    local_acc_list = []
    global_acc_list = []
    best_acc = -1

    # input val_char_list  output 1000_img_tensor
    gt_folder = ''
    tmp_image = []
    transform = resizeNormalize((128, 128))
    for char in val_char_list:
        img = Image.open(os.path.join(gt_folder, char + '.png'))
        img = img.convert("RGB")
        tmp_image.append(transform(img))
    tmp_image = torch.stack(tmp_image)
    tmp_image = tmp_image.cuda()

    def val(model):
        model.eval()

        if val_num == 0:
            return
        image_local_features = []
        image_global_features = []
        iters = (val_num - 1) // 100
        with torch.no_grad():
            for i in range(iters + 1):
                s = i * 100
                e = (i + 1) * 100
                if e > val_num:
                    e = val_num
                image_local_features_tmp, image_global_features_tmp = model.val_gt_image(tmp_image[s:e])
                image_local_features.append(image_local_features_tmp)
                image_global_features.append(image_global_features_tmp)
            image_local_features = torch.cat(image_local_features, dim=0)
            image_global_features = torch.cat(image_global_features, dim=0)

            local_correct = 0
            global_correct = 0
            local_total = 0
            global_total = 0
            for image, label_idx in val_loader:
                label_unicode = [char_list[l] for l in label_idx]
                image = image.cuda()
                logits_local_per_image, logits_global_per_image = model.val_encode_data(image, image_local_features,
                                                                                        image_global_features)
                local_probs, local_index = logits_local_per_image.softmax(dim=-1).max(dim=-1)
                global_probs, global_index = logits_global_per_image.softmax(dim=-1).max(dim=-1)
                for i in range(len(label_unicode)):
                    if val_char_list[local_index[i]] == label_unicode[i]:
                        local_correct += 1
                    local_total += 1
                for i in range(len(label_unicode)):
                    if val_char_list[global_index[i]] == label_unicode[i]:
                        global_correct += 1
                    global_total += 1
            local_current_acc = local_correct / local_total
            global_current_acc = global_correct / global_total
            global local_acc_list, global_acc_list
            local_acc_list.append(local_current_acc)
            global_acc_list.append(global_current_acc)
            global best_acc
            if local_current_acc > best_acc:
                best_acc = local_current_acc
            if global_current_acc > best_acc:
                best_acc = global_current_acc

            print(
                f"local_Acc: {local_current_acc:.5f}\tglobal_Acc: {global_current_acc:.5f}\tBest_Acc: {best_acc:.5f}\n")

    model.load_state_dict(torch.load('', map_location='cuda:0'))
    val(model)

if __name__ == '__main__':
    main()
