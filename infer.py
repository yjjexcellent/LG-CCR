from cfgs.CCDT_config import config
from model.CCDT import Clip_Codebook
from utils.pretrain_utils import *
import os
from PIL import Image
import numpy as np
from datasets.dataset import TrainDataset
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

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
    
    model.load_state_dict(torch.load('', map_location='cuda:0'), strict=False)

    val_num = config['val_class']
    val_char_list = char_list[-val_num:]
    val_dataset = TrainDataset(config['val_dataset'], val_num, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=12)

    global best_acc, global_acc_list, local_acc_list, acc_list
    global_acc_list = []
    local_acc_list = []
    acc_list = []
    best_acc = -1

    def val(model):
        model.eval()

        if val_num == 0:
            return
        tmp_text = convert(val_char_list).cuda()
        text_global_features = []
        text_local_features = []
        text_features = []
        iters = (val_num - 1) // 100
        with torch.no_grad():
            for i in range(iters + 1):
                s = i * 100
                e = (i + 1) * 100
                if e > val_num:
                    e = val_num
                text_global_features_tmp, text_local_features_tmp = model.val_encode_text(tmp_text[s:e])
                # text_global_features_tmp, text_local_features_tmp, text_features_tmp = model.val_encode_text(tmp_text[s:e])
                text_global_features.append(text_global_features_tmp)
                text_local_features.append(text_local_features_tmp)
                # text_features.append(text_features_tmp)
            text_global_features = torch.cat(text_global_features, dim=0)
            text_local_features = torch.cat(text_local_features, dim=0)
            # text_features = torch.cat(text_features, dim=0)

            global_correct = 0
            global_total = 0
            local_correct = 0
            local_total = 0
            correct = 0
            total = 0
            for image, label_idx in val_loader:
                label_unicode = [char_list[l] for l in label_idx]
                image = image.cuda()
                # print(image.shape)
                logits_global_per_image, logits_local_per_image, logits_per_image = model.val_encode_data(image, text_global_features, text_local_features)
                # logits_global_per_image, logits_local_per_image, logits_per_image = model.val_encode_data(image, text_global_features, text_local_features, text_features)
                global_probs, global_index = logits_global_per_image.softmax(dim=-1).max(dim=-1)
                local_probs, local_index = logits_local_per_image.softmax(dim=-1).max(dim=-1)
                probs, index = logits_per_image.softmax(dim=-1).max(dim=-1)
                for i in range(len(label_unicode)):
                    if val_char_list[global_index[i]] == label_unicode[i]:
                        global_correct += 1
                    global_total += 1
                for i in range(len(label_unicode)):
                    if val_char_list[local_index[i]] == label_unicode[i]:
                        local_correct += 1
                    local_total += 1
                for i in range(len(label_unicode)):
                    if val_char_list[index[i]] == label_unicode[i]:
                        correct += 1
                    total += 1
                    
            print("dynamic_acc:", correct/total)
            print("global_acc:", global_correct/global_total)
            print("local_acc:", local_correct/local_total)
                # for i in range(len(label_unicode)):
                #     # if (val_char_list[global_index[i]] == label_unicode[i]) and (global_probs[i]<local_probs[i]).item():
                #     #     image = image.cpu()
                #     #     image = image.squeeze(0)
                #     #     numpy_array = image.permute(1, 2, 0).numpy()
                #     #     image = Image.fromarray((numpy_array * 255).astype(np.uint8))
                #     #     image.save('./glo_loc_vis/loc_'+str(global_probs[i].item())+'_'+str(local_probs[i].item())+'.png')
                #     if (val_char_list[global_index[i]] == label_unicode[i]) and (val_char_list[local_index[i]] != label_unicode[i]):
                #         image = image.cpu()
                #         image = image.squeeze(0)
                #         image = (image + 1) / 2.0
                #         numpy_array = image.permute(1, 2, 0).numpy()
                #         image = Image.fromarray((numpy_array * 255).astype(np.uint8))
                #         image.save('./glo_loc_vis/glo_'+str(global_probs[i].item())+'_'+str(local_probs[i].item())+'.png')
            #     for i in range(len(label_unicode)):
            #         if val_char_list[global_index[i]] == label_unicode[i]:
            #             global_correct += 1
            #         global_total += 1
            #     for i in range(len(label_unicode)):
            #         if val_char_list[local_index[i]] == label_unicode[i]:
            #             local_correct += 1
            #         local_total += 1
            #     for i in range(len(label_unicode)):
            #         if (global_probs[i]<local_probs[i]).item() & (local_probs[i]>0.9).item():
            #             if val_char_list[local_index[i]] == label_unicode[i]:
            #                 correct += 1     
            #         else:
            #             if val_char_list[global_index[i]] == label_unicode[i]:
            #                 correct += 1              
            #         total += 1
            # print("dynamic_acc:", correct/total)
            # print("global_acc:", global_correct/global_total)
            # print("local_acc:", local_correct/local_total)
                

    val(model)


if __name__ == "__main__":
    main()




#     for i in range(len(label_unicode)):
            #         if val_char_list[global_index[i]] == label_unicode[i]:
            #             global_correct += 1
            #         global_total += 1
            #     for i in range(len(label_unicode)):
            #         if val_char_list[local_index[i]] == label_unicode[i]:
            #             local_correct += 1
            #         local_total += 1
            #     for i in range(len(label_unicode)):
            #         if (global_probs[i]<local_probs[i]).item() & (local_probs[i]>0).item():
            #             if val_char_list[local_index[i]] == label_unicode[i]:
            #                 correct += 1     
            #         else:
            #             if val_char_list[global_index[i]] == label_unicode[i]:
            #                 correct += 1              
            #         total += 1
            # print("dynamic_acc:", correct/total)
            # print("global_acc:", global_correct/global_total)
            # print("local_acc:", local_correct/local_total)

