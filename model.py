import numpy as np
import torch
import math
from torch import nn
from collections import OrderedDict
import os
# from model.sparsemax import Sparsemax



class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, codebook_dim=256):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.codebook_dim = codebook_dim
        self.q_map = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, codebook_dim),
            nn.GELU(),
            nn.LayerNorm(codebook_dim),
            nn.Linear(codebook_dim, codebook_dim)
        )
        self.att_activation = nn.Softmax(dim=-1)

        self.map_fusion = nn.Linear(codebook_dim, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # def attention(self, x: torch.Tensor):
    #     # 修改前：need_weights=False
    #     # 修改后：
    #     if self.attn_mask != None:
    #         self.attn_mask = self.attn_mask.cuda()
    #     attn_output, attn_weights = self.attn(
    #         x, x, x,
    #         need_weights=True,  # 强制返回权重
    #         average_attn_weights=False,
    #         attn_mask=self.attn_mask
    #     )
    #     return attn_output  # 保持原有输出格式

    def forward(self, inputs):
        x, codebook, local_list, mask, eos = inputs
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        if mask is None:
            # vis
            local_tokens = x.permute(1, 0, 2)[:, 1:, :]  # (B, N-1, C)
        else:
            # txt
            local_tokens = x.permute(1, 0, 2)

        q = self.q_map(local_tokens)  # [batch, patch_token, codebook_dim] [b, 64, 256]
        # q = local_tokens
        k = codebook
        k = k.unsqueeze(0)  # [1, codebook_num, codebook_dim]
        k = k.transpose(2, 1)  # [1, 512, 200]

        # print(q.shape,k.shape)  # [batch, token_num, code_num] [b, 64, 200]
        inner_dot = torch.matmul(q, k)
        inner_dot = inner_dot / math.sqrt(self.codebook_dim)  # scale dot norm

        # text mask
        if mask is not None:
            txt_mask = (mask == 0) * 1  # 0 --> 1, inf --> 0
            inner_dot = inner_dot * txt_mask.unsqueeze(-1)

        inner_dot_ = inner_dot.max(1)[0]  # [batch, codebook_num]
        att_weight = self.att_activation(inner_dot_)  # [batch, codebook_num]
        local_feature = att_weight @ codebook  # [batch, codebook_dim]
        # 记录每层local特征用于损失
        local_list.append(local_feature)

        # # multi_fusion 【ln（ln（loc） concat glo）】
        # x = x.permute(1, 0, 2)
        # if mask is None:
        #     # vis
        #     cls_token = x[:, 0, :]
        #     ext_local_feature = self.map_fusion(local_feature)
        #     fused_input = torch.cat([cls_token, ext_local_feature], dim=-1)
        #     fused_cls_token = self.fusion(fused_input)
        #     # x[:, 0, :] = fused_cls_token
        #     x = torch.cat([fused_cls_token.unsqueeze(1), x[:, 1:, :]], dim=1)
        # else:
        #     # txt
        #     eos_token = x[torch.arange(x.size(0)), eos, :]
        #     ext_local_feature = self.map_fusion(local_feature)
        #     fused_input = torch.cat([eos_token, ext_local_feature], dim=-1)
        #     fused_cls_token = self.fusion(fused_input)
        #     # x[torch.arange(x.size(0)), eos, :] = fused_cls_token
        #     x = x.clone()  # 避免 inplace 修改
        #     x[torch.arange(x.size(0)), eos, :] = fused_cls_token
        # x = x.permute(1, 0, 2)

        return x, codebook, local_list, mask, eos


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, codebook_dim=512):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, codebook_dim=codebook_dim) for _ in range(layers)])

    def forward(self, x: torch.Tensor, codebook=None, local_list=None, mask=None, eos=None):
        return self.resblocks((x, codebook, local_list, mask, eos))


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 codebook_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, codebook_dim=codebook_dim)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: torch.Tensor, codebook):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        img_local_list = []
        x, _, img_local_list, _, _ = self.transformer(x, codebook, img_local_list)
        x = x.permute(1, 0, 2)  # LND -> NLD

        patch_tokens = x[:, 1:, :]

        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        # return x
        # return patch_tokens
        return x, img_local_list


class TextTransformer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 codebook_dim: int):
        super().__init__()
        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            codebook_dim=codebook_dim
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.positional_embedding.dtype

    def forward(self, text, codebook):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        txt_mask = torch.zeros(len(text), self.context_length).cuda()
        txt_mask = txt_mask.masked_fill(text == 0, float("-inf"))
        eos_indices = text.argmax(dim=-1)
        txt_mask[torch.arange(len(text)), eos_indices] = float("-inf")
        txt_local_list = []
        x, _, txt_local_list, _, _ = self.transformer(x, codebook, txt_local_list, mask=txt_mask, eos=eos_indices)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        stroke_tokens = x

        x = x[torch.arange(x.shape[0]), eos_indices] @ self.text_projection

        # return x
        # return stroke_tokens
        return x, txt_local_list


class SignWithSigmoidGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input


class Clip_Codebook(nn.Module):
    def __init__(self,
                 vision_input_resolution: int,
                 vision_patch_size: int,
                 vision_width: int,
                 vision_layers: int,
                 vision_heads: int,
                 vision_output_dim: int,
                 text_embed_dim: int,
                 text_context_length: int,
                 text_vocab_size: int,
                 text_width: int,
                 text_heads: int,
                 text_layers: int,
                 codebook_num: int,
                 codebook_dim: int,
                 temperature: int,
                 is_threshold: bool):
        super().__init__()

        self.layers=vision_layers

        self.visual = VisionTransformer(
            input_resolution=vision_input_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=vision_output_dim,
            codebook_dim=codebook_dim,
        )
        self.encode_text = TextTransformer(
            embed_dim=text_embed_dim,
            context_length=text_context_length,
            vocab_size=text_vocab_size,
            transformer_width=text_width,
            transformer_heads=text_heads,
            transformer_layers=text_layers,
            codebook_dim=codebook_dim,
        )
        self.context_length = text_context_length
        self.codebook_num = codebook_num
        self.codebook_dim = codebook_dim

        # codebook
        self.codebook = nn.Parameter(torch.randn(self.codebook_num, self.codebook_dim))

        self.temperature = temperature

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale3 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale4 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.local_norm = nn.BatchNorm1d(codebook_dim)
        self.global_norm = nn.BatchNorm1d(vision_output_dim)

        # self.restore=nn.Parameter(torch.ones(1))

        self.img_mix = nn.Sequential(
            nn.Linear(codebook_dim + vision_output_dim, 3072),
            nn.ReLU(),
            nn.Linear(3072, codebook_dim + vision_output_dim)
        )
        self.txt_mix = nn.Sequential(
            nn.Linear(codebook_dim + text_embed_dim, 3072),
            nn.ReLU(),
            nn.Linear(3072, codebook_dim + text_embed_dim)
        )
        self.q_map = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512)
        )
        self.q_map1 = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512)
        )
        self.mix = nn.Linear(1024,512)
        self.mix1 = nn.Linear(1024,512)
        # self.img_mix = nn.Linear(codebook_dim + vision_output_dim, codebook_dim + vision_output_dim)
        # self.txt_mix = nn.Linear(codebook_dim + text_embed_dim, codebook_dim + text_embed_dim)
        # self.img_mix = LayerAttentionFusion(vision_width,codebook_dim)
        # self.txt_mix = LayerAttentionFusion(text_width,codebook_dim)
        self.min_value = -1e+6
        self.threshold = 0.0
        self.is_threshold = is_threshold

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image, codebook):
        return self.visual(image.type(self.dtype), codebook)

    def val_encode_text(self, text):
        text_global_features, txt_local_list = self.encode_text(text, self.codebook)
        # final_fusion
        # text_features = self.txt_mix(text_global_features, txt_local_list)
        # text_features = self.txt_mix(torch.cat((txt_local, text_global), dim=1))
        # multi_add_final_fusion
        # text_local_features = torch.mean(torch.stack(txt_local_list), dim=0)
        # text_features = self.txt_mix(torch.cat((text_local_features, text_global_features), dim=1))

        # return text_global_features, txt_local_list[-1]
        # print(text_global_features.shape)
        return text_global_features, txt_local_list[-1]
        # return text_global_features, text_local_features, text_features

    def val_encode_data(self, image, text_global, text_local):
    # def val_encode_data(self, image, text_global, text_local):
        text_global_features = text_global
        text_local_features = text_local
        image_global_features, img_local_list = self.encode_image(image, self.codebook)     # [batch, patch_token, vision_width] [b, 64, 256]

        image_local_features = img_local_list[-1]

        # Global
        image_global_features1 = image_global_features / image_global_features.norm(dim=1, keepdim=True)
        text_global_features1 = text_global_features / text_global_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_global_per_image = logit_scale * image_global_features1 @ text_global_features1.t()
        # print(image_global_features.shape, text_global_features.shape)
        # Local
        image_local_features1 = image_local_features / image_local_features.norm(dim=1, keepdim=True)
        text_local_features1 = text_local_features / text_local_features.norm(dim=1, keepdim=True)
        logit_scale2 = self.logit_scale2.exp()
        logits_local_per_image = logit_scale2 * image_local_features1 @ text_local_features1.t()

        # fusion
        # img_local = self.local_norm(image_local_features)
        # txt_local = self.local_norm(text_local_features)
        # image_global_features = self.global_norm(image_global_features)
        # text_global_features = self.global_norm(text_global_features)
        # img_local = self.q_map(img_local)
        # txt_local = self.q_map1(txt_local) 
        # image_mix = self.mix(torch.cat((img_local, image_global_features), dim=1))
        # text_mix = self.mix1(torch.cat((txt_local, text_global_features), dim=1))
        # G_i = torch.sigmoid(image_mix)
        # G_t = torch.sigmoid(text_mix)
        # img_mf = image_global_features * G_i  + (1 - G_i) * img_local
        # text_mf = text_global_features * G_t + (1 - G_t) * txt_local

        # img_mf = img_mf / img_mf.norm(dim=1, keepdim=True)
        # text_mf = text_mf / text_mf.norm(dim=1, keepdim=True)
        # logit_scale4 = self.logit_scale4.exp()
        # logits = logit_scale4 * img_mf @ text_mf.t()
        # print(text_global_features.shape)
        
        image_mix_features = torch.cat((image_global_features, image_local_features), dim=1)
        image_mix_features = (image_mix_features - torch.mean(image_mix_features))/torch.var(image_mix_features)
        text_mix_features = torch.cat((text_global_features, text_local_features), dim=1)
        text_mix_features = (text_mix_features - torch.mean(text_mix_features))/torch.var(text_mix_features)

        image_mix_features = image_mix_features / image_mix_features.norm(dim=1, keepdim=True)
        text_mix_features = text_mix_features / text_mix_features.norm(dim=1, keepdim=True)
        logit_scale4 = self.logit_scale4.exp()
        logits = logit_scale4 * image_mix_features @ text_mix_features.t()

        return logits_global_per_image, logits_local_per_image, logits
        # return logits_global_per_image, logits_local_per_image, logits

    def visualization(self, image):
        image_global_features, img_local_list = self.encode_image(image, self.codebook)
        # print(img_local_list[0].shape) 16 256
        img_local = self.local_norm(img_local_list[-1])
        image_global=self.global_norm(image_global_features)
        image_global_features = image_global_features / image_global_features.norm(dim=1, keepdim=True)
        image_local_features = img_local/img_local.norm(dim=1, keepdim=True)
        return image_local_features, image_global_features

    def forward(self, image, text):
        image_global_features, img_local_list = self.encode_image(image, self.codebook)
        text_global_features, txt_local_list = self.encode_text(text, self.codebook)

        
        # image_features = self.img_mix(torch.cat((image_local_features, image_global_features), dim=1))
        # text_features = self.txt_mix(torch.cat((text_local_features, text_global_features), dim=1))
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # logit_scale3 = self.logit_scale3.exp()
        # logits_per_image = logit_scale3 * image_features @ text_features.t()

        # Global
        image_global_features1 = image_global_features / image_global_features.norm(dim=1, keepdim=True)
        text_global_features1 = text_global_features / text_global_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_global_per_image = logit_scale * image_global_features1 @ text_global_features1.t()

        # Local 每层local都损失的方案
        logit_scale2 = self.logit_scale2.exp()
        logits_local_list = []
        # for i, image_local_features in enumerate(img_local_list):
        #     if i > self.layers-3:
        #         image_local_features = image_local_features / image_local_features.norm(dim=1, keepdim=True)
        #         text_local_features = txt_local_list[i]
        #         text_local_features = text_local_features / text_local_features.norm(dim=1, keepdim=True)
        #         logits_local_per_image = logit_scale2 * image_local_features @ text_local_features.t()
        #         logits_local_list.append(logits_local_per_image)


        # start_layer = max(0, self.layers - 6) 
        start_layer = 0
        relevant_img_features = img_local_list[start_layer:]
        relevant_txt_features = txt_local_list[start_layer:]
        img_stack = torch.stack(relevant_img_features)
        txt_stack = torch.stack(relevant_txt_features)

        img_stack = img_stack / img_stack.norm(dim=2, keepdim=True)
        txt_stack = txt_stack / txt_stack.norm(dim=2, keepdim=True)

        logits_local = logit_scale2 * torch.bmm(img_stack, txt_stack.transpose(1, 2))

        logits_local_list.extend(logits_local.unbind(0))
        # logits_local_list.extend([logits_local[i] for i in range(logits_local.shape[0])])

        
        # final_fusion
        # image_features = self.img_mix(image_global_features, img_local_list)
        # text_features = self.txt_mix(text_global_features, txt_local_list)
        image_mix_features = torch.cat((image_global_features, img_local_list[-1]), dim=1)
        image_mix_features = (image_mix_features - torch.mean(image_mix_features))/torch.var(image_mix_features)
        text_mix_features = torch.cat((text_global_features, txt_local_list[-1]), dim=1)
        text_mix_features = (text_mix_features - torch.mean(text_mix_features))/torch.var(text_mix_features)

        image_mix_features = image_mix_features / image_mix_features.norm(dim=1, keepdim=True)
        text_mix_features = text_mix_features / text_mix_features.norm(dim=1, keepdim=True)
        logit_scale4 = self.logit_scale4.exp()
        logits = logit_scale4 * image_mix_features @ text_mix_features.t()
        return logits_global_per_image, logits_local_list, logits

        # final_fusion
        # image_features = self.img_mix(torch.cat((img_local_list[-1], image_global_features), dim=1))
        # text_features = self.txt_mix(torch.cat((txt_local_list[-1], text_global_features), dim=1))

        # multi_add_final_fusion
        # image_local_features = torch.mean(torch.stack(img_local_list), dim=0)
        # image_features = self.img_mix(torch.cat((image_local_features, image_global_features), dim=1))
        # text_local_features = torch.mean(torch.stack(txt_local_list), dim=0)
        # text_features = self.txt_mix(torch.cat((text_local_features, text_global_features), dim=1))
        #
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # logit_scale3 = self.logit_scale3.exp()
        # logits_per_image = logit_scale3 * image_features @ text_features.t()

        # return logits_global_per_image, logits_local_list, logits_per_image


    


