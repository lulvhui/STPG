from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataload import tokenizer,CFG


class module(nn.Module):
    def __init__(self):
        super(module, self).__init__()

        # Text Encoder
        self.BertModel = BertModel.from_pretrained(
            CFG['context_model']
        )

        self.sticer_bert = BertModel.from_pretrained(
            CFG['context_model']
        )

        # Sticker Image Encoder
        self.sticker_cnn = nn.Conv1d(
            in_channels=512,
            out_channels=768,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Eq.(5)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            batch_first=True
        )

        # Eq.(6)
        self.diff_linear = nn.Linear(
            768,
            768
        )

        self.sticker_align = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            batch_first=True
        )

        # Eq.(7)
        self.sticker_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            batch_first=True
        )

        # Eq.(8)
        self.context_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            batch_first=True
        )

        # Eq.(9)
        self.fusion_fc = nn.Linear(
            768 * 5,
            768
        )

        # Eq.(10)
        self.emo_classifier = nn.Linear(
            768,
            3
        )

        # Eq.(11)
        self.intent_classifier = nn.Linear(
            768,
            20
        )

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        context_ids,
        context_masks,
        token_type_ids,
        sticker_txt_ids,
        sticertext_masks,
        stickertext_type_ids,
        sticker_features,
        emo_lable,
        intent_lable,
        emo_t=None,
        emo_s=None
    ):

        # =====================================================
        # Eq.(1) Context Encoder
        # =====================================================
        context_output = self.BertModel(
            input_ids=context_ids,
            attention_mask=context_masks,
            token_type_ids=token_type_ids
        )

        E_X = context_output.last_hidden_state
        # [bs, context_len, 768]

        # =====================================================
        # Eq.(2) Sticker Text Encoder
        # =====================================================
        sticker_output = self.sticer_bert(
            input_ids=sticker_txt_ids,
            attention_mask=sticertext_masks,
            token_type_ids=stickertext_type_ids
        )

        E_S = sticker_output.last_hidden_state
        # [bs, sticker_len, 768]

        # =====================================================
        # Eq.(3) Sticker Image Encoder
        # =====================================================
        E_I = sticker_features.float()

        E_I = E_I.unsqueeze(1)
        E_I = E_I.transpose(1, 2)

        E_I = self.sticker_cnn(E_I)

        E_I = E_I.transpose(1, 2)
        # [bs,1,768]

        # =====================================================
        # Eq.(4)
        # =====================================================
        E_IS = torch.cat(
            [E_I, E_S],
            dim=1
        )
        # [bs, sticker_len+1, 768]

        # =====================================================
        # Eq.(5)
        # O_MHA = MultiHead(E_X,E_IS,E_IS)
        # =====================================================
        O_MHA, _ = self.cross_attention(
            query=E_X,
            key=E_IS,
            value=E_IS
        )

        # =====================================================
        # Eq.(6)
        # V_diff = W_diff(O_MHA-E_X)+b
        # =====================================================
        V_diff = self.diff_linear(
            O_MHA - E_X
        )

        # =====================================================
        # Eq.(7)
        # O_S
        # =====================================================
        # length alignment
        E_S_align, _ = self.sticker_align(
            query=E_X,
            key=E_S,
            value=E_S
            )

        
        O_S, _ = self.sticker_attention(
            query=E_S_align,
            key=E_S_align,
            value=O_MHA
        )


        # =====================================================
        # Eq.(8)
        # O_X
        # =====================================================
        O_X, _ = self.context_attention(
            query=E_X,
            key=E_X,
            value=O_MHA
        )

        # =====================================================
        # Pooling
        # =====================================================
        E_X_cls = E_X[:, 0]

        V_diff_cls = V_diff[:, 0]

        O_X_cls = O_X[:, 0]

        E_IS_pool = E_IS.mean(dim=1)

        O_S_pool = O_S.mean(dim=1)

        # =====================================================
        # Eq.(9)
        # E_combined
        # =====================================================
        E_combined = torch.cat(
            [
                E_IS_pool,
                E_X_cls,
                V_diff_cls,
                O_S_pool,
                O_X_cls
            ],
            dim=-1
        )

        E_combined = self.dropout(
            E_combined
        )

        E_combined = self.fusion_fc(
            E_combined
        )

        E_combined = torch.tanh(
            E_combined
        )

        # =====================================================
        # Eq.(10)
        # Sentiment Head
        # =====================================================
        emo_logits = self.emo_classifier(
            E_combined
        )

        # =====================================================
        # Eq.(11)
        # Intent Head
        # =====================================================
        intent_logits = self.intent_classifier(
            E_combined
        )

        # =====================================================
        # Eq.(12)(13)(14)
        # =====================================================
        loss_emo = F.cross_entropy(
            emo_logits,
            emo_lable
        )

        loss_intent = F.cross_entropy(
            intent_logits,
            intent_lable
        )

        alpha = 1.0
        beta = 1.0

        loss = (
            alpha * loss_emo
            + beta * loss_intent
        )

        emo_prob = torch.softmax(
            emo_logits,
            dim=-1
        )

        intent_prob = torch.softmax(
            intent_logits,
            dim=-1
        )

        return (
            loss,
            emo_prob,
            intent_prob
        )


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class CSFusion(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CSFusion, self).__init__()
        w4C = torch.empty(dim, 1)
        w4S = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4S)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4S = nn.Parameter(w4S, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cs_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.cnn0 = nn.Conv1d(768, 11, kernel_size=1, stride=1, padding=0)
        self.cnn1 = nn.Conv1d(768, 50, kernel_size=1, stride=1, padding=0)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)


    def forward(self, context, sticker, c_mask, s_mask):
        score = self.cs_attention(context, sticker)  # (batch_size, c_len, s_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, s_mask.unsqueeze(1)))  # (batch_size, c_len, s_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_len, s_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, s_len, c_len)
        c2s = torch.matmul(score_, sticker)  # (batch_size, c_len, dim)
        s2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_len, dim)
        output = torch.cat([context, c2s, torch.mul(context, c2s), torch.mul(context, s2c)], dim=2)
        output = self.cs_linear(output)  # (batch_size, c_len, dim)
        return output

    def cs_attention(self, context, sticker):
        c_len = context.shape[1]
        s_len = sticker.shape[1]
        context = self.dropout(context)
        sticker = self.dropout(sticker)
        #subres0 = torch.matmul(context, self.w4C).expand([-1, -1, s_len])  # (batch_size, c_len, s_len)
        subres0 = self.cnn0(context.transpose(1,2)).transpose(1,2)
        #subres1 = torch.matmul(sticker, self.w4S).transpose(1, 2).expand([-1, c_len, -1])
        subres1 = self.cnn1(sticker.transpose(1,2))
        #subres2 = torch.matmul(context * self.w4mlu, sticker.transpose(1, 2))
        subres2 = self.attention(context, sticker, sticker)[0]
        subres2 = self.cnn0(subres2.transpose(1,2)).transpose(1,2)
        res = subres0 + subres1 + subres2  # (batch_size, c_len, s_len)
        return res

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class LayerNorm1(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm1, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class AttFlat(nn.Module):
    def __init__(self):
        super(AttFlat, self).__init__()
        

        self.mlp = MLP(
            in_size=768,
            mid_size=768,
            out_size=768,
            dropout_r=0.1,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
           768 * 1,
           768
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        x_mask = x_mask.bool()
        x_mask = x_mask.unsqueeze(2)
        att = att.masked_fill(
            ~x_mask,
            value = torch.tensor(-1e9, dtype = att.dtype)
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(768):
            a = att[:, :, i: i + 1]
            b = a * x
            c = torch.sum(b, dim=1)
            att_list.append(
                c.unsqueeze(2)
            )

        x_atted = torch.cat(att_list, dim=2)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self):
        super(MHAtt, self).__init__()
        

        self.linear_v = nn.Linear(768, 768)
        self.linear_k = nn.Linear(768, 768)
        self.linear_q = nn.Linear(768, 768)
        self.linear_merge = nn.Linear(768, 768)

        self.dropout = nn.Dropout(0.1)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            8,
            96
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            8,
            96
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            8,
            96
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            768
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(~(mask.unsqueeze(1).unsqueeze(2).bool()), value = torch.tensor(-1e9, dtype = scores.dtype))

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        if not att_map.size(3) == value.size(2):
            value = value.expand(value.size(0), value.size(1), att_map.size(3), value.size(3))
        return torch.matmul(att_map, value)



# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=768,
            mid_size=768*4,
            out_size=768,
            dropout_r=0.1,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.mhatt = MHAtt()
        self.ffn = FFN()

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm1(768)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm1(768)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x

# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt()
        self.mhatt2 = MHAtt()
        self.ffn = FFN()

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm1(768)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm1(768)

        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = LayerNorm1(768)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


class VLC(nn.Module):
    def __init__(self):
        super(VLC, self).__init__()

        self.enc_list = nn.ModuleList([SA() for _ in range(6)])
        self.dec_lang_frames_list = nn.ModuleList([SGA() for _ in range(6)])
        self.dec_lang_clips_list = nn.ModuleList([SGA() for _ in range(6)])


    def forward(self, x, y, z, x_mask, y_mask, z_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_lang_frames_list:
            y = dec(y, x, y_mask, x_mask)

        for dec in self.dec_lang_clips_list:
            z = dec(z, x, z_mask, x_mask)
        return x, y, z
