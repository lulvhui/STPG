import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from my_dataload import CFG, TASK_CONFIG, MOE

class MultiTaskModel(nn.Module):
    def __init__(self, num_experts=None, expert_hidden_dim=None,
                 top_k_router=2, noise_std=0.1,
                 use_load_balancing=True, load_balance_weight=0.01,
                 use_ln=True):
        super().__init__()
        if num_experts is None:
            num_experts = MOE['num_experts']
        if expert_hidden_dim is None:
            expert_hidden_dim = MOE['expert_hidden']
        
        # ---------- 共享编码器 (保持不变) ----------
        self.BertModel = BertModel.from_pretrained(CFG['context_model'])
        self.sticer_bert = BertModel.from_pretrained(CFG['context_model'])

        self.sticker_cnn = nn.Conv1d(512, 768, kernel_size=1, stride=1, padding=0)
        self.cross_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        self.diff_linear = nn.Linear(768, 768)
        self.sticker_align = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        self.sticker_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        self.context_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)

        self.fusion_fc = nn.Linear(768 * 5, 768)
        self.dropout = nn.Dropout(0.1)

        # ---------- 任务列表 ----------
        self.task_names = list(TASK_CONFIG.keys())
        num_tasks = len(self.task_names)

        # ---------- MoE 参数 ----------
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.top_k_router = min(top_k_router, num_experts)
        self.noise_std = noise_std
        self.use_load_balancing = use_load_balancing
        self.load_balance_weight = load_balance_weight
        self.use_ln = use_ln

        # ---------- Expert 网络 (共享) ----------
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden_dim, 768)
            ) for _ in range(num_experts)
        ])

        # ---------- 任务专属 Router ----------
        self.routers = nn.ModuleDict()
        for task_name in self.task_names:
            self.routers[task_name] = nn.Sequential(
                nn.Linear(768, num_experts),
            )

        # ---------- 层归一化 ----------
        if use_ln:
            self.task_norm = nn.LayerNorm(768)
        else:
            self.task_norm = None

        # ---------- 分类头 ----------
        self.classifiers = nn.ModuleDict()
        for task_name, cfg in TASK_CONFIG.items():
            self.classifiers[task_name] = nn.Linear(768, cfg['num_classes'])

    # ---------- 对外接口 ----------
    def set_epoch(self, epoch):
        pass

    def get_router_weights(self, shared_feature):
        """获取各任务的 router 权重分布"""
        weights = {}
        with torch.no_grad():
            for task_name in self.task_names:
                logits = self.routers[task_name](shared_feature)
                weights[task_name] = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
        return weights

    # ---------- MoE 前向传播 ----------
    def moe_forward(self, shared_feature):
        """
        MoE 前向传播
        Args:
            shared_feature: [B, 768] 共享特征
        Returns:
            task_features: dict of [B, 768] 每个任务的输出特征
            router_weights: dict of [B, num_experts] 各任务的router权重
        """
        B = shared_feature.size(0)
        
        # 计算 Expert 输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(shared_feature))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, 768]
        
        task_features = {}
        router_weights = {}
        
        for task_name in self.task_names:
            # Router logits
            logits = self.routers[task_name](shared_feature)  # [B, num_experts]
            
            # 训练时添加噪声增强探索
            if self.training and self.noise_std > 0:
                noise = torch.randn_like(logits) * self.noise_std
                logits = logits + noise * F.softplus(logits)
            
            # Top-K 路由
            if self.training and self.top_k_router < self.num_experts:
                topk_vals, topk_idx = torch.topk(logits, k=self.top_k_router, dim=-1)
                gate = F.softmax(topk_vals, dim=-1)  # [B, top_k]
                sparse_gate = torch.zeros_like(logits)
                sparse_gate.scatter_(1, topk_idx, gate)  # [B, num_experts]
                router_w = sparse_gate
            else:
                router_w = F.softmax(logits, dim=-1)  # [B, num_experts]
            
            router_weights[task_name] = router_w
            
            # 加权求和
            task_feat = torch.einsum('be,bed->bd', router_w, expert_outputs)  # [B, 768]
            task_features[task_name] = task_feat
        
        # 可选层归一化
        if self.use_ln and self.task_norm is not None:
            for task_name in self.task_names:
                task_features[task_name] = self.task_norm(task_features[task_name])
        
        return task_features, router_weights

    # ---------- 负载均衡损失 ----------
    def compute_load_balance_loss(self, router_weights):
        """
        计算负载均衡损失，防止路由坍塌
        Args:
            router_weights: dict of [B, num_experts]
        Returns:
            loss: scalar tensor
        """
        loss = 0.0
        for gate in router_weights.values():
            avg_gate = gate.mean(dim=0)  # [num_experts]
            loss += ((avg_gate - 1.0 / self.num_experts) ** 2).sum()
        return loss / len(router_weights)

    # ---------- 前向 ----------
    def forward(self, context_ids, context_masks, token_type_ids,
                sticker_txt_ids, sticertext_masks, stickertext_type_ids,
                sticker_features, labels_dict=None):
        # ---- 共享编码 ----
        E_X = self.BertModel(input_ids=context_ids, attention_mask=context_masks,
                             token_type_ids=token_type_ids).last_hidden_state
        E_S = self.sticer_bert(input_ids=sticker_txt_ids, attention_mask=sticertext_masks,
                               token_type_ids=stickertext_type_ids).last_hidden_state
        E_I = sticker_features.float().unsqueeze(1).transpose(1, 2)
        E_I = self.sticker_cnn(E_I).transpose(1, 2)

        E_IS = torch.cat([E_I, E_S], dim=1)
        O_MHA, _ = self.cross_attention(query=E_X, key=E_IS, value=E_IS)
        V_diff = self.diff_linear(O_MHA - E_X)
        E_S_align, _ = self.sticker_align(query=E_X, key=E_S, value=E_S)
        O_S, _ = self.sticker_attention(query=E_S_align, key=E_S_align, value=O_MHA)
        O_X, _ = self.context_attention(query=E_X, key=E_X, value=O_MHA)

        E_X_cls = E_X[:, 0]
        V_diff_cls = V_diff[:, 0]
        O_X_cls = O_X[:, 0]
        E_IS_pool = E_IS.mean(dim=1)
        O_S_pool = O_S.mean(dim=1)

        E_combined = torch.cat([E_IS_pool, E_X_cls, V_diff_cls, O_S_pool, O_X_cls], dim=-1)
        E_combined = self.dropout(E_combined)
        E_combined = self.fusion_fc(E_combined)
        E_combined = torch.tanh(E_combined)      # [B, 768]

        # ---- MoE 前向传播 ----
        task_features, router_weights = self.moe_forward(E_combined)

        # ---- 分类 ----
        logits = {}
        for task_name in self.task_names:
            logits[task_name] = self.classifiers[task_name](task_features[task_name])

        if labels_dict is not None:
            total_loss = 0.0
            for task_name in self.task_names:
                target = labels_dict[task_name]
                total_loss += F.cross_entropy(logits[task_name], target)
            
            # 负载均衡损失
            if self.use_load_balancing:
                lb_loss = self.compute_load_balance_loss(router_weights)
                total_loss = total_loss + self.load_balance_weight * lb_loss
            
            return total_loss, logits, router_weights
        else:
            return logits


# ===================== 辅助类保持不变 =====================
class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        return x.transpose(1, 2)


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
        score = self.cs_attention(context, sticker)
        score_ = nn.Softmax(dim=2)(mask_logits(score, s_mask.unsqueeze(1)))
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))
        score_t = score_t.transpose(1, 2)
        c2s = torch.matmul(score_, sticker)
        s2c = torch.matmul(torch.matmul(score_, score_t), context)
        output = torch.cat([context, c2s, torch.mul(context, c2s), torch.mul(context, s2c)], dim=2)
        output = self.cs_linear(output)
        return output

    def cs_attention(self, context, sticker):
        context = self.dropout(context)
        sticker = self.dropout(sticker)
        subres0 = self.cnn0(context.transpose(1, 2)).transpose(1, 2)
        subres1 = self.cnn1(sticker.transpose(1, 2))
        subres2 = self.attention(context, sticker, sticker)[0]
        subres2 = self.cnn0(subres2.transpose(1, 2)).transpose(1, 2)
        return subres0 + subres1 + subres2


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
        self.mlp = MLP(in_size=768, mid_size=768, out_size=768, dropout_r=0.1, use_relu=True)
        self.linear_merge = nn.Linear(768 * 1, 768)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        x_mask = x_mask.bool().unsqueeze(2)
        att = att.masked_fill(~x_mask, value=torch.tensor(-1e9, dtype=att.dtype))
        att = F.softmax(att, dim=1)
        att_list = []
        for i in range(768):
            a = att[:, :, i: i + 1]
            b = a * x
            c = torch.sum(b, dim=1)
            att_list.append(c.unsqueeze(2))
        x_atted = torch.cat(att_list, dim=2)
        return self.linear_merge(x_atted)


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
        v = self.linear_v(v).view(n_batches, -1, 8, 96).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, 8, 96).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, 8, 96).transpose(1, 2)
        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, 768)
        return self.linear_merge(atted)

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~(mask.unsqueeze(1).unsqueeze(2).bool()), value=torch.tensor(-1e9, dtype=scores.dtype))
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        if not att_map.size(3) == value.size(2):
            value = value.expand(value.size(0), value.size(1), att_map.size(3), value.size(3))
        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.mlp = MLP(in_size=768, mid_size=768 * 4, out_size=768, dropout_r=0.1, use_relu=True)

    def forward(self, x):
        return self.mlp(x)


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
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


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
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class VLC(nn.Module):
    def __init__(self):
        super(VLC, self).__init__()
        self.enc_list = nn.ModuleList([SA() for _ in range(6)])
        self.dec_lang_frames_list = nn.ModuleList([SGA() for _ in range(6)])
        self.dec_lang_clips_list = nn.ModuleList([SGA() for _ in range(6)])

    def forward(self, x, y, z, x_mask, y_mask, z_mask):
        for enc in self.enc_list:
            x = enc(x, x_mask)
        for dec in self.dec_lang_frames_list:
            y = dec(y, x, y_mask, x_mask)
        for dec in self.dec_lang_clips_list:
            z = dec(z, x, z_mask, x_mask)
        return x, y, z


import math

class SingleTaskModel(nn.Module):
    """
    单任务基线（无 TPG）：
    保留原 MultiTaskModel 的共享编码器与融合结构，
    移除所有 TPG 部件（原型、图传播、门控、层归一化、gamma 等），
    并将多任务分类头替换为单一分类头。
    """
    def __init__(self, num_classes):
        super().__init__()
        # ---------- 共享编码器（与 MultiTaskModel 完全相同）----------
        self.BertModel = BertModel.from_pretrained(CFG['context_model'])
        self.sticer_bert = BertModel.from_pretrained(CFG['context_model'])

        self.sticker_cnn = nn.Conv1d(512, 768, kernel_size=1, stride=1, padding=0)
        self.cross_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        self.diff_linear = nn.Linear(768, 768)
        self.sticker_align = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        self.sticker_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        self.context_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)

        self.fusion_fc = nn.Linear(768 * 5, 768)
        self.dropout = nn.Dropout(0.1)

        # ---------- 单任务分类头 ----------
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, context_ids, context_masks, token_type_ids,
                sticker_txt_ids, sticertext_masks, stickertext_type_ids,
                sticker_features, labels=None):
        # ---- 共享编码 ----
        E_X = self.BertModel(input_ids=context_ids, attention_mask=context_masks,
                             token_type_ids=token_type_ids).last_hidden_state
        E_S = self.sticer_bert(input_ids=sticker_txt_ids, attention_mask=sticertext_masks,
                               token_type_ids=stickertext_type_ids).last_hidden_state
        E_I = sticker_features.float().unsqueeze(1).transpose(1, 2)
        E_I = self.sticker_cnn(E_I).transpose(1, 2)

        E_IS = torch.cat([E_I, E_S], dim=1)
        O_MHA, _ = self.cross_attention(query=E_X, key=E_IS, value=E_IS)
        V_diff = self.diff_linear(O_MHA - E_X)
        E_S_align, _ = self.sticker_align(query=E_X, key=E_S, value=E_S)
        O_S, _ = self.sticker_attention(query=E_S_align, key=E_S_align, value=O_MHA)
        O_X, _ = self.context_attention(query=E_X, key=E_X, value=O_MHA)

        E_X_cls = E_X[:, 0]
        V_diff_cls = V_diff[:, 0]
        O_X_cls = O_X[:, 0]
        E_IS_pool = E_IS.mean(dim=1)
        O_S_pool = O_S.mean(dim=1)

        E_combined = torch.cat([E_IS_pool, E_X_cls, V_diff_cls, O_S_pool, O_X_cls], dim=-1)
        E_combined = self.dropout(E_combined)
        E_combined = self.fusion_fc(E_combined)
        E_combined = torch.tanh(E_combined)      # [B, 768]

        # ---- 单任务分类 ----
        logits = self.classifier(E_combined)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        return logits