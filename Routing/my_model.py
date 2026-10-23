import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from my_dataload import CFG, TASK_CONFIG, ROUTING_CONFIG


class ModalityRouter(nn.Module):
    """模态路由器：决定使用文本路径还是图像路径"""
    
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # T_path vs I_path
        )
    
    def forward(self, x):
        """返回模态路由概率 [T_prob, I_prob]"""
        logits = self.mlp(x)
        return F.softmax(logits, dim=-1)


class TaskRouter(nn.Module):
    """任务路由器：为每个任务决定使用 STL 还是 MTL 策略"""
    
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # STL vs MTL
        )
    
    def forward(self, x):
        """返回路由概率分布 [STL_prob, MTL_prob]"""
        logits = self.mlp(x)
        return F.softmax(logits, dim=-1)


class Expert(nn.Module):
    """专家网络：处理特征并输出任务特定表示"""
    
    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class MultiTaskModel(nn.Module):
    """
    基于 learning-to-route 思想的双层路由多任务模型
    
    核心架构：
    1. 第一层：模态路由（选择文本路径T或图像路径I）
    2. 第二层：任务路由（每个模态路径内，选择STL或MTL专家）
    
    模态路径定义：
    - T路径 (Text): 评论文本 + 表情包文本
    - I路径 (Image): 表情包图像特征
    """
    def __init__(self, routing_config=None, use_routing=True):
        super().__init__()
        if routing_config is None:
            routing_config = ROUTING_CONFIG.copy()
        
        expert_hidden_dim = routing_config.get('expert_hidden', 512)
        router_hidden_dim = routing_config.get('router_hidden', 128)
        self.use_routing = use_routing
        
        # ---------- 模态编码器 ----------
        # 文本编码器
        self.BertModel = BertModel.from_pretrained(CFG['context_model'])
        self.sticer_bert = BertModel.from_pretrained(CFG['context_model'])
        
        # 图像编码器
        self.sticker_cnn = nn.Conv1d(512, 768, kernel_size=1, stride=1, padding=0)
        
        # 跨模态注意力（用于T路径的文本融合）
        self.cross_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        self.diff_linear = nn.Linear(768, 768)
        self.sticker_align = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        self.sticker_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        self.context_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)
        
        # T路径融合层
        self.text_fusion_fc = nn.Linear(768 * 5, 768)
        
        # I路径投影层
        self.image_proj = nn.Linear(768, 768)
        
        self.dropout = nn.Dropout(0.1)

        # ---------- 任务列表 ----------
        self.task_names = list(TASK_CONFIG.keys())
        num_tasks = len(self.task_names)

        # ---------- 第一层：模态路由器 ----------
        self.modality_router = ModalityRouter(768, router_hidden_dim)
        
        # ---------- 第二层：任务路由器（每个模态路径独立） ----------
        # T路径的任务路由器
        self.task_routers_T = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_routers_T[task_name] = TaskRouter(768, router_hidden_dim)
        
        # I路径的任务路由器
        self.task_routers_I = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_routers_I[task_name] = TaskRouter(768, router_hidden_dim)
        
        # ---------- 专家网络 ----------
        num_shared_experts = routing_config.get('num_shared_experts', 4)
        num_task_experts = routing_config.get('num_task_experts', 2)
        
        # T路径的专家
        self.shared_experts_T = nn.ModuleList([
            Expert(768, expert_hidden_dim) for _ in range(num_shared_experts)
        ])
        self.task_experts_T = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_experts_T[task_name] = nn.ModuleList([
                Expert(768, expert_hidden_dim) for _ in range(num_task_experts)
            ])
        
        # I路径的专家
        self.shared_experts_I = nn.ModuleList([
            Expert(768, expert_hidden_dim) for _ in range(num_shared_experts)
        ])
        self.task_experts_I = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_experts_I[task_name] = nn.ModuleList([
                Expert(768, expert_hidden_dim) for _ in range(num_task_experts)
            ])
        
        # 层归一化
        self.task_norm = nn.LayerNorm(768)

        # ---------- 分类头 ----------
        self.classifiers = nn.ModuleDict()
        for task_name, cfg in TASK_CONFIG.items():
            self.classifiers[task_name] = nn.Linear(768, cfg['num_classes'])
        
        # 保存配置
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.num_modalities = 2  # T和I

    def set_epoch(self, epoch):
        """设置 epoch 编号（用于兼容性）"""
        pass

    def get_gate_weights(self, modality_feature, modality_name='T'):
        """获取各任务的路由权重分布"""
        weights = {}
        with torch.no_grad():
            routers = self.task_routers_T if modality_name == 'T' else self.task_routers_I
            for task_name in self.task_names:
                probs = routers[task_name](modality_feature)
                weights[task_name] = probs.mean(dim=0).cpu().numpy()
        return weights

    def extract_text_features(self, context_ids, context_masks, token_type_ids,
                              sticker_txt_ids, sticertext_masks, stickertext_type_ids):
        """提取T路径（文本）特征"""
        E_X = self.BertModel(input_ids=context_ids, attention_mask=context_masks,
                             token_type_ids=token_type_ids).last_hidden_state
        E_S = self.sticer_bert(input_ids=sticker_txt_ids, attention_mask=sticertext_masks,
                               token_type_ids=stickertext_type_ids).last_hidden_state
        
        # 创建虚拟图像特征（用于跨模态注意力）
        B = E_X.shape[0]
        E_I_dummy = torch.zeros(B, 1, 768, device=E_X.device)
        
        E_IS = torch.cat([E_I_dummy, E_S], dim=1)
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

        E_text = torch.cat([E_IS_pool, E_X_cls, V_diff_cls, O_S_pool, O_X_cls], dim=-1)
        E_text = self.dropout(E_text)
        E_text = self.text_fusion_fc(E_text)
        E_text = torch.tanh(E_text)
        
        return E_text

    def extract_image_features(self, sticker_features):
        """提取I路径（图像）特征"""
        E_I = sticker_features.float().unsqueeze(1).transpose(1, 2)
        E_I = self.sticker_cnn(E_I).transpose(1, 2)
        E_I = E_I.mean(dim=1)  # [B, 768]
        E_I = self.dropout(E_I)
        E_I = self.image_proj(E_I)
        E_I = torch.tanh(E_I)
        
        return E_I

    def route_single_modality(self, modality_feature, task_routers, 
                             shared_experts, task_experts):
        """
        单模态路径内的任务路由
        
        Args:
            modality_feature: [B, 768] 模态特征
            task_routers: 该模态路径的任务路由器
            shared_experts: 该模态路径的共享专家
            task_experts: 该模态路径的任务特定专家
        
        Returns:
            task_features: dict of [B, 768] 每个任务的特征
            gate_weights: dict of [B, 2] 每个任务的STL/MTL概率
        """
        task_features = {}
        gate_weights = {}
        
        # 获取共享专家输出
        shared_outputs = []
        for expert in shared_experts:
            shared_outputs.append(expert(modality_feature))
        shared_outputs = torch.stack(shared_outputs, dim=1)  # [B, num_shared, 768]
        
        for task_name in self.task_names:
            # 获取任务特定专家输出
            task_outputs = []
            for expert in task_experts[task_name]:
                task_outputs.append(expert(modality_feature))
            task_outputs = torch.stack(task_outputs, dim=1)  # [B, num_task, 768]
            
            # 获取路由权重
            route_probs = task_routers[task_name](modality_feature)
            # [B, 2] -> [STL_prob, MTL_prob]
            gate_weights[task_name] = route_probs
            
            # 加权融合
            stl_weight = route_probs[:, 0:1]  # [B, 1]
            mtl_weight = route_probs[:, 1:2]  # [B, 1]
            
            task_feat = (
                mtl_weight * shared_outputs.mean(dim=1) +
                stl_weight * task_outputs.mean(dim=1)
            )  # [B, 768]
            
            task_features[task_name] = self.task_norm(task_feat)
        
        return task_features, gate_weights

    def routing_forward(self, E_text, E_image):
        """
        双层路由前向传播
        
        核心思想：
        1. 第一层：模态路由决定使用T路径还是I路径
        2. 第二层：每个模态路径内的任务路由决定STL/MTL
        3. 最终特征 = T_prob × T_path_features + I_prob × I_path_features
        """
        # ---- 第一层：模态路由 ----
        # 使用T路径特征作为模态路由的输入
        modality_probs = self.modality_router(E_text)  # [B, 2] -> [T_prob, I_prob]
        
        # ---- 第二层：T路径任务路由 ----
        task_features_T, gate_weights_T = self.route_single_modality(
            E_text, 
            self.task_routers_T, 
            self.shared_experts_T, 
            self.task_experts_T
        )
        
        # ---- 第二层：I路径任务路由 ----
        task_features_I, gate_weights_I = self.route_single_modality(
            E_image, 
            self.task_routers_I, 
            self.shared_experts_I, 
            self.task_experts_I
        )
        
        # ---- 双层加权融合 ----
        task_features = {}
        gate_weights = {}
        
        T_prob = modality_probs[:, 0:1]  # [B, 1]
        I_prob = modality_probs[:, 1:2]  # [B, 1]
        
        for task_name in self.task_names:
            # 融合两个模态路径的特征
            final_feat = (
                T_prob * task_features_T[task_name] +
                I_prob * task_features_I[task_name]
            )  # [B, 768]
            
            task_features[task_name] = self.task_norm(final_feat)
            
            # 记录路由权重（用于分析）
            gate_weights[task_name] = {
                'modality': modality_probs,
                'T_task': gate_weights_T[task_name],
                'I_task': gate_weights_I[task_name]
            }
        
        return task_features, gate_weights, modality_probs

    # ---------- 前向 ----------
    def forward(self, context_ids, context_masks, token_type_ids,
                sticker_txt_ids, sticertext_masks, stickertext_type_ids,
                sticker_features, labels_dict=None):
        # ---- 提取模态特征 ----
        E_text = self.extract_text_features(
            context_ids, context_masks, token_type_ids,
            sticker_txt_ids, sticertext_masks, stickertext_type_ids
        )
        E_image = self.extract_image_features(sticker_features)

        # ---- 双层路由前向传播 ----
        if self.use_routing:
            task_features, gate_weights, modality_probs = self.routing_forward(E_text, E_image)
        else:
            # 不使用路由时，直接融合两个模态特征
            E_combined = (E_text + E_image) / 2
            task_features = {task: E_combined for task in self.task_names}
            gate_weights = {}
            modality_probs = None

        # ---- 分类 ----
        logits = {}
        for task_name in self.task_names:
            logits[task_name] = self.classifiers[task_name](task_features[task_name])

        if labels_dict is not None:
            total_loss = 0.0
            for task_name in self.task_names:
                target = labels_dict[task_name]
                total_loss += F.cross_entropy(logits[task_name], target)
            
            # 添加路由熵正则化（鼓励明确的路由决策）
            if self.use_routing and gate_weights:
                entropy_loss = 0.0
                
                # 模态路由熵
                entropy_modality = -(modality_probs * torch.log(modality_probs + 1e-10)).sum(dim=-1).mean()
                entropy_loss += entropy_modality
                
                # 任务路由熵
                for task_name in self.task_names:
                    probs_T = gate_weights[task_name]['T_task']
                    probs_I = gate_weights[task_name]['I_task']
                    entropy_T = -(probs_T * torch.log(probs_T + 1e-10)).sum(dim=-1).mean()
                    entropy_I = -(probs_I * torch.log(probs_I + 1e-10)).sum(dim=-1).mean()
                    entropy_loss += (entropy_T + entropy_I)
                
                total_loss += 0.1 * entropy_loss  # 熵正则化系数
            
            return total_loss, logits, gate_weights
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