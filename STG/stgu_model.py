import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from my_dataload import CFG, TASK_CONFIG
import copy
import numpy as np
from collections import defaultdict

class STGUModel(nn.Module):
    """
    ICLR 2025: Selective Task Group Updates for Multi-Task Optimization
    基于Proximal Inter-Task Affinity的动态任务分组与选择性更新
    
    核心创新：
    1. Proximal Inter-Task Affinity (PITA): 衡量任务组对目标任务的影响
    2. Dynamic Task Grouping: 基于PITA动态调整任务分组
    3. Selective Updates: 按组顺序进行参数更新，避免梯度冲突
    """
    
    def __init__(self, task_hidden_dim=768, use_ln=True, num_groups=3, beta=0.9):
        super().__init__()
        
        # ============ 保留原有的多模态编码器结构 ============
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
        
        # ============ 任务相关配置 ============
        self.task_names = list(TASK_CONFIG.keys())
        num_tasks = len(self.task_names)
        self.task_hidden_dim = task_hidden_dim
        self.use_ln = use_ln
        
        # ============ STGU核心参数 ============
        self.num_groups = num_groups          # 初始分组数量
        self.beta = beta                      # PITA衰减率 (0.9)
        
        # ============ 共享网络和任务头 ============
        self.shared_mlp = nn.Sequential(
            nn.Linear(768, task_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(task_hidden_dim, 768),
        )
        
        self.task_heads = nn.ModuleDict()
        for task_name, cfg in TASK_CONFIG.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(768, task_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(task_hidden_dim, cfg['num_classes']),
            )
        
        if use_ln:
            self.shared_norm = nn.LayerNorm(768)
        else:
            self.shared_norm = None
        
        # ============ STGU状态管理 ============
        self._init_stgu_state()
    
    def _init_stgu_state(self):
        """初始化STGU的状态变量"""
        num_tasks = len(self.task_names)
        
        # 任务分组字典 {group_id: [task_indices]}
        self.task_groups = {}
        
        # PITA历史记录 (用于平滑)
        self.pita_history = {
            'intra': {},   # 组内 PITA: {(group_id, target_task): value}
            'inter': {}    # 组间 PITA: {(source_group, target_task): value}
        }
        
        # 参数快照 (用于计算PITA)
        self.param_snapshots = {}
        
        # 初始化随机分组
        self._initialize_random_groups()
    
    def _initialize_random_groups(self):
        """初始化随机任务分组"""
        num_tasks = len(self.task_names)
        task_indices = list(range(num_tasks))
        np.random.shuffle(task_indices)
        
        # 尽量均匀分配
        base_size = num_tasks // self.num_groups
        remainder = num_tasks % self.num_groups
        
        self.task_groups = {}
        idx = 0
        for g in range(self.num_groups):
            size = base_size + (1 if g < remainder else 0)
            self.task_groups[g] = task_indices[idx:idx+size]
            idx += size
        
        print(f"[STGU] 初始化分组: {self._format_groups()}")
    
    def _format_groups(self):
        """格式化输出当前分组信息"""
        group_info = {}
        for g_id, tasks in self.task_groups.items():
            task_names = [self.task_names[t] for t in tasks]
            group_info[f'G{g_id}'] = task_names
        return group_info
    
    def set_epoch(self, epoch):
        pass
    
    def encode_shared(self, context_ids, context_masks, token_type_ids,
                       sticker_txt_ids, sticertext_masks, stickertext_type_ids,
                       sticker_features):
        """共享的多模态编码器（与原模型完全一致）"""
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
        E_combined = torch.tanh(E_combined)
        
        shared_out = self.shared_mlp(E_combined)
        if self.shared_norm is not None:
            shared_out = self.shared_norm(shared_out)
        return shared_out
    
    def forward_single_task(self, shared_feature, task_idx):
        """前向传播单个任务"""
        task_name = self.task_names[task_idx]
        logits = self.task_heads[task_name](shared_feature)
        return logits
    
    def forward_all_tasks(self, shared_feature, labels_dict=None):
        """前向传播所有任务（用于评估）"""
        logits = {}
        for task_name in self.task_names:
            logits[task_name] = self.task_heads[task_name](shared_feature)
        
        if labels_dict is not None:
            task_losses = {}
            for task_name in self.task_names:
                target = labels_dict[task_name]
                task_losses[task_name] = F.cross_entropy(logits[task_name], target)
            return task_losses, logits
        else:
            return logits
    
    def forward(self, context_ids, context_masks, token_type_ids,
                sticker_txt_ids, sticertext_masks, stickertext_type_ids,
                sticker_features, labels_dict=None):
        """标准前向传播接口（与原模型兼容）"""
        shared_feature = self.encode_shared(
            context_ids, context_masks, token_type_ids,
            sticker_txt_ids, sticertext_masks, stickertext_type_ids,
            sticker_features
        )
        return self.forward_all_tasks(shared_feature, labels_dict)
    
    def compute_pita_intra(self, loss_before, loss_after):
        """
        计算组内 Proximal Inter-Task Affinity (Formula 4)
        B^i_{G←k} = 1 - L_k(z^{t+1}_{*|M}, Θ^{t+1}_{*|M}, Θ_k) / L_k(z^t, Θ_s^t, Θ_k^t)
        
        Args:
            loss_before: 更新前的损失 L_k(z^t, Θ_s^t, Θ_k^t)
            loss_after: 更新后的损失 L_k(z^{t+1}, Θ^{t+1}_{*|G}, Θ_k^t)
        
        Returns:
            pita_value: 组内PITA值，正值表示有益影响，负值表示有害影响
        """
        if loss_before == 0:
            return 0.0
        
        pita = 1 - (loss_after / loss_before)
        return pita
    
    def compute_pita_inter(self, loss_before, loss_after):
        """
        计算组间 Proximal Inter-Task Affinity (Formula 5)
        B^i_{G_j→i} = 1 - L_k(z^{t+1}_{*|M'}, Θ^{t+1}_{*|M'}, Θ_k) / L_k(z^t, Θ_s^t, Θ_k^t)
        
        与组内PITA相同公式，但来源是其他组的更新
        """
        return self.compute_pita_intra(loss_before, loss_after)
    
    def smooth_pita(self, pita_key, new_pita, pita_type='intra'):
        """
        使用指数移动平均平滑PITA值
        B^{t+1} = β * B^t + (1-β) * B^{measured}
        """
        if pita_key not in self.pita_history[pita_type]:
            self.pita_history[pita_type][pita_key] = new_pita
        else:
            old_pita = self.pita_history[pita_type][pita_key]
            smoothed = self.beta * old_pita + (1 - self.beta) * new_pita
            self.pita_history[pita_type][pita_key] = smoothed
        
        return self.pita_history[pita_type][pita_key]
    
    def save_param_snapshot(self, snapshot_name='before_update'):
        """保存参数快照（用于计算PITA）"""
        self.param_snapshots[snapshot_name] = {
            name: param.clone().detach() 
            for name, param in self.named_parameters()
        }
    
    def restore_param_snapshot(self, snapshot_name='before_update'):
        """恢复参数快照"""
        if snapshot_name not in self.param_snapshots:
            return
        
        state_dict = self.param_snapshots[snapshot_name]
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])
    
    def update_task_grouping(self, intra_pitas, inter_pitas):
        """
        基于PITA动态调整任务分组 (Algorithm 1 Line 8-11)
        
        规则 (Theorem 3):
        - 若 B^i_{G←k} ≥ 0 且 B^j_{G→k} ≥ 0 → 合并到同一组
        - 否则 → 分离到不同组
        
        Args:
            intra_pitas: {target_task: {source_group: pita_value}}
            inter_pitas: 同上格式
        """
        new_groups = defaultdict(list)
        merged_groups = set()
        
        for target_task in range(len(self.task_names)):
            best_group = None
            best_affinity = float('-inf')
            
            # 找到对目标任务最有益的源组
            for source_group in self.task_groups.keys():
                affinity = 0.0
                
                # 组内亲和度
                if (source_group, target_task) in intra_pitas:
                    affinity += intra_pitas[(source_group, target_task)]
                
                # 组间亲和度
                if (source_group, target_task) in inter_pitas:
                    affinity += inter_pitas[(source_group, target_task)]
                
                if affinity > best_affinity:
                    best_affinity = affinity
                    best_group = source_group
            
            if best_group is not None and best_affinity >= 0:
                new_groups[best_group].append(target_task)
                merged_groups.add(best_group)
            else:
                # 负亲和度任务创建新组或加入最差组
                new_group_id = max(self.task_groups.keys()) + 1 if self.task_groups else 0
                new_groups[new_group_id].append(target_task)
        
        # 确保每个组非空并重新编号
        final_groups = {}
        for new_g_id, tasks in new_groups.items():
            if tasks:  # 非空组
                final_groups[len(final_groups)] = tasks
        
        # 如果分组发生变化，打印日志
        if final_groups != self.task_groups:
            print(f"[STGU] 分组更新: {self._format_groups()} ->")
            self.task_groups = final_groups
            print(f"         -> {self._format_groups()}")
        
        return self.task_groups
    
    def get_group_tasks(self, group_id):
        """获取指定组的所有任务索引"""
        return self.task_groups.get(group_id, [])
    
    def get_task_group(self, task_idx):
        """获取指定任务所属的组ID"""
        for g_id, tasks in self.task_groups.items():
            if task_idx in tasks:
                return g_id
        return None