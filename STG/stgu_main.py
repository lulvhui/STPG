import os
import json
import torch
import datetime
import csv
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from my_dataload import *
from stgu_model import STGUModel

# ======================= 工具函数 =======================
def log_message(msg, log_file=None, to_console=True):
    if log_file is not None:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    if to_console:
        print(msg)

def evaluate(model, data_loader, device, task_names, desc=None):
    """评估函数（与原模型兼容）"""
    model.eval()
    total_loss = 0
    all_preds = {task: [] for task in task_names}
    all_labels = {task: [] for task in task_names}
    loader = tqdm(data_loader, desc=desc, leave=False) if desc else data_loader
    
    with torch.no_grad():
        for batch in loader:
            c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, *label_tensors = batch
            c_ids = c_ids.to(device)
            c_masks = c_masks.to(device)
            t_ids = t_ids.to(device)
            s_ids = s_ids.to(device)
            s_masks = s_masks.to(device)
            s_tids = s_tids.to(device)
            s_feats = s_feats.to(device)
            labels_dict = {task: label_tensors[i].to(device) for i, task in enumerate(task_names)}
            
            # 使用标准前向传播（所有任务同时）
            task_losses, logits = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats,
                                        labels_dict=labels_dict)
            total_loss += sum(l.item() for l in task_losses.values())
            
            for i, task in enumerate(task_names):
                pred = torch.argmax(logits[task], dim=1)
                all_preds[task].extend(pred.cpu().tolist())
                all_labels[task].extend(labels_dict[task].cpu().tolist())
    
    metrics = {}
    for task in task_names:
        preds = all_preds[task]
        labels = all_labels[task]
        if len(labels) == 0:
            metrics[task] = {'acc': 0.0, 'f1_weighted': 0.0, 'f1_macro': 0.0}
        else:
            acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
            f1_w = f1_score(labels, preds, average='weighted', zero_division=0)
            f1_m = f1_score(labels, preds, average='macro', zero_division=0)
            p = precision_score(labels, preds, average='macro', zero_division=0)
            r = recall_score(labels, preds, average='macro', zero_division=0)
            metrics[task] = {'acc': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m,
                             'precision': p, 'recall': r}
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, metrics

# ======================= 数据与路径准备 =======================
def load_and_filter_data():
    """数据加载（复用原函数）"""
    train_path = '/data/yyang/M^4Comment_Detection/multimodal_annotation/train.json'
    dev_path   = '/data/yyang/M^4Comment_Detection/multimodal_annotation/dev.json'
    test_path  = '/data/yyang/M^4Comment_Detection/multimodal_annotation/test.json'
    
    train_raw = load_data(train_path)
    valid_raw = load_data(dev_path)
    test_raw  = load_data(test_path)
    
    train = [x for x in train_raw if x['sticker'] in all_sticker_features]
    valid = [x for x in valid_raw if x['sticker'] in all_sticker_features]
    test  = [x for x in test_raw  if x['sticker'] in all_sticker_features]
    
    return train, valid, test

def setup_directories(timestamp):
    base_dir = '/home/lnhuang/yyang/M^4/reproduction/MSAIRS'
    script_dir = f'{base_dir}/script/{timestamp}'
    model_save_dir = f'/data/yyang/Temp_Model/2026-06-05/{timestamp}'
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    paths = {
        'script_dir': script_dir,
        'model_save_dir': model_save_dir,
        'log_file': os.path.join(script_dir, f'train_log_stgu_{timestamp}.txt'),
        'csv_file': os.path.join(script_dir, f'train_metrics_stgu_{timestamp}.csv'),
        'pita_csv': os.path.join(script_dir, f'pita_history_{timestamp}.csv'),
        'group_csv': os.path.join(script_dir, f'group_evolution_{timestamp}.csv'),
    }
    return paths

def create_dataloaders(train, valid, test):
    train_dataset = MyDataset(train)
    valid_dataset = MyDataset(valid)
    test_dataset  = MyDataset(test)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG['train_bs'], collate_fn=collate_fn,
                              shuffle=True, num_workers=CFG['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=CFG['valid_bs'], collate_fn=collate_fn,
                              shuffle=False, num_workers=CFG['num_workers'])
    test_loader  = DataLoader(test_dataset, batch_size=CFG['valid_bs'], collate_fn=collate_fn,
                             shuffle=False, num_workers=CFG['num_workers'])
    return train_loader, valid_loader, test_loader

# ======================= STGU核心训练算法 =======================
def stgu_train_step(model, optimizer, batch, device, task_names, epoch_info):
    """
    STGU单步训练 - 实现 Algorithm 1 的核心逻辑
    
    流程：
    1. 随机选择任务组更新序列
    2. 对每个组Gi：
       a. 前向传播计算Gi中所有任务的损失
       b. 更新共享参数和Gi的任务特定参数
       c. 计算PITA (组内和组间)
    3. 基于PITA调整分组
    """
    c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, *label_tensors = batch
    c_ids = c_ids.to(device)
    c_masks = c_masks.to(device)
    t_ids = t_ids.to(device)
    s_ids = s_ids.to(device)
    s_masks = s_masks.to(device)
    s_tids = s_tids.to(device)
    s_feats = s_feats.to(device)
    labels_dict = {task: label_tensors[i].to(device) for i, task in enumerate(task_names)}
    
    num_tasks = len(task_names)
    batch_loss = 0
    intra_pitas = {}   # 组内 PITA {(group_id, target_task): value}
    inter_pitas = {}   # 组间 PITA {(source_group, target_task): value}
    
    # Step 1: 随机选择任务组更新序列 (Algorithm 1 Line 3)
    group_sequence = list(model.task_groups.keys())
    np.random.shuffle(group_sequence)
    
    # 记录更新前的损失（用于PITA计算）
    losses_before_update = {}
    with torch.no_grad():
        shared_feature = model.encode_shared(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats)
        for task_idx in range(num_tasks):
            task_name = task_names[task_idx]
            logits = model.forward_single_task(shared_feature, task_idx)
            loss = F.cross_entropy(logits, labels_dict[task_name])
            losses_before_update[task_idx] = loss.item()
    
    # Step 2: 按顺序更新每个任务组 (Algorithm 1 Line 4-11)
    for step_idx, current_group_id in enumerate(group_sequence):
        group_tasks = model.get_group_tasks(current_group_id)
        if not group_tasks:
            continue
        
        # 2a: 前向传播当前组的任务 (Algorithm 1 Line 4)
        optimizer.zero_grad()
        shared_feature = model.encode_shared(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats)
        
        group_losses = {}
        for task_idx in group_tasks:
            task_name = task_names[task_idx]
            logits = model.forward_single_task(shared_feature, task_idx)
            loss = F.cross_entropy(logits, labels_dict[task_name])
            group_losses[task_idx] = loss
        
        # 组损失求和
        total_group_loss = sum(group_losses.values())
        
        # 2b: 反向传播并更新参数 (Algorithm 1 Line 5-6)
        # 只对共享参数和当前组的任务特定参数计算梯度
        total_group_loss.backward()
        
        # 创建掩码：只更新共享参数和当前组的任务头参数
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            # 判断是否是当前组的任务头参数
            is_current_group_param = False
            for task_idx in group_tasks:
                task_name = task_names[task_idx]
                if f'task_heads.{task_name}.' in name:
                    is_current_group_param = True
                    break
            
            # 共享参数 或 当前组参数 → 保留梯度
            # 其他组的任务参数 → 清零梯度
            is_shared_param = ('shared_mlp.' in name or 
                              'BertModel.' in name or 
                              'sticer_bert.' in name or
                              'sticker_cnn.' in name or
                              'cross_attention.' in name or
                              'diff_linear.' in name or
                              'sticker_align.' in name or
                              'sticker_attention.' in name or
                              'context_attention.' in name or
                              'fusion_fc.' in name or
                              'dropout.' in name or
                              ('shared_norm.' in name and model.shared_norm is not None))
            
            if not (is_shared_param or is_current_group_param):
                param.grad.zero_()
        
        optimizer.step()
        
        # 累加损失
        batch_loss += total_group_loss.item()
        
        # 2c: 计算PITA (Algorithm 1 Line 7)
        with torch.no_grad():
            shared_feature_after = model.encode_shared(
                c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats
            )
            
            # 计算所有目标任务的更新后损失
            losses_after_update = {}
            for target_task_idx in range(num_tasks):
                target_task_name = task_names[target_task_idx]  # 获取任务名称
                logits = model.forward_single_task(shared_feature_after, target_task_idx)
                loss_after = F.cross_entropy(logits, labels_dict[target_task_name])  # 使用名称作为key
                losses_after_update[target_task_idx] = loss_after.item()
            
            # 计算组内PITA (Formula 4)
            for target_task_idx in range(num_tasks):
                loss_bef = losses_before_update[target_task_idx]
                loss_aft = losses_after_update[target_task_idx]
                
                raw_pita = model.compute_pita_intra(loss_bef, loss_aft)
                smoothed_pita = model.smooth_pita(
                    (current_group_id, target_task_idx), 
                    raw_pita, 
                    pita_type='intra'
                )
                intra_pitas[(current_group_id, target_task_idx)] = smoothed_pita
                
                # 同时记录到组间PITA（对于非本组的任务）
                if target_task_idx not in group_tasks:
                    inter_pitas[(current_group_id, target_task_idx)] = smoothed_pita
    
    # Step 3: 动态调整任务分组 (Algorithm 1 Line 8-11)
    # 每 N 个batch或每个epoch调整一次（这里简化为每步都评估）
    if epoch_info['batch_count'] % CFG.get('group_update_freq', 10) == 0:
        model.update_task_grouping(intra_pitas, inter_pitas)
    
    return batch_loss, intra_pitas, inter_pitas

# ======================= 训练主循环 =======================
def train_model_stgu(model, train_loader, valid_loader, test_loader, device, task_names,
                     paths=None, patience=10, save_artifacts=True):
    """
    STGU训练主循环
    """
    log_file = paths['log_file'] if save_artifacts else None
    csv_file = paths['csv_file'] if save_artifacts else None
    pita_csv = paths.get('pita_csv') if save_artifacts else None
    group_csv = paths.get('group_csv') if save_artifacts else None
    
    log = lambda msg: log_message(msg, log_file, to_console=True)
    
    # 优化器配置（STGU可能需要稍大的学习率）
    lr = CFG['lr'] * 1.5  # STGU建议使用稍大学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    
    best_avg_f1_macro = 0.0
    no_improve = 0
    early_stop = False
    
    log(f"="*80)
    log("[STGU] ICLR 2025 Selective Task Group Updates Training")
    log(f"="*80)
    log(f"Tasks ({len(task_names)}): {task_names}")
    log(f"Initial groups: {model._format_groups()}")
    log(f"Learning rate: {lr} (base: {CFG['lr']} × 1.5)")
    log(f"PITA decay beta: {model.beta}")
    log(f"="*80)
    
    # 初始化CSV文件
    if save_artifacts and csv_file is not None:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'test_loss']
        for task in task_names:
            fieldnames += [f'val_{task}_acc', f'val_{task}_f1_w', f'val_{task}_f1_m',
                           f'test_{task}_acc', f'test_{task}_f1_w', f'test_{task}_f1_m']
        fieldnames += ['num_groups', 'avg_intra_pita', 'avg_inter_pita']
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    if save_artifacts and group_csv is not None:
        with open(group_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'group_id', 'tasks'])
    
    for epoch in range(1, CFG['epochs']+1):
        if early_stop:
            log("Early stopping triggered.")
            break
        
        model.train()
        train_loss_sum = 0
        epoch_intra_pitas = []
        epoch_inter_pitas = []
        epoch_info = {'epoch': epoch, 'batch_count': 0}
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [STGU]", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            epoch_info['batch_count'] = batch_idx + 1
            
            # STGU训练步骤
            batch_loss, intra_pitas, inter_pitas = stgu_train_step(
                model, optimizer, batch, device, task_names, epoch_info
            )
            
            train_loss_sum += batch_loss
            
            # 收集PITA统计信息
            if intra_pitas:
                epoch_intra_pitas.append(list(intra_pitas.values()))
            if inter_pitas:
                epoch_inter_pitas.append(list(inter_pitas.values()))
            
            # 更新进度条
            if batch_idx % 50 == 0:
                groups_info = model._format_groups()
                train_pbar.set_postfix({
                    'loss': f'{batch_loss:.3f}',
                    'groups': len(groups_info)
                })
        
        scheduler.step()
        train_loss_avg = train_loss_sum / len(train_loader)
        
        # 评估
        val_loss, val_metrics = evaluate(model, valid_loader, device, task_names, desc="Valid eval")
        test_loss, test_metrics = evaluate(model, test_loader, device, task_names, desc="Test eval")
        
        # 日志输出
        log(f"\nEpoch {epoch}: train_loss={train_loss_avg:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
        log(f"  Current grouping: {model._format_groups()} (groups: {len(model.task_groups)})")
        
        for task in task_names:
            log(f"  {task:12s}: val_acc={val_metrics[task]['acc']:.4f} test_acc={test_metrics[task]['acc']:.4f} | "
                f"val_f1={val_metrics[task]['f1_weighted']:.4f} test_f1={test_metrics[task]['f1_weighted']:.4f}")
        
        avg_val_f1m = sum(m['f1_macro'] for m in val_metrics.values()) / len(val_metrics)
        avg_test_f1m = sum(m['f1_macro'] for m in test_metrics.values()) / len(test_metrics)
        log(f"  AVG macro-F1: val={avg_val_f1m:.4f} test={avg_test_f1m:.4f}")
        
        # PITA统计
        avg_intra = np.mean([p for plist in epoch_intra_pitas for p in plist]) if epoch_intra_pitas else 0
        avg_inter = np.mean([p for plist in epoch_inter_pitas for p in plist]) if epoch_inter_pitas else 0
        log(f"  PITA stats: intra_avg={avg_intra:.4f}, inter_avg={avg_inter:.4f}")
        
        # 保存指标到CSV
        if save_artifacts and csv_file is not None:
            row = {
                'epoch': epoch,
                'train_loss': train_loss_avg,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'num_groups': len(model.task_groups),
                'avg_intra_pita': avg_intra,
                'avg_inter_pita': avg_inter
            }
            metric_map = [('acc', 'acc'), ('f1_w', 'f1_weighted'), ('f1_m', 'f1_macro')]
            for task in task_names:
                for csv_suffix, metric_key in metric_map:
                    row[f'val_{task}_{csv_suffix}'] = val_metrics[task][metric_key]
                    row[f'test_{task}_{csv_suffix}'] = test_metrics[task][metric_key]
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
        
        # 保存分组演化
        if save_artifacts and group_csv is not None:
            with open(group_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for g_id, tasks in model.task_groups.items():
                    task_names_str = ','.join([task_names[t] for t in tasks])
                    writer.writerow([epoch, f'G{g_id}', task_names_str])
        
        # Early stopping检查
        if avg_val_f1m > best_avg_f1_macro + 1e-4:
            best_avg_f1_macro = avg_val_f1m
            no_improve = 0
            
            if save_artifacts:
                ckpt_name = f'checkpoint_stgu_{paths["timestamp"]}.pt'
                ckpt_path = os.path.join(paths['model_save_dir'], ckpt_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'task_groups': model.task_groups,
                    'best_f1_macro': best_avg_f1_macro,
                }, ckpt_path)
                log(f"✓ Best model saved! (macro-F1={avg_val_f1m:.4f})")
                
                # 保存测试结果
                test_res_name = f'test_predictions_stgu_{paths["timestamp"]}.json'
                test_res_path = os.path.join(paths['model_save_dir'], test_res_name)
                with open(test_res_path, 'w', encoding='utf-8') as f:
                    json.dump(test_metrics, f, indent=2)
        else:
            no_improve += 1
            log(f"No improvement for {no_improve} epoch(s). Best: {best_avg_f1_macro:.4f}")
            if no_improve >= patience:
                early_stop = True
    
    test_avg_macro_f1 = sum(m['f1_macro'] for m in test_metrics.values()) / len(test_metrics)
    return best_avg_f1_macro, test_metrics, test_avg_macro_f1

# ======================= 主入口函数 =======================
def run_training_stgu(task_hidden_dim=768,
                      use_ln=True,
                      num_groups=3,
                      beta=0.9,
                      save_artifacts=True):
    """
    运行STGU训练
    
    Args:
        task_hidden_dim: 任务隐藏层维度
        use_ln: 是否使用LayerNorm
        num_groups: 初始任务组数量
        beta: PITA衰减率
        save_artifacts: 是否保存日志和模型
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = setup_directories(timestamp) if save_artifacts else None
    if save_artifacts:
        paths['timestamp'] = timestamp
    
    train, valid, test = load_and_filter_data()
    log = lambda msg: log_message(msg, paths['log_file'] if save_artifacts else None, to_console=True)
    
    log(f"\n{'='*80}")
    log(f"[STGU] ICLR 2025 Baseline Experiment")
    log(f"{'='*80}")
    log(f"Data: train={len(train)}, valid={len(valid)}, test={len(test)}")
    log(f"Config: hidden_dim={task_hidden_dim}, ln={use_ln}, groups={num_groups}, beta={beta}")
    log(f"Tasks: {list(TASK_CONFIG.keys())}")
    log(f"{'='*80}\n")
    
    train_loader, valid_loader, test_loader = create_dataloaders(train, valid, test)
    task_names = list(TASK_CONFIG.keys())
    device = torch.device(CFG['device'])
    
    # 创建STGU模型
    model = STGUModel(
        task_hidden_dim=task_hidden_dim,
        use_ln=use_ln,
        num_groups=num_groups,
        beta=beta
    ).to(device)
    
    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model params: {total_params:,} (trainable: {trainable_params:,})")
    
    # 开始训练
    best_val_f1, test_metrics, test_avg_f1 = train_model_stgu(
        model, train_loader, valid_loader, test_loader, device, task_names,
        paths=paths, patience=CFG['patience'], save_artifacts=save_artifacts
    )
    
    log(f"\n{'='*80}")
    log(f"[STGU] Training Completed!")
    log(f"Best validation macro-F1: {best_val_f1:.4f}")
    log(f"Test macro-F1: {test_avg_f1:.4f}")
    log(f"Final grouping: {model._format_groups()}")
    log(f"{'='*80}\n")
    
    return {
        'method': 'STGU-ICLR2025',
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'test_avg_macro_f1': test_avg_f1,
        'final_groups': model._format_groups(),
        'model_file': os.path.join(paths['model_save_dir'], f'checkpoint_stgu_{timestamp}.pt') if save_artifacts else None,
        'script_dir': paths['script_dir'] if save_artifacts else None,
        'timestamp': timestamp
    }


if __name__ == '__main__':
    result = run_training_stgu(
        task_hidden_dim=768,
        use_ln=True,
        num_groups=3,      # 初始分为3组
        beta=0.9,          # PITA衰减率
        save_artifacts=True
    )
    
    print(f"\n{'='*80}")
    print(f"✓ STGU Training Completed Successfully!")
    print(f"{'='*80}")
    
    print(f"\n[Basic Info]")
    print(f"  Method:           {result['method']}")
    print(f"  Best Val F1:      {result['best_val_f1']:.4f}")
    print(f"  Test Macro-F1:    {result['test_avg_macro_f1']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"[Test Set Detailed Metrics - All 9 Tasks]")
    print(f"{'='*80}")
    
    test_metrics = result['test_metrics']
    task_names = list(test_metrics.keys())
    
    # 计算总体平均指标
    all_accs = []
    all_macro_f1s = []
    all_weighted_f1s = []
    
    # 输出每个任务的详细指标
    print(f"\n{'Task':<15} {'Accuracy':>10} {'Macro-F1':>10} {'Weight-F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"{'-'*75}")
    
    for task in task_names:
        metrics = test_metrics[task]
        acc = metrics['acc']
        macro_f1 = metrics['f1_macro']
        weighted_f1 = metrics['f1_weighted']
        precision = metrics['precision']
        recall = metrics['recall']
        
        all_accs.append(acc)
        all_macro_f1s.append(macro_f1)
        all_weighted_f1s.append(weighted_f1)
        
        print(f"{task:<15} {acc:>10.4f} {macro_f1:>10.4f} {weighted_f1:>10.4f} {precision:>10.4f} {recall:>10.4f}")
    
    # 计算并输出总体平均值
    avg_acc = sum(all_accs) / len(all_accs)
    avg_macro_f1 = sum(all_macro_f1s) / len(all_macro_f1s)
    avg_weighted_f1 = sum(all_weighted_f1s) / len(all_weighted_f1s)
    
    print(f"{'-'*75}")
    print(f"{'AVERAGE':<15} {avg_acc:>10.4f} {avg_macro_f1:>10.4f} {avg_weighted_f1:>10.4f}")
    
    print(f"\n{'='*80}")
    print(f"[Summary Statistics]")
    print(f"{'='*80}")
    print(f"  📊 Average Accuracy:     {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print(f"  📊 Average Macro-F1:     {avg_macro_f1:.4f} ({avg_macro_f1*100:.2f}%)")
    print(f"  📊 Average Weighted-F1:  {avg_weighted_f1:.4f} ({avg_weighted_f1*100:.2f}%)")
    
    # 找出最佳和最差任务
    best_task_idx = all_accs.index(max(all_accs))
    worst_task_idx = all_accs.index(min(all_accs))
    
    print(f"\n  🏆 Best Task:  {task_names[best_task_idx]:<12} (Acc={all_accs[best_task_idx]:.4f})")
    print(f"  ⚠️  Worst Task: {task_names[worst_task_idx]:<12} (Acc={all_accs[worst_task_idx]:.4f})")
    
    print(f"\n  📁 Results saved to: {result['script_dir']}")
    print(f"  📁 Model checkpoint:  {result['model_file']}")
    print(f"\n{'='*80}\n")