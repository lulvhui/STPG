import os
import json
import torch
import datetime
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from my_dataload import *
from my_model import MultiTaskModel
from my_dataload import MMOE

# ======================= 工具函数 =======================
def log_message(msg, log_file=None, to_console=True):
    if log_file is not None:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    if to_console:
        print(msg)

def evaluate(model, data_loader, device, task_names, desc=None):
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
            loss, logits, _ = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats,
                                     labels_dict=labels_dict)
            total_loss += loss.item()
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

def compute_avg_metrics(metrics):
    n = len(metrics)
    avg_acc = sum(m['acc'] for m in metrics.values()) / n
    avg_f1_macro = sum(m['f1_macro'] for m in metrics.values()) / n
    avg_f1_weighted = sum(m['f1_weighted'] for m in metrics.values()) / n
    return avg_acc, avg_f1_macro, avg_f1_weighted

def record_gate_weights(model, epoch, train_loader, device, task_names, gate_csv):
    """记录各 epoch 的 gate 权重到 CSV"""
    if epoch == 1 and gate_csv is not None:
        columns = ['epoch'] + [f'{task}_expert_{i}' for task in task_names for i in range(model.num_experts)]
        with open(gate_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

# ======================= 数据与路径准备 =======================
def load_and_filter_data():
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
        'log_file': os.path.join(script_dir, f'train_log_moe_{timestamp}.txt'),
        'csv_file': os.path.join(script_dir, f'train_metrics_moe_{timestamp}.csv'),
        'gate_csv': os.path.join(script_dir, f'gate_weights_{timestamp}.csv'),
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

# ======================= 训练主循环 =======================
def train_model(model, train_loader, valid_loader, test_loader, device, task_names,
                paths=None, patience=10, save_artifacts=True):
    log_file = paths['log_file'] if save_artifacts else None
    csv_file = paths['csv_file'] if save_artifacts else None
    gate_csv = paths['gate_csv'] if save_artifacts else None

    log = lambda msg: log_message(msg, log_file, to_console=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    best_avg_f1_macro = 0.0
    best_test_metrics = None
    no_improve = 0
    early_stop = False

    if save_artifacts and csv_file is not None:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'test_loss']
        for task in task_names:
            fieldnames += [f'val_{task}_acc', f'val_{task}_f1_w', f'val_{task}_f1_m',
                           f'test_{task}_acc', f'test_{task}_f1_w', f'test_{task}_f1_m']
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(1, CFG['epochs']+1):
        if early_stop:
            log("Early stopping triggered.")
            break
        model.set_epoch(epoch)
        model.train()
        train_loss_sum = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
        for batch in train_pbar:
            c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, *label_tensors = batch
            c_ids = c_ids.to(device)
            c_masks = c_masks.to(device)
            t_ids = t_ids.to(device)
            s_ids = s_ids.to(device)
            s_masks = s_masks.to(device)
            s_tids = s_tids.to(device)
            s_feats = s_feats.to(device)
            labels_dict = {task: label_tensors[i].to(device) for i, task in enumerate(task_names)}
            total_loss, _, _ = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, labels_dict)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss_sum += total_loss.item()
        train_loss_avg = train_loss_sum / len(train_loader)

        val_loss, val_metrics = evaluate(model, valid_loader, device, task_names, desc="Valid eval")
        test_loss, test_metrics = evaluate(model, test_loader, device, task_names, desc="Test eval")

        val_avg_acc, val_avg_f1m, val_avg_f1w = compute_avg_metrics(val_metrics)
        test_avg_acc, test_avg_f1m, test_avg_f1w = compute_avg_metrics(test_metrics)
        log(f"Epoch {epoch}: train_loss={train_loss_avg:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
        log(f"  [AVG]  val_acc={val_avg_acc:.4f} val_f1w={val_avg_f1w:.4f} val_f1m={val_avg_f1m:.4f} | test_acc={test_avg_acc:.4f} test_f1w={test_avg_f1w:.4f} test_f1m={test_avg_f1m:.4f}")
        for task in task_names:
            log(f"  {task}:  val_acc={val_metrics[task]['acc']:.4f} test_acc={test_metrics[task]['acc']:.4f}")
            log(f"  {task}:  val_f1={val_metrics[task]['f1_weighted']:.4f} test_f1={test_metrics[task]['f1_weighted']:.4f}")
            log(f"  {task}:  val_mf1={val_metrics[task]['f1_macro']:.4f} test_mf1={test_metrics[task]['f1_macro']:.4f}")

        if save_artifacts and csv_file is not None:
            row = {'epoch': epoch, 'train_loss': train_loss_avg, 'val_loss': val_loss, 'test_loss': test_loss}
            metric_map = [('acc', 'acc'), ('f1_w', 'f1_weighted'), ('f1_m', 'f1_macro')]
            for task in task_names:
                for csv_suffix, metric_key in metric_map:
                    row[f'val_{task}_{csv_suffix}']   = val_metrics[task][metric_key]
                    row[f'test_{task}_{csv_suffix}']  = test_metrics[task][metric_key]
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

        if save_artifacts and gate_csv is not None:
            record_gate_weights(model, epoch, train_loader, device, task_names, gate_csv)

        avg_val_f1_macro = sum(m['f1_macro'] for m in val_metrics.values()) / len(val_metrics)
        if avg_val_f1_macro > best_avg_f1_macro + 1e-4:
            best_avg_f1_macro = avg_val_f1_macro
            no_improve = 0
            best_test_metrics = test_metrics
            if save_artifacts:
                ckpt_name = f'checkpoint_mmoe_{paths["timestamp"]}.pt'
                ckpt_path = os.path.join(paths['model_save_dir'], ckpt_name)
                torch.save(model.state_dict(), ckpt_path)
                log(f"Best model saved to {ckpt_path} (avg_val_f1_macro={avg_val_f1_macro:.4f})")
                test_res_name = f'test_predictions_mmoe_{paths["timestamp"]}.json'
                test_res_path = os.path.join(paths['model_save_dir'], test_res_name)
                with open(test_res_path, 'w', encoding='utf-8') as f:
                    json.dump(test_metrics, f, indent=2)
        else:
            no_improve += 1
            log(f"No improvement for {no_improve} epoch(s).")
            if no_improve >= patience:
                early_stop = True

    if best_test_metrics is not None:
        b_acc, b_f1m, b_f1w = compute_avg_metrics(best_test_metrics)
        log(f"\n{'='*60}")
        log(f"Best model test results (avg across tasks):")
        log(f"  avg_acc={b_acc:.4f}  avg_f1_weighted={b_f1w:.4f}  avg_f1_macro={b_f1m:.4f}")
        log(f"{'='*60}")
    test_avg_macro_f1 = sum(m['f1_macro'] for m in test_metrics.values()) / len(test_metrics)
    return best_avg_f1_macro, test_metrics, test_avg_macro_f1

# ======================= 主入口函数 =======================
def run_training(moe_params=None,
                 use_ln=True,
                 save_artifacts=True):
    from my_dataload import MMOE as default_mmoe
    if moe_params is None:
        mmoe_params = default_mmoe.copy()
    else:
        mmoe_params = {**default_mmoe, **moe_params}

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = setup_directories(timestamp) if save_artifacts else None
    if save_artifacts:
        paths['timestamp'] = timestamp

    train, valid, test = load_and_filter_data()
    log = lambda msg: log_message(msg, paths['log_file'] if save_artifacts else None, to_console=True)
    log(f"过滤后：train {len(train)}, valid {len(valid)}, test {len(test)}")

    train_loader, valid_loader, test_loader = create_dataloaders(train, valid, test)
    task_names = list(TASK_CONFIG.keys())
    device = torch.device(CFG['device'])

    model = MultiTaskModel(
        num_experts=mmoe_params['num_experts'],
        expert_hidden_dim=mmoe_params['expert_hidden'],
        use_ln=use_ln
    ).to(device)

    best_val_f1, test_metrics, test_avg_f1 = train_model(
        model, train_loader, valid_loader, test_loader, device, task_names,
        paths=paths, patience=CFG['patience'], save_artifacts=save_artifacts
    )

    return {
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'test_avg_macro_f1': test_avg_f1,
        'model_file': os.path.join(paths['model_save_dir'], f'checkpoint_mmoe_{timestamp}.pt') if save_artifacts else None,
        'gate_csv': paths['gate_csv'] if save_artifacts else None,
        'script_dir': paths['script_dir'] if save_artifacts else None,
        'timestamp': timestamp
    }


if __name__ == '__main__':
    result = run_training(save_artifacts=True)
    print(f"Best val macro-F1: {result['best_val_f1']:.4f}")