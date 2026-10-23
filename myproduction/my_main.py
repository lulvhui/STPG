import os
import json
import torch
import datetime
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from my_dataload import *
from my_model import MultiTaskModel

# ========== 路径配置 ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
model_save_dir = '/data/yyang/Temp_Model/2026-05-31'
os.makedirs(model_save_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(script_dir, f'train_log_multitask_{timestamp}.txt')
csv_file = os.path.join(script_dir, f'train_metrics_multitask_{timestamp}.csv')

def log_message(msg, to_console=True):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    if to_console:
        print(msg)

# ========== 数据加载 ==========
train_path = '/data/yyang/M^4Comment_Detection/multimodal_annotation/train.json'
dev_path = '/data/yyang/M^4Comment_Detection/multimodal_annotation/dev.json'
test_path = '/data/yyang/M^4Comment_Detection/multimodal_annotation/test.json'

log_message("Loading training data...")
train_raw = load_data(train_path)
log_message("Loading validation data...")
valid_raw = load_data(dev_path)
log_message("Loading test data...")
test_raw = load_data(test_path)

log_message(f"过滤前：train {len(train_raw)}, valid {len(valid_raw)}, test {len(test_raw)}")
train = [x for x in train_raw if x['sticker'] in all_sticker_features]
valid = [x for x in valid_raw if x['sticker'] in all_sticker_features]
test  = [x for x in test_raw  if x['sticker'] in all_sticker_features]
log_message(f"过滤后：train {len(train)}, valid {len(valid)}, test {len(test)}")

test_dataset = MyDataset(test)
test_loader = DataLoader(test_dataset, batch_size=CFG['valid_bs'], collate_fn=collate_fn,
                         shuffle=False, num_workers=CFG['num_workers'])

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = {task: [] for task in TASK_CONFIG}
    all_labels = {task: [] for task in TASK_CONFIG}
    with torch.no_grad():
        for batch in data_loader:
            c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, *label_tensors = batch
            c_ids = c_ids.to(device)
            c_masks = c_masks.to(device)
            t_ids = t_ids.to(device)
            s_ids = s_ids.to(device)
            s_masks = s_masks.to(device)
            s_tids = s_tids.to(device)
            s_feats = s_feats.to(device)

            labels_dict = {task: label_tensors[i].to(device) for i, task in enumerate(TASK_CONFIG)}
            loss, logits = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, labels_dict)
            total_loss += loss.item()

            for i, task in enumerate(TASK_CONFIG):
                pred = torch.argmax(logits[task], dim=1)
                all_preds[task].extend(pred.cpu().tolist())
                all_labels[task].extend(labels_dict[task].cpu().tolist())

    metrics = {}
    for task in TASK_CONFIG:
        preds = all_preds[task]
        labels = all_labels[task]
        if len(labels) == 0:
            metrics[task] = {'acc': 0.0, 'f1_weighted': 0.0, 'f1_macro': 0.0}
        else:
            acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
            f1_w = f1_score(labels, preds, average='weighted', zero_division=0)
            f1_m = f1_score(labels, preds, average='macro', zero_division=0)
            metrics[task] = {'acc': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m}
    avg_loss = total_loss / len(data_loader)
    return avg_loss, metrics

def train_multitask(model, train_loader, valid_loader, optimizer, device):
    patience = CFG['patience']
    best_avg_metric = 0.0
    no_improve = 0
    early_stop = False

    # 准备 CSV 文件
    fieldnames = ['epoch']
    for split in ['train', 'val', 'test']:
        fieldnames.append(f'{split}_loss')
        for task in TASK_CONFIG:
            fieldnames.append(f'{split}_{task}_acc')
            fieldnames.append(f'{split}_{task}_f1_w')
            fieldnames.append(f'{split}_{task}_f1_m')

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, CFG['epochs']+1):
        if early_stop:
            log_message("Early stopping triggered.")
            break

        model.train()
        for batch in train_loader:
            c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, *label_tensors = batch
            c_ids = c_ids.to(device)
            c_masks = c_masks.to(device)
            t_ids = t_ids.to(device)
            s_ids = s_ids.to(device)
            s_masks = s_masks.to(device)
            s_tids = s_tids.to(device)
            s_feats = s_feats.to(device)
            labels_dict = {task: label_tensors[i].to(device) for i, task in enumerate(TASK_CONFIG)}
            loss, _ = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, labels_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估三个数据集
        train_loss, train_metrics = evaluate(model, train_loader, device)
        val_loss, val_metrics = evaluate(model, valid_loader, device)
        test_loss, test_metrics = evaluate(model, test_loader, device)

        log_message(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
        for task in TASK_CONFIG:
            log_message(f"  {task}: "
                        f"train_acc={train_metrics[task]['acc']:.4f} "
                        f"val_acc={val_metrics[task]['acc']:.4f} "
                        f"test_acc={test_metrics[task]['acc']:.4f}")
            log_message(f"  {task}: "
                        f"train_f1={train_metrics[task]['f1_weighted']:.4f} "
                        f"val_f1={val_metrics[task]['f1_weighted']:.4f} "
                        f"test_f1={test_metrics[task]['f1_weighted']:.4f}")            
            log_message(f"  {task}: "
                        f"train_mf1={train_metrics[task]['f1_macro']:.4f} "
                        f"val_mf1={val_metrics[task]['f1_macro']:.4f} "
                        f"test_mf1={test_metrics[task]['f1_macro']:.4f}")
        # 写入 CSV
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
        for task in TASK_CONFIG:
            row[f'train_{task}_acc'] = train_metrics[task]['acc']
            row[f'train_{task}_f1_w'] = train_metrics[task]['f1_weighted']
            row[f'train_{task}_f1_m'] = train_metrics[task]['f1_macro']
            row[f'val_{task}_acc'] = val_metrics[task]['acc']
            row[f'val_{task}_f1_w'] = val_metrics[task]['f1_weighted']
            row[f'val_{task}_f1_m'] = val_metrics[task]['f1_macro']
            row[f'test_{task}_acc'] = test_metrics[task]['acc']
            row[f'test_{task}_f1_w'] = test_metrics[task]['f1_weighted']
            row[f'test_{task}_f1_m'] = test_metrics[task]['f1_macro']
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        # 早停与模型保存（基于验证集平均 acc）
        avg_val_acc = sum(m['f1_macro'] for m in val_metrics.values()) / len(val_metrics)
        if avg_val_acc > best_avg_metric + 1e-4:
            best_avg_metric = avg_val_acc
            no_improve = 0
            ckpt_name = f'checkpoint_multitask_{timestamp}.pt'
            ckpt_path = os.path.join(model_save_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            log_message(f"Best model saved to {ckpt_path} (avg_f1_macro={avg_val_acc:.4f})")

            # 保存测试集详细预测结果
            test_res_name = f'test_predictions_multitask_{timestamp}.json'
            test_res_path = os.path.join(model_save_dir, test_res_name)
            with open(test_res_path, 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, indent=2)
            log_message(f"Test metrics saved to {test_res_path}")
        else:
            no_improve += 1
            log_message(f"No improvement for {no_improve} epoch(s).")
            if no_improve >= patience:
                early_stop = True

if __name__ == '__main__':
    train_dataset = MyDataset(train)
    valid_dataset = MyDataset(valid)
    train_loader = DataLoader(train_dataset, batch_size=CFG['train_bs'], collate_fn=collate_fn,
                              shuffle=True, num_workers=CFG['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=CFG['valid_bs'], collate_fn=collate_fn,
                              shuffle=False, num_workers=CFG['num_workers'])

    device = torch.device(CFG['device'])
    model = MultiTaskModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    train_multitask(model, train_loader, valid_loader, optimizer, device)