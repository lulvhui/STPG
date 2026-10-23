import os
import json
import torch
import datetime
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from my_dataload import *
from my_model_single import SingleTaskModel

# ========== 路径配置 ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
model_save_dir = '/data/yyang/Temp_Model/2026-05-31'
os.makedirs(model_save_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(script_dir, f'train_log_single_{timestamp}.txt')
# 所有任务共用一个 CSV
csv_file = os.path.join(script_dir, f'train_metrics_single_{timestamp}.csv')

def log_message(msg, to_console=True):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    if to_console:
        print(msg)

# ========== 数据准备 ==========
train_path = '/data/yyang/M^4Comment_Detection/multimodal_annotation/train.json'
dev_path = '/data/yyang/M^4Comment_Detection/multimodal_annotation/dev.json'
test_path = '/data/yyang/M^4Comment_Detection/multimodal_annotation/test.json'

log_message("Loading data...")
train_raw = load_data(train_path)
valid_raw = load_data(dev_path)
test_raw = load_data(test_path)

log_message(f"Before filtering: train {len(train_raw)}, valid {len(valid_raw)}, test {len(test_raw)}")
train = [x for x in train_raw if x['sticker'] in all_sticker_features]
valid = [x for x in valid_raw if x['sticker'] in all_sticker_features]
test  = [x for x in test_raw  if x['sticker'] in all_sticker_features]
log_message(f"After filtering: train {len(train)}, valid {len(valid)}, test {len(test)}")

test_dataset = MyDataset(test)
test_loader = DataLoader(test_dataset, batch_size=CFG['valid_bs'], collate_fn=collate_fn,
                         shuffle=False, num_workers=CFG['num_workers'])

TASK_ORDER = list(TASK_CONFIG.keys())  # 标签顺序

def evaluate(model, data_loader, device, task_name):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    label_idx = TASK_ORDER.index(task_name)
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
            labels = label_tensors[label_idx].to(device)
            loss, logits = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) if all_labels else 0
    f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if all_labels else 0
    f1_m = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0
    return total_loss / len(data_loader), acc, f1_w, f1_m, all_preds, all_labels

def train_single_task(task_name, num_classes, train_loader, valid_loader, test_loader, device, csv_writer, csv_fields):
    model = SingleTaskModel(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    patience = CFG['patience']
    best_metric = 0.0
    no_improve = 0
    early_stop = False

    for epoch in range(1, CFG['epochs']+1):
        if early_stop:
            log_message(f"Task {task_name}: Early stopped at epoch {epoch}")
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
            labels = label_tensors[TASK_ORDER.index(task_name)].to(device)
            loss, logits = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估三个数据集
        train_loss, train_acc, train_f1_w, train_f1_m, _, _ = evaluate(model, train_loader, device, task_name)
        val_loss, val_acc, val_f1_w, val_f1_m, _, _ = evaluate(model, valid_loader, device, task_name)
        test_loss, test_acc, test_f1_w, test_f1_m, _, _ = evaluate(model, test_loader, device, task_name)

        log_message(f"Task {task_name} Epoch {epoch}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
        log_message(f"  train: acc={train_acc:.4f} f1_w={train_f1_w:.4f} f1_m={train_f1_m:.4f}")
        log_message(f"  val:   acc={val_acc:.4f} f1_w={val_f1_w:.4f} f1_m={val_f1_m:.4f}")
        log_message(f"  test:  acc={test_acc:.4f} f1_w={test_f1_w:.4f} f1_m={test_f1_m:.4f}")

        # 写入 CSV（所有任务共用同一个 csv_writer）
        csv_writer.writerow({
            'task': task_name,
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc, 'train_f1_w': train_f1_w, 'train_f1_m': train_f1_m,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1_w': val_f1_w, 'val_f1_m': val_f1_m,
            'test_loss': test_loss, 'test_acc': test_acc, 'test_f1_w': test_f1_w, 'test_f1_m': test_f1_m
        })

        current_metric = val_f1_m
        if current_metric > best_metric + 1e-4:
            best_metric = current_metric
            no_improve = 0
            ckpt_name = f"checkpoint_single_{task_name}_{timestamp}.pt"
            ckpt_path = os.path.join(model_save_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            log_message(f"Task {task_name}: best model saved to {ckpt_path} (val_acc={val_acc:.4f})")

            test_res_name = f"test_predictions_{task_name}_{timestamp}.json"
            test_res_path = os.path.join(model_save_dir, test_res_name)
            _, _, _, _, preds, labels = evaluate(model, test_loader, device, task_name)
            results = [{'sample_id': i, 'true_label': l, 'pred_label': p} for i, (l, p) in enumerate(zip(labels, preds))]
            with open(test_res_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            log_message(f"Test predictions saved to {test_res_path}")
        else:
            no_improve += 1
            log_message(f"Task {task_name}: no improvement for {no_improve} epoch(s)")
            if no_improve >= patience:
                early_stop = True

    log_message(f"Task {task_name} training finished. Best val_acc={best_metric:.4f}")

if __name__ == '__main__':
    device = torch.device(CFG['device'])
    train_dataset = MyDataset(train)
    valid_dataset = MyDataset(valid)
    train_loader = DataLoader(train_dataset, batch_size=CFG['train_bs'], collate_fn=collate_fn,
                              shuffle=True, num_workers=CFG['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=CFG['valid_bs'], collate_fn=collate_fn,
                              shuffle=False, num_workers=CFG['num_workers'])

    # 创建统一的 CSV 文件
    csv_fields = ['task', 'epoch',
                  'train_loss', 'train_acc', 'train_f1_w', 'train_f1_m',
                  'val_loss', 'val_acc', 'val_f1_w', 'val_f1_m',
                  'test_loss', 'test_acc', 'test_f1_w', 'test_f1_m']
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        # 传入同一个 writer，让每个任务追加写入
        for task_name in TASK_ORDER:
            num_classes = TASK_CONFIG[task_name]['num_classes']
            log_message(f"===== Start training for task: {task_name} ({num_classes} classes) =====")
            train_single_task(task_name, num_classes, train_loader, valid_loader, test_loader,
                              device, writer, csv_fields)

    log_message("All single-task trainings finished.")
    log_message(f"Training metrics CSV saved to {csv_file}")