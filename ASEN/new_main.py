import os
import csv
import torch
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from my_dataload import *
from my_model import SingleTaskModel

# ======================= 评估函数 =======================
def evaluate_single(model, data_loader, device, task_idx, task_names, desc=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    loader = tqdm(data_loader, desc=desc, leave=False) if desc else data_loader
    with torch.no_grad():
        for batch in loader:
            # 解包
            c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, *label_tensors = batch
            c_ids = c_ids.to(device)
            c_masks = c_masks.to(device)
            t_ids = t_ids.to(device)
            s_ids = s_ids.to(device)
            s_masks = s_masks.to(device)
            s_tids = s_tids.to(device)
            s_feats = s_feats.to(device)

            labels = label_tensors[task_idx].to(device)
            loss, logits = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, labels=labels)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    if len(all_labels) == 0:
        metrics = {'acc': 0.0, 'f1_weighted': 0.0, 'f1_macro': 0.0,
                   'precision': 0.0, 'recall': 0.0}
    else:
        acc = accuracy_score(all_labels, all_preds)
        f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics = {'acc': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m,
                   'precision': p, 'recall': r}
    return avg_loss, metrics


# ======================= 数据准备（复用原 my_main 的逻辑） =======================
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


# ======================= 单任务训练 =======================
def train_single_task(task_name, train_loader, valid_loader, test_loader,
                      device, task_names, patience=10):
    task_idx = task_names.index(task_name)
    num_classes = TASK_CONFIG[task_name]['num_classes']

    model = SingleTaskModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    best_val_f1_macro = 0.0
    best_test_metrics = None
    no_improve = 0

    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        train_loss_sum = 0.0
        train_pbar = tqdm(train_loader, desc=f"{task_name} Epoch {epoch} Train", leave=False)
        for batch in train_pbar:
            c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, *label_tensors = batch
            c_ids = c_ids.to(device)
            c_masks = c_masks.to(device)
            t_ids = t_ids.to(device)
            s_ids = s_ids.to(device)
            s_masks = s_masks.to(device)
            s_tids = s_tids.to(device)
            s_feats = s_feats.to(device)

            labels = label_tensors[task_idx].to(device)
            loss, _ = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        val_loss, val_metrics = evaluate_single(model, valid_loader, device, task_idx,
                                                task_names, desc=f"{task_name} Valid")
        test_loss, test_metrics = evaluate_single(model, test_loader, device, task_idx,
                                                  task_names, desc=f"{task_name} Test")

        # 记录最佳验证时的测试指标
        if val_metrics['f1_macro'] > best_val_f1_macro + 1e-4:
            best_val_f1_macro = val_metrics['f1_macro']
            best_test_metrics = test_metrics
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping for task {task_name} at epoch {epoch}")
                break

    return best_val_f1_macro, best_test_metrics


# ======================= 主入口 =======================
def main():
    train, valid, test = load_and_filter_data()
    print(f"Data: train {len(train)}, valid {len(valid)}, test {len(test)}")

    device = torch.device(CFG['device'])
    task_names = list(TASK_CONFIG.keys())

    train_loader, valid_loader, test_loader = create_dataloaders(train, valid, test)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"single_task_baseline_{timestamp}.csv"
    fieldnames = ['task', 'test_acc', 'test_f1_weighted', 'test_f1_macro', 'test_precision', 'test_recall']

    all_results = []

    for task in task_names:
        print(f"\n===== Training single-task model for: {task} =====")
        val_f1, test_metrics = train_single_task(task, train_loader, valid_loader, test_loader,
                                                 device, task_names, patience=CFG['patience'])
        record = {
            'task': task,
            'test_acc': test_metrics['acc'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall']
        }
        all_results.append(record)
        print(f"  Best Val F1-macro: {val_f1:.4f}  |  "
              f"Test Acc: {record['test_acc']:.4f}  |  "
              f"Test F1-macro: {record['test_f1_macro']:.4f}")

    # 保存汇总 CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_results:
            writer.writerow(rec)
        # 平均值
        avg = {k: sum(r[k] for r in all_results) / len(all_results) for k in fieldnames if k != 'task'}
        avg['task'] = 'AVERAGE'
        writer.writerow(avg)

    avg_macro_f1 = sum(r['test_f1_macro'] for r in all_results) / len(all_results)
    print(f"\n===== Single-task baseline finished =====")
    print(f"Average test macro-F1 across all 9 tasks: {avg_macro_f1:.4f}")
    print(f"Detailed metrics saved to {csv_path}")


if __name__ == '__main__':
    main()