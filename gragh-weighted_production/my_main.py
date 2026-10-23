import os
import json
import torch
import datetime
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from my_dataload import *
from my_model import MultiTaskModel

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
            loss, logits = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats,
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

def plot_task_gates(gate_values, task_names, save_dir, timestamp):
    """绘制任务门控权重柱状图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(task_names, gate_values, color='steelblue')
    ax.set_ylabel('Gate value')
    ax.set_title('Task-specific propagation gates')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(save_dir, f'task_gates_{timestamp}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def save_task_graph_image(adj_matrix, task_names, epoch, img_dir):
    G = nx.DiGraph()
    T = len(task_names)
    for i, name in enumerate(task_names):
        G.add_node(i, label=name)
    edges, weights = [], []
    for i in range(T):
        for j in range(T):
            w = adj_matrix[i, j]
            if w > 0 and i != j:
                edges.append((i, j))
                weights.append(w)
    if not edges:
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
        labels = {i: task_names[i] for i in range(T)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.title(f'Task Prototype Graph - Epoch {epoch}')
        plt.axis('off')
        for fmt in ['png', 'pdf']:
            plt.savefig(os.path.join(img_dir, f'task_graph_epoch_{epoch:03d}.{fmt}'), dpi=150, bbox_inches='tight')
        plt.close()
        return
    for (src, dst), w in zip(edges, weights):
        G.add_edge(src, dst, weight=w)
    pos = nx.spring_layout(G, seed=43, k=2.0, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    labels = {i: task_names[i] for i in range(T)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    cmap = plt.cm.viridis
    norm_weights = np.array(weights) / max(weights) if max(weights) > 0 else np.array(weights)
    edge_colors = [cmap(w) for w in norm_weights]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                           arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1',
                           width=2.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Propagation Weight', fontsize=9)
    plt.title(f'Epoch {epoch}')
    plt.axis('off')
    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(img_dir, f'task_graph_epoch_{epoch:03d}.{fmt}'), dpi=150, bbox_inches='tight')
    plt.close()

def record_task_graph(model, epoch, task_names, task_graph_csv, img_dir):
    prototypes = model.get_prototypes()
    proto_list = [prototypes[task] for task in task_names]
    proto_matrix = torch.stack(proto_list, dim=0)
    norm = torch.norm(proto_matrix, dim=1, keepdim=True) + 1e-8
    proto_normed = proto_matrix / norm
    sim = torch.mm(proto_normed, proto_normed.t())
    T = len(task_names)
    mask = 1 - torch.eye(T, device=sim.device)
    sim_masked = sim * mask + (-1e9) * (1 - mask)
    topk_vals, topk_idx = torch.topk(sim_masked, k=min(model.top_k, T-1), dim=1)
    sparse_sim = torch.full_like(sim, -1e9)
    sparse_sim.scatter_(1, topk_idx, topk_vals)
    adj = F.softmax(sparse_sim / model.graph_temperature, dim=1).cpu().numpy()
    if epoch == 1 and task_graph_csv is not None:
        columns = ['epoch'] + [f'{src}_to_{tgt}' for src in task_names for tgt in task_names]
        with open(task_graph_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
    if task_graph_csv is not None:
        row = [epoch] + adj.flatten().tolist()
        with open(task_graph_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    if img_dir is not None:
        save_task_graph_image(adj, task_names, epoch, img_dir)

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
    image_save_dir = f'{base_dir}/image/{timestamp}'
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)
    paths = {
        'script_dir': script_dir,
        'model_save_dir': model_save_dir,
        'image_save_dir': image_save_dir,
        'log_file': os.path.join(script_dir, f'train_log_tpg_{timestamp}.txt'),
        'csv_file': os.path.join(script_dir, f'train_metrics_tpg_{timestamp}.csv'),
        'task_graph_csv': os.path.join(script_dir, f'task_graph_{timestamp}.csv'),
        'graph_img_dir': os.path.join(image_save_dir, f'graph_imgs_{timestamp}'),
    }
    os.makedirs(paths['graph_img_dir'], exist_ok=True)
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
    task_graph_csv = paths['task_graph_csv'] if save_artifacts else None
    graph_img_dir = paths['graph_img_dir'] if save_artifacts else None

    log = lambda msg: log_message(msg, log_file, to_console=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    best_avg_f1_macro = 0.0
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
            total_loss, _ = model(c_ids, c_masks, t_ids, s_ids, s_masks, s_tids, s_feats, labels_dict)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss_sum += total_loss.item()
        train_loss_avg = train_loss_sum / len(train_loader)

        val_loss, val_metrics = evaluate(model, valid_loader, device, task_names, desc="Valid eval")
        test_loss, test_metrics = evaluate(model, test_loader, device, task_names, desc="Test eval")

        log(f"Epoch {epoch}: train_loss={train_loss_avg:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
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

        if save_artifacts and model.use_graph:
            record_task_graph(model, epoch, task_names, task_graph_csv, graph_img_dir)

        avg_val_f1_macro = sum(m['f1_macro'] for m in val_metrics.values()) / len(val_metrics)
        if avg_val_f1_macro > best_avg_f1_macro + 1e-4:
            best_avg_f1_macro = avg_val_f1_macro
            no_improve = 0
            if save_artifacts:
                ckpt_name = f'checkpoint_tpg_{paths["timestamp"]}.pt'
                ckpt_path = os.path.join(paths['model_save_dir'], ckpt_name)
                torch.save(model.state_dict(), ckpt_path)
                log(f"Best model saved to {ckpt_path} (avg_val_f1_macro={avg_val_f1_macro:.4f})")
                test_res_name = f'test_predictions_tpg_{paths["timestamp"]}.json'
                test_res_path = os.path.join(paths['model_save_dir'], test_res_name)
                with open(test_res_path, 'w', encoding='utf-8') as f:
                    json.dump(test_metrics, f, indent=2)
        else:
            no_improve += 1
            log(f"No improvement for {no_improve} epoch(s).")
            if no_improve >= patience:
                early_stop = True

    # 训练结束后绘制 gate 柱状图
    if save_artifacts and model.use_gate and model.task_gate is not None:
        gate_vals = model.get_task_gates()
        if gate_vals is not None:
            plot_task_gates(gate_vals, task_names, paths['image_save_dir'], paths['timestamp'])

    test_avg_macro_f1 = sum(m['f1_macro'] for m in test_metrics.values()) / len(test_metrics)
    return best_avg_f1_macro, test_metrics, test_avg_macro_f1

# ======================= 主入口函数 =======================
def run_training(tpg_params=None, use_tpg=True,
                 use_projector=True, use_graph=True,
                 use_gate=True, use_ln=True,
                 save_artifacts=True):
    from my_dataload import TPG as default_tpg
    if tpg_params is None:
        tpg_params = default_tpg.copy()
    else:
        tpg_params = {**default_tpg, **tpg_params}

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
        proto_momentum=tpg_params['proto_momentum'],
        graph_temperature=tpg_params['graph_temperature'],
        top_k=tpg_params['top_k'],
        warmup_epochs=tpg_params['warmup_epochs'],
        use_tpg=use_tpg,
        use_projector=use_projector,
        use_graph=use_graph,
        use_gate=use_gate,
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
        'model_file': os.path.join(paths['model_save_dir'], f'checkpoint_tpg_{timestamp}.pt') if save_artifacts else None,
        'task_graph_dir': paths['graph_img_dir'] if save_artifacts else None,
        'script_dir': paths['script_dir'] if save_artifacts else None,
        'image_save_dir': paths['image_save_dir'] if save_artifacts else None,
        'timestamp': timestamp
    }

if __name__ == '__main__':
    result = run_training(save_artifacts=True)
    print(f"Best val macro-F1: {result['best_val_f1']:.4f}")