import itertools
import csv
import json
import datetime
from my_main import run_training

# -------------------- 0. 固定消融实验配置 --------------------
ablation_configs = [
    ("Nothing",      {"use_projector": False, "use_graph": False, "use_gate": False, "use_ln": False}),
    ("+Projector",   {"use_projector": True,  "use_graph": False, "use_gate": False, "use_ln": False}),
    ("+Graph",       {"use_projector": True,  "use_graph": True,  "use_gate": False, "use_ln": False}),
    ("+Graph+Gate",  {"use_projector": True,  "use_graph": True,  "use_gate": True,  "use_ln": False}),
    ("+Graph+LN",    {"use_projector": True,  "use_graph": True,  "use_gate": False, "use_ln": True}),
    ("ALL",          {"use_projector": True,  "use_graph": True,  "use_gate": True,  "use_ln": True}),
]

# -------------------- 1. 超参搜索（不保存文件） --------------------
param_grid = {
    'proto_momentum': [0.3, 0.5, 0.7, 0.9],
    'graph_temperature': [0.02, 0.05, 0.1],
    'top_k': [2, 3, 4],
    'warmup_epochs': [0, 2, 5],
}
keys = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))
search_results = []

print("===== Hyper-parameter Search (w/o saving) =====")
for combo in combos:
    tpg_params = dict(zip(keys, combo))
    print(f"\n--- Search: {tpg_params} ---")
    res = run_training(
        tpg_params=tpg_params,
        use_tpg=True,
        use_projector=True, use_graph=True,
        use_gate=True, use_ln=True,  # 搜索基础 TPG 配置
        save_artifacts=False
    )
    search_results.append({
        'tpg_params': tpg_params,
        'best_val_f1': res['best_val_f1'],
        'test_avg_macro_f1': res['test_avg_macro_f1'],
        'test_metrics': res['test_metrics']   # 包含所有任务 p, r, f1
    })

# 选出 top-10
search_results.sort(key=lambda x: x['test_avg_macro_f1'], reverse=True)
top10 = search_results[:10]

# -------------------- 2. 重新运行 top-10 并保存文件 --------------------
print("\n===== Saving top-10 best experiments =====")
top10_saved = []
for rank, exp in enumerate(top10, 1):
    print(f"\n--- Top-10 #{rank}: {exp['tpg_params']} ---")
    res = run_training(
        tpg_params=exp['tpg_params'],
        use_tpg=True,
        use_projector=True, use_graph=True,
        use_gate=True, use_ln=True,
        save_artifacts=True
    )
    # 收集路径信息
    top10_saved.append({
        'rank': rank,
        'params': exp['tpg_params'],
        'test_avg_macro_f1': res['test_avg_macro_f1'],
        'test_metrics': res['test_metrics'],
        'model_file': res['model_file'],
        'task_graph_dir': res['task_graph_dir'],
        'gate_plot': f"{res['image_save_dir']}/task_gates_{res['timestamp']}.png",
        'script_dir': res['script_dir']
    })

# 将 top-10 详细指标写入 CSV
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
top10_csv = f'top10_metrics_{timestamp}.csv'
fieldnames = ['rank', 'task', 'acc', 'precision', 'recall', 'f1_weighted', 'f1_macro']
with open(top10_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for exp in top10_saved:
        for task, m in exp['test_metrics'].items():
            writer.writerow({
                'rank': exp['rank'],
                'task': task,
                'acc': m['acc'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1_weighted': m['f1_weighted'],
                'f1_macro': m['f1_macro']
            })
        # 写入平均值行
        avg = {k: sum(exp['test_metrics'][t][k] for t in exp['test_metrics']) / len(exp['test_metrics'])
               for k in ['acc', 'precision', 'recall', 'f1_weighted', 'f1_macro']}
        writer.writerow({
            'rank': exp['rank'],
            'task': 'AVERAGE',
            **avg
        })
print(f"Top-10 detailed metrics saved to {top10_csv}")

# -------------------- 3. 选择前3个最佳配置进行消融实验 --------------------
best3 = top10_saved[:3]
ablation_summary = []

print("\n===== Ablation Studies for top-3 configurations =====")
for exp in best3:
    print(f"\n>>> Ablation for rank {exp['rank']}: {exp['params']}")
    for name, flags in ablation_configs:
        print(f"  - {name}")
        res = run_training(
            tpg_params=exp['params'],
            use_tpg=True,
            use_projector=flags['use_projector'],
            use_graph=flags['use_graph'],
            use_gate=flags['use_gate'],
            use_ln=flags['use_ln'],
            save_artifacts=True
        )
        ablation_summary.append({
            'rank': exp['rank'],
            'ablation': name,
            'test_avg_macro_f1': res['test_avg_macro_f1'],
            'test_metrics': res['test_metrics'],
            'model_file': res['model_file'],
            'task_graph_dir': res['task_graph_dir'],
            'gate_plot': f"{res['image_save_dir']}/task_gates_{res['timestamp']}.png",
            'script_dir': res['script_dir']
        })

# 保存消融实验汇总 CSV
ablation_csv = f'ablation_metrics_{timestamp}.csv'
with open(ablation_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames + ['rank', 'ablation'])
    writer.writeheader()
    for rec in ablation_summary:
        for task, m in rec['test_metrics'].items():
            writer.writerow({
                'rank': rec['rank'],
                'ablation': rec['ablation'],
                'task': task,
                'acc': m['acc'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1_weighted': m['f1_weighted'],
                'f1_macro': m['f1_macro']
            })
        avg = {k: sum(rec['test_metrics'][t][k] for t in rec['test_metrics']) / len(rec['test_metrics'])
               for k in ['acc', 'precision', 'recall', 'f1_weighted', 'f1_macro']}
        writer.writerow({
            'rank': rec['rank'],
            'ablation': rec['ablation'],
            'task': 'AVERAGE',
            **avg
        })
print(f"Ablation metrics saved to {ablation_csv}")