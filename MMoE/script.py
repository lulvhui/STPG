import itertools
import csv
import json
import datetime
from my_main import run_training

# -------------------- 0. MoE 消融实验配置 --------------------
ablation_configs = [
    ("Baseline(1 Expert)", {"num_experts": 1,  "use_load_balancing": False, "use_ln": False}),
    ("+Experts(4)",       {"num_experts": 4,  "use_load_balancing": False, "use_ln": False}),
    ("+LoadBalance",      {"num_experts": 4,  "use_load_balancing": True,  "use_ln": False}),
    ("+LN",               {"num_experts": 4,  "use_load_balancing": True,  "use_ln": True}),
    ("+TopKRouter",       {"num_experts": 8,  "use_load_balancing": True,  "use_ln": True}),
    ("ALL",               {"num_experts": 8,  "use_load_balancing": True,  "use_ln": True}),
]

# -------------------- 1. MoE 超参搜索（不保存文件） --------------------
param_grid = {
    'num_experts': [2, 4, 6, 8],
    'expert_hidden': [384, 768],
    'top_k_router': [1, 2],
    'noise_std': [0.0, 0.1, 0.2],
    'load_balance_weight': [0.01, 0.05, 0.1],
}
keys = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))
search_results = []

print(f"===== MoE Hyper-parameter Search: {len(combos)} combinations =====")
for i, combo in enumerate(combos, 1):
    moe_params = dict(zip(keys, combo))
    moe_params['use_load_balancing'] = True
    print(f"\n--- [{i}/{len(combos)}] Search: {moe_params} ---")
    res = run_training(
        moe_params=moe_params,
        use_ln=True,
        save_artifacts=False
    )
    search_results.append({
        'moe_params': moe_params,
        'best_val_f1': res['best_val_f1'],
        'test_avg_macro_f1': res['test_avg_macro_f1'],
        'test_metrics': res['test_metrics']
    })

# 选出 top-10
search_results.sort(key=lambda x: x['test_avg_macro_f1'], reverse=True)
top10 = search_results[:10]

# -------------------- 2. 重新运行 top-10 并保存文件 --------------------
print("\n===== Saving top-10 best MoE experiments =====")
top10_saved = []
for rank, exp in enumerate(top10, 1):
    print(f"\n--- Top-10 #{rank}: {exp['moe_params']} ---")
    res = run_training(
        moe_params=exp['moe_params'],
        use_ln=True,
        save_artifacts=True
    )
    top10_saved.append({
        'rank': rank,
        'params': exp['moe_params'],
        'test_avg_macro_f1': res['test_avg_macro_f1'],
        'test_metrics': res['test_metrics'],
        'model_file': res['model_file'],
        'router_csv': res['router_csv'],
        'script_dir': res['script_dir']
    })

# 将 top-10 详细指标写入 CSV
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
top10_csv = f'top10_moe_metrics_{timestamp}.csv'
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

print("\n===== MoE Ablation Studies for top-3 configurations =====")
for exp in best3:
    print(f"\n>>> Ablation for rank {exp['rank']}: {exp['params']}")
    for name, flags in ablation_configs:
        print(f"  - {name}")
        ablation_moe = exp['params'].copy()
        ablation_moe['num_experts'] = flags['num_experts']
        ablation_moe['use_load_balancing'] = flags['use_load_balancing']
        res = run_training(
            moe_params=ablation_moe,
            use_ln=flags['use_ln'],
            save_artifacts=True
        )
        ablation_summary.append({
            'rank': exp['rank'],
            'ablation': name,
            'test_avg_macro_f1': res['test_avg_macro_f1'],
            'test_metrics': res['test_metrics'],
            'model_file': res['model_file'],
            'router_csv': res['router_csv'],
            'script_dir': res['script_dir']
        })

# 保存消融实验汇总 CSV
ablation_csv = f'ablation_moe_metrics_{timestamp}.csv'
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