import itertools
import csv
from my_main import run_training

# 定义 MoE 搜索空间
param_grid = {
    'num_experts': [6],
    'expert_hidden': [768],
    'top_k_router': [1, 2],
    'noise_std': [0.0],
    'load_balance_weight': [0.05],
}

keys = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))

best_result = None
best_score = -1.0
all_results = []

print(f"Total MoE search combinations: {len(combos)}")

for i, combo in enumerate(combos, 1):
    moe_params = dict(zip(keys, combo))
    moe_params['use_load_balancing'] = True
    print(f"\n=== [{i}/{len(combos)}] Running with {moe_params} ===")
    result = run_training(moe_params=moe_params, use_ln=True, save_artifacts=True)
    all_results.append(result)

    if result['best_val_f1'] > best_score:
        best_score = result['best_val_f1']
        best_result = result
        best_params = moe_params

# 输出最优组合
if best_result:
    print("\n" + "="*60)
    print("Best MoE hyper-parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best validation macro-F1: {best_result['best_val_f1']:.4f}")
    print(f"Corresponding test avg macro-F1: {best_result['test_avg_macro_f1']:.4f}")
    print(f"Model checkpoint: {best_result['model_file']}")
    print(f"Router weights CSV: {best_result['router_csv']}")
    print(f"Full logs and CSVs are in: {best_result['script_dir']}")
else:
    print("No successful trials.")

# 保存所有结果到 CSV
with open('moe_hyper_search_summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys + ['use_load_balancing', 'best_val_f1', 'test_avg_macro_f1', 'model_file', 'router_csv'])
    writer.writeheader()
    for i, res in enumerate(all_results):
        row = {k: combos[i][j] for j, k in enumerate(keys)}
        row['use_load_balancing'] = True
        row['best_val_f1'] = res['best_val_f1']
        row['test_avg_macro_f1'] = res['test_avg_macro_f1']
        row['model_file'] = res['model_file']
        row['router_csv'] = res['router_csv']
        writer.writerow(row)