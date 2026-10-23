import itertools
import csv
from my_main import run_training

# 定义搜索空间
param_grid = {
    'proto_momentum': [0.1],
    'graph_temperature': [0.05, 0.1],
    'top_k': [3],
    'warmup_epochs': [5],
}

keys = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))

best_result = None
best_score = -1.0   # 我们以验证集最佳宏F1为优化目标
all_results = []    # 可保存所有结果用于分析

for combo in combos:
    tpg_params = dict(zip(keys, combo))
    print(f"\n=== Running with {tpg_params} ===")
    result = run_training(tpg_params)   # result 是字典
    all_results.append(result)

    if result['best_val_f1'] > best_score:
        best_score = result['best_val_f1']
        best_result = result
        best_params = tpg_params

# 输出最优组合
if best_result:
    print("\n" + "="*60)
    print("Best hyper-parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best validation macro-F1: {best_result['best_val_f1']:.4f}")
    print(f"Corresponding test avg macro-F1: {best_result['test_avg_macro_f1']:.4f}")
    print(f"Model checkpoint: {best_result['model_file']}")
    print(f"Task graph images (PNG/PDF): {best_result['task_graph_dir']}")
    print(f"Full logs and CSVs are in: {best_result['script_dir']}")
else:
    print("No successful trials.")

# （可选）保存所有结果到 CSV
with open('tpg_hyper_search_summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys + ['best_val_f1', 'test_avg_macro_f1', 'model_file', 'task_graph_dir'])
    writer.writeheader()
    for i, res in enumerate(all_results):
        row = {k: combos[i][j] for j, k in enumerate(keys)}
        row['best_val_f1'] = res['best_val_f1']
        row['test_avg_macro_f1'] = res['test_avg_macro_f1']
        row['model_file'] = res['model_file']
        row['task_graph_dir'] = res['task_graph_dir']
        writer.writerow(row)