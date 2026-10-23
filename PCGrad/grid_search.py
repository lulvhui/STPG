import itertools
import csv
import torch
from my_main import run_training

ple_grid = {
    'num_shared_experts': [3, 6, 9],
    'num_task_experts': [1, 2],
    'expert_hidden': [768],
    'num_cgc_layers': [1, 2],
}

keys = list(ple_grid.keys())
combos = list(itertools.product(*ple_grid.values()))

print(f"Total PLE search combinations: {len(combos)}")

best_result = None
best_score = -1.0
all_results = []

for i, combo in enumerate(combos, 1):
    ple_params = dict(zip(keys, combo))
    print(f"\n{'='*60}")
    print(f"[{i}/{len(combos)}] PLE: {ple_params}")
    print(f"{'='*60}")

    try:
        result = run_training(ple_params=ple_params, use_ln=True, save_artifacts=True)
        all_results.append({
            **ple_params,
            'best_val_f1': result['best_val_f1'],
            'test_avg_macro_f1': result['test_avg_macro_f1'],
            'model_file': result['model_file'],
            'gate_csv': result['gate_csv'],
            'script_dir': result['script_dir'],
            'timestamp': result['timestamp'],
        })

        if result['best_val_f1'] > best_score:
            best_score = result['best_val_f1']
            best_result = result
            best_params = ple_params
    except Exception as e:
        print(f"Trial {i} failed: {e}")
        all_results.append({
            **ple_params,
            'best_val_f1': -1.0,
            'test_avg_macro_f1': -1.0,
            'model_file': None,
            'gate_csv': None,
            'script_dir': None,
            'timestamp': None,
        })
    finally:
        torch.cuda.empty_cache()

if best_result:
    print("\n" + "="*60)
    print("Best PLE hyper-parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best validation macro-F1: {best_result['best_val_f1']:.4f}")
    print(f"Corresponding test avg macro-F1: {best_result['test_avg_macro_f1']:.4f}")
    print(f"Model checkpoint: {best_result['model_file']}")
    print(f"Gate weights CSV: {best_result['gate_csv']}")
else:
    print("No successful trials.")

fieldnames = keys + ['best_val_f1', 'test_avg_macro_f1', 'model_file', 'gate_csv', 'script_dir', 'timestamp']
with open('ple_hyper_search_summary.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)
print(f"\nAll results saved to ple_hyper_search_summary.csv")