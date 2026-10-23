import itertools
import csv
import torch
from my_main import run_training

mmoe_grid = {
    'num_experts': [3,6,9],
    'expert_hidden': [384, 768],
}

keys = list(mmoe_grid.keys())
combos = list(itertools.product(*mmoe_grid.values()))

print(f"Total MMoE search combinations: {len(combos)}")

best_result = None
best_score = -1.0
all_results = []

for i, combo in enumerate(combos, 1):
    mmoe_params = dict(zip(keys, combo))
    print(f"\n{'='*60}")
    print(f"[{i}/{len(combos)}] MMoE: {mmoe_params}")
    print(f"{'='*60}")

    try:
        result = run_training(moe_params=mmoe_params, use_ln=True, save_artifacts=True)
        tm = result['test_metrics']
        n = len(tm)
        test_avg_acc = sum(m['acc'] for m in tm.values()) / n
        test_avg_f1_macro = sum(m['f1_macro'] for m in tm.values()) / n
        test_avg_f1_weighted = sum(m['f1_weighted'] for m in tm.values()) / n
        all_results.append({
            **mmoe_params,
            'best_val_f1': result['best_val_f1'],
            'test_avg_acc': test_avg_acc,
            'test_avg_f1_macro': test_avg_f1_macro,
            'test_avg_f1_weighted': test_avg_f1_weighted,
            'model_file': result['model_file'],
            'gate_csv': result['gate_csv'],
            'script_dir': result['script_dir'],
            'timestamp': result['timestamp'],
        })

        if result['best_val_f1'] > best_score:
            best_score = result['best_val_f1']
            best_result = result
            best_params = mmoe_params
    except Exception as e:
        print(f"Trial {i} failed: {e}")
        all_results.append({
            **mmoe_params,
            'best_val_f1': -1.0,
            'test_avg_acc': -1.0,
            'test_avg_f1_macro': -1.0,
            'test_avg_f1_weighted': -1.0,
            'model_file': None,
            'gate_csv': None,
            'script_dir': None,
            'timestamp': None,
        })
    finally:
        torch.cuda.empty_cache()

if best_result:
    tm = best_result['test_metrics']
    n = len(tm)
    b_acc = sum(m['acc'] for m in tm.values()) / n
    b_f1m = sum(m['f1_macro'] for m in tm.values()) / n
    b_f1w = sum(m['f1_weighted'] for m in tm.values()) / n
    print("\n" + "="*60)
    print("Best MMoE hyper-parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best validation avg macro-F1: {best_result['best_val_f1']:.4f}")
    print(f"Best model test avg across tasks:")
    print(f"  avg_acc={b_acc:.4f}  avg_f1_weighted={b_f1w:.4f}  avg_f1_macro={b_f1m:.4f}")
    print(f"Model checkpoint: {best_result['model_file']}")
    print(f"Gate weights CSV: {best_result['gate_csv']}")
else:
    print("No successful trials.")

fieldnames = keys + ['best_val_f1', 'test_avg_acc', 'test_avg_f1_macro', 'test_avg_f1_weighted', 'model_file', 'gate_csv', 'script_dir', 'timestamp']
with open('mmoe_hyper_search_summary.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)
print(f"\nAll results saved to mmoe_hyper_search_summary.csv")