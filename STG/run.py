from my_main import run_training
from my_dataload import CFG

CFG['device'] = 'cuda:1'

result = run_training(
    task_hidden_dim=768,
    use_ln=True,
    save_artifacts=True,
    gradient_method='pcgrad',
)

test_metrics = result['test_metrics']
n = len(test_metrics)
avg_acc = sum(m['acc'] for m in test_metrics.values()) / n
avg_f1w = sum(m['f1_weighted'] for m in test_metrics.values()) / n
avg_f1m = sum(m['f1_macro'] for m in test_metrics.values()) / n

print(f"\n{'='*50}")
print(f"Best val macro-F1:  {result['best_val_f1']:.4f}")
print(f"Test avg acc:       {avg_acc:.4f}")
print(f"Test avg f1_w:      {avg_f1w:.4f}")
print(f"Test avg f1_m:      {avg_f1m:.4f}")
print(f"Model checkpoint:   {result['model_file']}")
print(f"Gradient stats CSV: {result.get('gradient_csv')}")
print(f"Log directory:      {result['script_dir']}")