"""
TPG Scaling Experiment (Random Task Selection)
验证 TPG 在不同任务数量下是否增益多任务训练效果

实验设计:
  - 任务数量: 4, 5, 6, 7, 8 (从9个任务中随机选取)
  - 每个任务数量跑2轮，每轮随机选不同的任务子集
  - 每组对比 有完整TPG vs 无TPG

对比条件:
  - With_TPG:    use_tpg=True,  use_projector=True, use_graph=True, use_gate=True, use_ln=True
  - No_TPG:      use_tpg=False, use_projector=False, use_graph=False, use_gate=False, use_ln=False
"""

import csv
import datetime
import copy
import random
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import my_dataload
import my_model
import my_main

my_dataload.CFG['device'] = 'cuda:0'
torch.cuda.set_device(torch.device('cuda:0'))

# ===================== 配置 =====================
ALL_TASKS = list(my_dataload.TASK_CONFIG.keys())
NUM_ROUNDS = 2
TASK_COUNTS = [4, 5, 6, 7, 8]
RANDOM_SEED = 42

TPG_PARAMS = {
    'proto_momentum': 0.5,
    'graph_temperature': 0.1,
    'top_k': 3,
    'warmup_epochs': 10,
}

# ===================== 生成随机任务子集 =====================
random.seed(RANDOM_SEED)
task_subsets = []
for n in TASK_COUNTS:
    for rnd in range(1, NUM_ROUNDS + 1):
        subset = random.sample(ALL_TASKS, n)
        subset.sort(key=lambda t: ALL_TASKS.index(t))
        task_subsets.append((n, rnd, subset))

print("=" * 70)
print("  TPG Scaling Experiment - Random Task Selection Plan")
print("=" * 70)
for n, rnd, subset in task_subsets:
    print(f"  {n} tasks | Round {rnd} | {subset}")
print("=" * 70)

# ===================== 保存原始全局变量 =====================
ORIGINAL_TASK_CONFIG = copy.deepcopy(my_dataload.TASK_CONFIG)
ORIGINAL_TASK_LABEL_MAPS = copy.deepcopy(my_dataload.task_label_maps)


def patch_globals(task_subset):
    new_config = {k: ORIGINAL_TASK_CONFIG[k] for k in task_subset if k in ORIGINAL_TASK_CONFIG}
    new_maps = {k: ORIGINAL_TASK_LABEL_MAPS[k] for k in task_subset if k in ORIGINAL_TASK_LABEL_MAPS}
    my_dataload.TASK_CONFIG = new_config
    my_dataload.task_label_maps = new_maps
    my_model.TASK_CONFIG = new_config
    if hasattr(my_main, 'TASK_CONFIG'):
        my_main.TASK_CONFIG = new_config
    if hasattr(my_main, 'task_label_maps'):
        my_main.task_label_maps = new_maps


def restore_globals():
    my_dataload.TASK_CONFIG = ORIGINAL_TASK_CONFIG
    my_dataload.task_label_maps = ORIGINAL_TASK_LABEL_MAPS
    my_model.TASK_CONFIG = ORIGINAL_TASK_CONFIG
    if hasattr(my_main, 'TASK_CONFIG'):
        my_main.TASK_CONFIG = ORIGINAL_TASK_CONFIG
    if hasattr(my_main, 'task_label_maps'):
        my_main.task_label_maps = ORIGINAL_TASK_LABEL_MAPS


# ===================== 主实验循环 =====================
results = []
summary = []

for num_tasks, rnd, task_subset in task_subsets:
    label = f"{num_tasks}tasks_R{rnd}"
    print(f"\n{'='*70}")
    print(f"  >>> {label}: {task_subset}")
    print(f"{'='*70}")

    patch_globals(task_subset)

    # ---- Condition A: Without TPG ----
    print(f"\n  [{label}] Condition: No_TPG (base multi-task)")
    res_no_tpg = None
    no_tpg_ok = False
    try:
        res_no_tpg = my_main.run_training(
            tpg_params=TPG_PARAMS,
            use_tpg=False,
            use_projector=False,
            use_graph=False,
            use_gate=False,
            use_ln=False,
            save_artifacts=True
        )
        no_tpg_ok = True
    except Exception as e:
        print(f"  ERROR (No_TPG): {e}")
        traceback.print_exc()

    # ---- Condition B: With TPG (full) ----
    print(f"\n  [{label}] Condition: With_TPG (full TPG)")
    res_with_tpg = None
    with_tpg_ok = False
    try:
        res_with_tpg = my_main.run_training(
            tpg_params=TPG_PARAMS,
            use_tpg=True,
            use_projector=True,
            use_graph=True,
            use_gate=True,
            use_ln=True,
            save_artifacts=True
        )
        with_tpg_ok = True
    except Exception as e:
        print(f"  ERROR (With_TPG): {e}")
        traceback.print_exc()

    restore_globals()

    # ---- 打印每组条件的逐任务结果 ----
    for condition, res, ok in [('No_TPG', res_no_tpg, no_tpg_ok),
                                ('With_TPG', res_with_tpg, with_tpg_ok)]:
        if not ok or res is None:
            print(f"\n  [{label}] {condition}: FAILED")
            continue
        print(f"\n  [{label}] {condition} - Per-task results:")
        print(f"  {'Task':<14} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1_w':>8} {'F1_m':>8}")
        print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for task in task_subset:
            m = res['test_metrics'][task]
            print(f"  {task:<14} {m['acc']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1_weighted']:>8.4f} {m['f1_macro']:>8.4f}")
        avg = {k: sum(res['test_metrics'][t][k] for t in task_subset) / len(task_subset)
               for k in ['acc', 'precision', 'recall', 'f1_weighted', 'f1_macro']}
        print(f"  {'AVERAGE':<14} {avg['acc']:>8.4f} {avg['precision']:>8.4f} {avg['recall']:>8.4f} {avg['f1_weighted']:>8.4f} {avg['f1_macro']:>8.4f}")

    # ---- 记录每个任务的详细结果 ----
    for condition, res, ok in [('No_TPG', res_no_tpg, no_tpg_ok),
                                ('With_TPG', res_with_tpg, with_tpg_ok)]:
        if not ok or res is None:
            for task in task_subset:
                results.append({
                    'label': label, 'num_tasks': num_tasks, 'round': rnd,
                    'task_subset': ', '.join(task_subset),
                    'condition': condition, 'task': task,
                    'acc': '', 'precision': '', 'recall': '',
                    'f1_weighted': '', 'f1_macro': ''
                })
            results.append({
                'label': label, 'num_tasks': num_tasks, 'round': rnd,
                'task_subset': ', '.join(task_subset),
                'condition': condition, 'task': 'AVERAGE',
                'acc': '', 'precision': '', 'recall': '',
                'f1_weighted': '', 'f1_macro': ''
            })
            continue

        for task in task_subset:
            m = res['test_metrics'][task]
            results.append({
                'label': label, 'num_tasks': num_tasks, 'round': rnd,
                'task_subset': ', '.join(task_subset),
                'condition': condition, 'task': task,
                'acc': f"{m['acc']:.4f}", 'precision': f"{m['precision']:.4f}",
                'recall': f"{m['recall']:.4f}", 'f1_weighted': f"{m['f1_weighted']:.4f}",
                'f1_macro': f"{m['f1_macro']:.4f}",
            })

        avg = {k: sum(res['test_metrics'][t][k] for t in task_subset) / len(task_subset)
               for k in ['acc', 'precision', 'recall', 'f1_weighted', 'f1_macro']}
        results.append({
            'label': label, 'num_tasks': num_tasks, 'round': rnd,
            'task_subset': ', '.join(task_subset),
            'condition': condition, 'task': 'AVERAGE',
            'acc': f"{avg['acc']:.4f}", 'precision': f"{avg['precision']:.4f}",
            'recall': f"{avg['recall']:.4f}", 'f1_weighted': f"{avg['f1_weighted']:.4f}",
            'f1_macro': f"{avg['f1_macro']:.4f}",
        })

    # ---- 汇总 ----
    no_f1 = None
    with_f1 = None
    if no_tpg_ok and res_no_tpg is not None:
        no_f1 = sum(res_no_tpg['test_metrics'][t]['f1_macro'] for t in task_subset) / len(task_subset)
    if with_tpg_ok and res_with_tpg is not None:
        with_f1 = sum(res_with_tpg['test_metrics'][t]['f1_macro'] for t in task_subset) / len(task_subset)

    delta = (with_f1 - no_f1) if (no_f1 is not None and with_f1 is not None) else None
    summary.append({
        'label': label, 'num_tasks': num_tasks, 'round': rnd,
        'tasks': ', '.join(task_subset),
        'no_tpg_avg_f1_macro': no_f1,
        'with_tpg_avg_f1_macro': with_f1,
        'delta_f1_macro': delta
    })

    if no_f1 is not None and with_f1 is not None:
        print(f"\n  >>> [{label}] No_TPG={no_f1:.4f} | With_TPG={with_f1:.4f} | Delta={delta:+.4f}")
    else:
        print(f"\n  >>> [{label}] One or both conditions failed.")

# ===================== 保存详细结果 CSV =====================
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = f'tpg_random_comparison_{timestamp}.csv'
fieldnames = ['label', 'num_tasks', 'round', 'task_subset', 'condition', 'task',
              'acc', 'precision', 'recall', 'f1_weighted', 'f1_macro']

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)
print(f"\nDetailed results saved to {csv_path}")

# ===================== 保存汇总 CSV =====================
summary_csv_path = f'tpg_random_summary_{timestamp}.csv'
summary_fields = ['label', 'num_tasks', 'round', 'tasks',
                  'no_tpg_avg_f1_macro', 'with_tpg_avg_f1_macro', 'delta_f1_macro']
with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    for s in summary:
        row = {
            'label': s['label'],
            'num_tasks': s['num_tasks'],
            'round': s['round'],
            'tasks': s['tasks'],
            'no_tpg_avg_f1_macro': f"{s['no_tpg_avg_f1_macro']:.4f}" if s['no_tpg_avg_f1_macro'] is not None else '',
            'with_tpg_avg_f1_macro': f"{s['with_tpg_avg_f1_macro']:.4f}" if s['with_tpg_avg_f1_macro'] is not None else '',
            'delta_f1_macro': f"{s['delta_f1_macro']:+.4f}" if s['delta_f1_macro'] is not None else '',
        }
        writer.writerow(row)
print(f"Summary saved to {summary_csv_path}")

# ===================== 打印汇总对比表 =====================
print(f"\n{'='*80}")
print(f"  TPG Scaling Experiment (Random) - Summary")
print(f"  (Does TPG provide increasing gains as task count grows?)")
print(f"{'='*80}")
print(f"{'Label':<16} | {'No_TPG':>10} | {'With_TPG':>10} | {'Delta':>10}")
print(f"{'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
for s in summary:
    no_str = f"{s['no_tpg_avg_f1_macro']:.4f}" if s['no_tpg_avg_f1_macro'] is not None else "N/A"
    with_str = f"{s['with_tpg_avg_f1_macro']:.4f}" if s['with_tpg_avg_f1_macro'] is not None else "N/A"
    delta_str = f"{s['delta_f1_macro']:+.4f}" if s['delta_f1_macro'] is not None else "N/A"
    print(f"{s['label']:<16} | {no_str:>10} | {with_str:>10} | {delta_str:>10}")

# ===================== 按任务数聚合（两轮平均） =====================
print(f"\n{'='*80}")
print(f"  Aggregated by Task Count (avg over {NUM_ROUNDS} rounds)")
print(f"{'='*80}")
print(f"{'#Tasks':>6} | {'No_TPG':>10} | {'With_TPG':>10} | {'Delta':>10}")
print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
for n in TASK_COUNTS:
    group = [s for s in summary if s['num_tasks'] == n]
    no_vals = [s['no_tpg_avg_f1_macro'] for s in group if s['no_tpg_avg_f1_macro'] is not None]
    with_vals = [s['with_tpg_avg_f1_macro'] for s in group if s['with_tpg_avg_f1_macro'] is not None]
    no_avg = sum(no_vals) / len(no_vals) if no_vals else None
    with_avg = sum(with_vals) / len(with_vals) if with_vals else None
    delta_avg = (with_avg - no_avg) if (no_avg is not None and with_avg is not None) else None
    no_str = f"{no_avg:.4f}" if no_avg is not None else "N/A"
    with_str = f"{with_avg:.4f}" if with_avg is not None else "N/A"
    delta_str = f"{delta_avg:+.4f}" if delta_avg is not None else "N/A"
    print(f"{n:>6} | {no_str:>10} | {with_str:>10} | {delta_str:>10}")

# ===================== 打印逐任务详细对比表 =====================
print(f"\n{'='*80}")
print(f"  Per-task Detailed Comparison")
print(f"{'='*80}")
for num_tasks, rnd, task_subset in task_subsets:
    label = f"{num_tasks}tasks_R{rnd}"
    no_rows = [r for r in results if r['label'] == label and r['condition'] == 'No_TPG']
    with_rows = [r for r in results if r['label'] == label and r['condition'] == 'With_TPG']
    print(f"\n  [{label}] {', '.join(task_subset)}")
    print(f"  {'Task':<14} {'No_TPG_F1m':>11} {'With_TPG_F1m':>12} {'Delta':>8}")
    print(f"  {'-'*14} {'-'*11} {'-'*12} {'-'*8}")
    for no_r, with_r in zip(no_rows, with_rows):
        task = no_r['task']
        no_val = no_r['f1_macro']
        with_val = with_r['f1_macro']
        if no_val != '' and with_val != '':
            delta_v = float(with_val) - float(no_val)
            print(f"  {task:<14} {no_val:>11} {with_val:>12} {delta_v:>+8.4f}")
        else:
            print(f"  {task:<14} {no_val:>11} {with_val:>12} {'N/A':>8}")