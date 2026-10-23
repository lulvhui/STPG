#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASEN (Approximate Subgroup Equivariance) 多模态评论分类训练脚本

运行方式:
    python run.py [--device cuda:0] [--lr 1e-6] [--epochs 100] [--method pcgrad]

输出:
    - 每个任务的 Accuracy, Macro-F1, Weighted-F1
    - 所有任务平均值
    - 训练日志和模型保存
"""

import os
import sys
import json
import argparse
from datetime import datetime
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(description='ASEN Multi-Modal Comment Classification')
    
    # 基础配置
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='设备 (cuda:0, cuda:1, cpu)')
    parser.add_argument('--seed', type=int, default=43,
                        help='随机种子')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--train_bs', type=int, default=16,
                        help='训练批次大小')
    parser.add_argument('--valid_bs', type=int, default=8,
                        help='验证/测试批次大小')
    
    # 学习率配置
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='基础学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    
    # ASEN 配置
    parser.add_argument('--task_hidden_dim', type=int, default=768,
                        help='任务隐藏层维度')
    parser.add_argument('--use_ln', action='store_true', default=True,
                        help='使用 LayerNorm')
    parser.add_argument('--asen_num_layers', type=int, default=2,
                        help='ASEN 等变编码器层数')
    
    # 梯度方法
    parser.add_argument('--gradient_method', type=str, default='pcgrad',
                        choices=['pcgrad', 'acgrad', 'none'],
                        help='梯度冲突解决方法')
    
    # 保存配置
    parser.add_argument('--save_artifacts', action='store_true', default=True,
                        help='保存训练产物')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='自定义输出目录')
    
    return parser.parse_args()


def print_config(args):
    """打印配置信息"""
    print("\n" + "=" * 70)
    print("ASEN Training Configuration")
    print("=" * 70)
    print(f"{'Device:':<25} {args.device}")
    print(f"{'Random Seed:':<25} {args.seed}")
    print(f"{'Epochs:':<25} {args.epochs}")
    print(f"{'Early Stop Patience:':<25} {args.patience}")
    print(f"{'Train Batch Size:':<25} {args.train_bs}")
    print(f"{'Valid Batch Size:':<25} {args.valid_bs}")
    print("-" * 70)
    print(f"{'Learning Rate:':<25} {args.lr}")
    print(f"{'Weight Decay:':<25} {args.weight_decay}")
    print(f"{'Task Hidden Dim:':<25} {args.task_hidden_dim}")
    print(f"{'ASEN Layers:':<25} {args.asen_num_layers}")
    print(f"{'Use LayerNorm:':<25} {args.use_ln}")
    print(f"{'Gradient Method:':<25} {args.gradient_method}")
    print("=" * 70 + "\n")


def print_task_metrics_table(test_metrics, task_names):
    """格式化输出各任务指标表格"""
    headers = ['Task', 'Accuracy', 'Macro-F1', 'Weighted-F1', 'Precision', 'Recall']
    rows = []
    
    for task in task_names:
        m = test_metrics[task]
        rows.append([
            task,
            f"{m['acc']:.4f}",
            f"{m['f1_macro']:.4f}",
            f"{m['f1_weighted']:.4f}",
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}"
        ])
    
    # 添加平均行
    n = len(task_names)
    avg_acc = sum(m['acc'] for m in test_metrics.values()) / n
    avg_f1m = sum(m['f1_macro'] for m in test_metrics.values()) / n
    avg_f1w = sum(m['f1_weighted'] for m in test_metrics.values()) / n
    avg_prec = sum(m.get('precision', 0) for m in test_metrics.values()) / n
    avg_rec = sum(m.get('recall', 0) for m in test_metrics.values()) / n
    
    rows.append(['---', '---', '---', '---', '---', '---'])
    rows.append([
        'AVERAGE',
        f"{avg_acc:.4f}",
        f"{avg_f1m:.4f}",
        f"{avg_f1w:.4f}",
        f"{avg_prec:.4f}",
        f"{avg_rec:.4f}"
    ])
    
    table = tabulate(rows, headers=headers, tablefmt='grid', floatfmt='.4f')
    return table, avg_acc, avg_f1m, avg_f1w


def save_results(result, args):
    """保存结果到 JSON 文件"""
    timestamp = result.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # 构建结果字典
    results_dict = {
        'timestamp': timestamp,
        'config': vars(args),
        'best_val_macro_f1': result['best_val_f1'],
        'test_avg_macro_f1': result['test_avg_macro_f1'],
        'test_metrics_per_task': {
            task: metrics for task, metrics in result['test_metrics'].items()
        },
        'averages': {
            'accuracy': sum(m['acc'] for m in result['test_metrics'].values()) / len(result['test_metrics']),
            'macro_f1': sum(m['f1_macro'] for m in result['test_metrics'].values()) / len(result['test_metrics']),
            'weighted_f1': sum(m['f1_weighted'] for m in result['test_metrics'].values()) / len(result['test_metrics'])
        },
        'model_file': result.get('model_file'),
        'script_dir': result.get('script_dir'),
    }
    
    # 保存路径
    if args.output_dir:
        output_path = os.path.join(args.output_dir, f'aseen_results_{timestamp}.json')
    else:
        output_path = os.path.join(
            result.get('script_dir', '.'), 
            f'train_results_{timestamp}.json'
        )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    return output_path


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 打印配置
    print_config(args)
    
    # 动态修改配置
    from my_dataload import CFG
    
    CFG['device'] = args.device
    CFG['seed'] = args.seed
    CFG['epochs'] = args.epochs
    CFG['patience'] = args.patience
    CFG['train_bs'] = args.train_bs
    CFG['valid_bs'] = args.valid_bs
    CFG['lr'] = args.lr
    CFG['weight_decay'] = args.weight_decay
    
    # 导入训练函数
    from my_main import run_training, TASK_CONFIG
    
    print("🚀 Starting ASEN Training...")
    print("=" * 70)
    
    # 开始训练
    try:
        result = run_training(
            task_hidden_dim=args.task_hidden_dim,
            use_ln=args.use_ln,
            save_artifacts=args.save_artifacts,
            gradient_method=args.gradient_method if args.gradient_method != 'none' else None,
        )
        
        # 获取测试指标
        test_metrics = result['test_metrics']
        task_names = list(TASK_CONFIG.keys())
        
        # 打印详细结果
        print("\n" + "=" * 70)
        print("🎯 TRAINING COMPLETED")
        print("=" * 70)
        
        # 打印最佳验证分数
        print(f"\n✅ Best Validation Macro-F1: {result['best_val_f1']:.4f}")
        
        # 打印任务级指标表格
        print("\n📊 Per-Task Test Metrics:")
        table, avg_acc, avg_f1m, avg_f1w = print_task_metrics_table(test_metrics, task_names)
        print(table)
        
        # 打印汇总统计
        print("\n" + "=" * 70)
        print("📈 SUMMARY STATISTICS")
        print("=" * 70)
        print(f"{'Average Accuracy:':<30} {avg_acc:.4f}")
        print(f"{'Average Macro-F1:':<30} {avg_f1m:.4f}")
        print(f"{'Average Weighted-F1:':<30} {avg_f1w:.4f}")
        print("=" * 70)
        
        # 保存结果
        if args.save_artifacts:
            result_path = save_results(result, args)
            print(f"\n💾 Results saved to: {result_path}")
            print(f"   Model checkpoint: {result.get('model_file')}")
            print(f"   Log directory: {result.get('script_dir')}")
        
        # 返回主要指标供后续使用
        return {
            'avg_accuracy': avg_acc,
            'avg_macro_f1': avg_f1m,
            'avg_weighted_f1': avg_f1w,
            'best_val_f1': result['best_val_f1'],
            'test_metrics': test_metrics
        }
        
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()