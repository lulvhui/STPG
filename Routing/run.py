#!/usr/bin/env python3
"""
运行 MSAIRS 双层路由多任务模型的脚本

功能：
1. 训练模型
2. 输出各任务的 ACC, Macro-F1, Weighted-F1
3. 输出平均 Macro-F1
4. 使用 GPU 卡 0

使用方法：
    python run.py
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

# 设置 GPU 卡 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from my_dataload import CFG, TASK_CONFIG, ROUTING_CONFIG, seed_everything
from my_dataload import load_data, all_sticker_features, meme_caption_map
from my_model import MultiTaskModel
from my_main import (
    load_and_filter_data,
    create_dataloaders,
    train_model,
    evaluate,
    log_message
)


def print_separator(char='=', length=80):
    """打印分隔线"""
    print(char * length)


def print_results(test_metrics, task_names):
    """打印详细的评估结果"""
    print_separator()
    print("📊 测试集评估结果")
    print_separator()
    
    # 表头
    print(f"{'Task':<15} | {'ACC':>8} | {'Macro-F1':>10} | {'Weighted-F1':>12}")
    print_separator('-')
    
    # 各任务结果
    avg_acc = 0.0
    avg_mf1 = 0.0
    avg_wf1 = 0.0
    
    for task_name in task_names:
        metrics = test_metrics[task_name]
        acc = metrics['acc']
        mf1 = metrics['f1_macro']
        wf1 = metrics['f1_weighted']
        
        avg_acc += acc
        avg_mf1 += mf1
        avg_wf1 += wf1
        
        print(f"{task_name:<15} | {acc:>8.4f} | {mf1:>10.4f} | {wf1:>12.4f}")
    
    print_separator('-')
    
    # 平均结果
    num_tasks = len(task_names)
    avg_acc /= num_tasks
    avg_mf1 /= num_tasks
    avg_wf1 /= num_tasks
    
    print(f"{'Average':<15} | {avg_acc:>8.4f} | {avg_mf1:>10.4f} | {avg_wf1:>12.4f}")
    print_separator()
    
    # 重点突出平均 Macro-F1
    print(f"\n🎯 平均 Macro-F1: {avg_mf1:.4f}")
    print_separator()
    
    return {
        'avg_acc': avg_acc,
        'avg_mf1': avg_mf1,
        'avg_wf1': avg_wf1,
        'task_metrics': test_metrics
    }


def save_results(results, output_dir, timestamp):
    """保存结果到 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f'results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到: {results_file}")


def main():
    """主函数"""
    print_separator()
    print("🚀 MSAIRS 双层路由多任务模型训练")
    print_separator()
    
    # 时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 设置随机种子
    seed_everything(CFG['seed'])
    print(f"🔧 随机种子: {CFG['seed']}")
    print(f"📅 时间戳: {timestamp}")
    
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {device}")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    # 配置信息
    print(f"\n⚙️  配置参数:")
    print(f"   - Epochs: {CFG['epochs']}")
    print(f"   - Batch Size (Train): {CFG['train_bs']}")
    print(f"   - Batch Size (Valid): {CFG['valid_bs']}")
    print(f"   - Learning Rate: {CFG['lr']}")
    print(f"   - Weight Decay: {CFG['weight_decay']}")
    print(f"   - Patience: {CFG['patience']}")
    
    print(f"\n📦 路由配置:")
    print(f"   - 共享专家数: {ROUTING_CONFIG['num_shared_experts']}")
    print(f"   - 任务专家数: {ROUTING_CONFIG['num_task_experts']}")
    print(f"   - 专家隐藏维度: {ROUTING_CONFIG['expert_hidden']}")
    print(f"   - 路由器隐藏维度: {ROUTING_CONFIG['router_hidden']}")
    
    # 加载数据
    print(f"\n📂 加载数据...")
    train, valid, test = load_and_filter_data()
    print(f"   - 训练集: {len(train)} 样本")
    print(f"   - 验证集: {len(valid)} 样本")
    print(f"   - 测试集: {len(test)} 样本")
    
    # 创建 DataLoader
    print(f"\n🔄 创建 DataLoader...")
    train_loader, valid_loader, test_loader = create_dataloaders(train, valid, test)
    print(f"   - 训练批次数: {len(train_loader)}")
    print(f"   - 验证批次数: {len(valid_loader)}")
    print(f"   - 测试批次数: {len(test_loader)}")
    
    # 任务列表
    task_names = list(TASK_CONFIG.keys())
    print(f"\n📋 任务列表 ({len(task_names)} 个):")
    for i, task_name in enumerate(task_names, 1):
        num_classes = TASK_CONFIG[task_name]['num_classes']
        print(f"   {i}. {task_name} ({num_classes} 类)")
    
    # 创建模型
    print(f"\n🏗️  创建模型...")
    model = MultiTaskModel(
        routing_config=ROUTING_CONFIG,
        use_routing=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - 总参数量: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
    
    # 训练模型
    print(f"\n🎓 开始训练...")
    print_separator()
    
    best_val_f1, test_metrics, test_avg_f1 = train_model(
        model, 
        train_loader, 
        valid_loader, 
        test_loader, 
        device, 
        task_names,
        paths=None,  # 不保存中间文件
        patience=CFG['patience'],
        save_artifacts=False
    )
    
    # 打印结果
    print_separator()
    results = print_results(test_metrics, task_names)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    save_results(results, output_dir, timestamp)
    
    print(f"\n✅ 训练完成！")
    print(f"🎯 最佳验证 Macro-F1: {best_val_f1:.4f}")
    print(f"🎯 测试集平均 Macro-F1: {results['avg_mf1']:.4f}")
    print_separator()
    
    return results


if __name__ == '__main__':
    results = main()