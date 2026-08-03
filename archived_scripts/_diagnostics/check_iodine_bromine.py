"""
========================================
GNN 项目诊断工具 - I/Br 元素分析
========================================
快速检查 4444 条数据中碘(I)和溴(Br)的分布
用法: python check_iodine_bromine.py
"""

import numpy as np
import sys
import os

# 添加项目路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

def check_element_distribution(data_path, label_path, target_elements=[53, 35]):
    """
    分析特定元素在数据中的分布
    target_elements: [53(I), 35(Br)] 默认检查碘和溴
    """
    print(f"加载数据: {data_path}")
    try:
        data = np.load(data_path, allow_pickle=True)
        label = np.load(label_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None
    
    print(f"✓ 数据加载成功: {len(data)} 条样本\n")
    
    elem_map = {53: 'I(碘)', 35: 'Br(溴)'}
    results = {}
    
    for target_elem in target_elements:
        elem_name = elem_map.get(target_elem, f'Element{target_elem}')
        
        # 统计
        samples_with_elem = []
        atom_count = 0
        max_atoms_in_sample = 0
        
        for idx, sample in enumerate(data):
            c_graph, a_graph, r_graph = sample[0], sample[1], sample[2]
            sample_atom_count = 0
            has_elem = False
            
            for graph in [c_graph, a_graph, r_graph]:
                if graph is None or not hasattr(graph, 'x') or graph.x is None:
                    continue
                
                for atom_feat in graph.x:
                    at = int(atom_feat[0].item())
                    if at == target_elem:
                        atom_count += 1
                        sample_atom_count += 1
                        has_elem = True
            
            if has_elem:
                samples_with_elem.append({
                    'idx': idx,
                    'atom_count': sample_atom_count,
                    'solubility': float(label[idx])
                })
            
            max_atoms_in_sample = max(max_atoms_in_sample, sample_atom_count)
        
        results[elem_name] = {
            'total_atoms': atom_count,
            'samples_count': len(samples_with_elem),
            'samples_ratio': len(samples_with_elem) / len(data) * 100,
            'max_atoms_in_one_sample': max_atoms_in_sample,
            'samples': samples_with_elem
        }
    
    # 打印结果
    print("="*60)
    print("📊 元素分布统计")
    print("="*60)
    
    for elem_name, info in results.items():
        print(f"\n{elem_name}:")
        print(f"  总原子数: {info['total_atoms']}")
        print(f"  含该元素的样本: {info['samples_count']} 条 ({info['samples_ratio']:.2f}%)")
        print(f"  最多的样本中含 {info['max_atoms_in_one_sample']} 个原子")
        
        if info['samples']:
            sols = [s['solubility'] for s in info['samples']]
            print(f"  溶解度范围: [{min(sols):.4f}, {max(sols):.4f}]")
            print(f"  溶解度平均值: {np.mean(sols):.4f}")
            print(f"  溶解度标准差: {np.std(sols):.4f}")
            
            # 显示前5个样本
            print(f"\n  前5个样本:")
            for i, s in enumerate(info['samples'][:5]):
                print(f"    样本 #{s['idx']}: {s['atom_count']} 个 {elem_name.split('(')[0]}, 溶解度={s['solubility']:.4f}")
    
    # 两个都有的样本
    print("\n" + "="*60)
    i_samples = set(s['idx'] for s in results.get('I(碘)', {}).get('samples', []))
    br_samples = set(s['idx'] for s in results.get('Br(溴)', {}).get('samples', []))
    both = i_samples & br_samples
    
    print(f"同时含有 I 和 Br 的样本: {len(both)} 条")
    if both:
        print(f"  样本索引: {sorted(list(both))[:10]}")
    
    return results


if __name__ == '__main__':
    # 检查 Colab 上的 4444 条数据
    data_path = os.path.join(ROOT, 'GNN_for_property_prediction/clean/data.npy')
    label_path = os.path.join(ROOT, 'GNN_for_property_prediction/clean/label.npy')
    
    results = check_element_distribution(data_path, label_path)
    
    print("\n" + "="*60)
    print("✓ 诊断完成！")
    print("="*60)
