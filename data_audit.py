"""
数据版本对比审计脚本
检查 4444 条 vs 2003 条数据中的元素分布差异
"""
import numpy as np
import os

def audit_data(data_path, label_path, name):
    """审计一个数据集"""
    try:
        data = np.load(data_path, allow_pickle=True)
        label = np.load(label_path, allow_pickle=True)
    except Exception as e:
        print(f"  [错误] {name}: {e}")
        return None
    
    if len(data) == 0:
        print(f"  [错误] {name}: 数据为空")
        return None
    
    # 统计元素分布
    element_count = {}
    atom_total = 0
    samples_with_I = 0
    samples_with_Br = 0
    
    for idx, sample in enumerate(data):
        sample_has_I = False
        sample_has_Br = False
        
        # 获取三个图
        c_graph, a_graph, r_graph = sample[0], sample[1], sample[2]
        
        for graph in [c_graph, a_graph, r_graph]:
            if graph is None or not hasattr(graph, 'x') or graph.x is None:
                continue
            
            for atom_feat in graph.x:
                at = int(atom_feat[0].item())
                element_count[at] = element_count.get(at, 0) + 1
                atom_total += 1
                
                if at == 53:  # Iodine
                    sample_has_I = True
                if at == 35:  # Bromine
                    sample_has_Br = True
        
        if sample_has_I:
            samples_with_I += 1
        if sample_has_Br:
            samples_with_Br += 1
    
    print(f"\n[{name}]")
    print(f"  数据条数: {len(data)}")
    print(f"  总原子数: {atom_total}")
    print(f"  元素分布:")
    
    elem_map = {
        5: 'B(硼)', 6: 'C(碳)', 7: 'N(氮)', 8: 'O(氧)', 9: 'F(氟)',
        15: 'P(磷)', 16: 'S(硫)', 17: 'Cl(氯)', 35: 'Br(溴)', 53: 'I(碘)'
    }
    
    for at in sorted(element_count.keys()):
        name_str = elem_map.get(at, f'At{at}')
        cnt = element_count[at]
        print(f"    {name_str}: {cnt}")
    
    print(f"  含碘(I)的样本: {samples_with_I}")
    print(f"  含溴(Br)的样本: {samples_with_Br}")
    
    return {
        'samples': len(data),
        'elements': element_count,
        'has_I': samples_with_I > 0,
        'has_Br': samples_with_Br > 0
    }

# 审计两个版本
path1 = 'GNN_for_property_prediction/clean/data.npy'
path2 = 'processed_tri_data/data.npy'

print("="*60)
print("数据版本对比")
print("="*60)

result1 = audit_data(path1, path1.replace('data.npy', 'label.npy'), '版本1 (4444条)')
result2 = audit_data(path2, path2.replace('data.npy', 'label.npy'), '版本2 (2003条)')

print("\n" + "="*60)
print("结论:")
print("="*60)
if result1 and result2:
    print(f"版本1 (GNN_for_property_prediction/clean): 样本数={result1['samples']}, 有I={result1['has_I']}, 有Br={result1['has_Br']}")
    print(f"版本2 (processed_tri_data):          样本数={result2['samples']}, 有I={result2['has_I']}, 有Br={result2['has_Br']}")
    print(f"\n⚠️  问题: 你的模型用版本2(2003条)训练, 但解释时可能用了版本1(4444条)")
    print(f"        这导致元素重要性图表不匹配!")
