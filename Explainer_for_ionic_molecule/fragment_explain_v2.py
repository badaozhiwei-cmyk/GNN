"""
fragment_explain_v2.py
===================================================
鍒跺喎鍓-绂诲瓙娑蹭綋涓夊浘浣撶郴 GNNExplainer 鐗囨甸噸瑕佹у垎鏋 (V2)

涓 CO2 鏃х増 fragment_explain.py 鐨勬牳蹇冨樊寮傦細
  1. 鐗囨靛瓧鍏 (Frag_importance)锛氭崲鎴愬埗鍐峰墏浣撶郴鐨勫叧閿瀹樿兘鍥
       鈥 鍒跺喎鍓備晶锛欳HF鈧傘丆F鈧冦丆H鈧侳銆丆F鈧 绛夋盁浠ｅ熀鍥
       鈥 闃寸诲瓙渚э細纾洪叞浜氳兒鍩哄洟銆佺：閰告牴銆佹盁纾洪吀鍩恒佺７閰搁叝鍩
       鈥 闃崇诲瓙渚э細鍜鍞戠幆銆佸悺鍟剁幆銆佸ｇ７銆佺兎鍩洪摼
  2. 鏁版嵁闆嗭細浠 Dataset_explain_v2.IL_set_v2 鍔犺浇涓夊浘鏁版嵁
  3. 妯″瀷璺寰勶細鑷鍔ㄦ娴 checkpoints/ 涓嬫渶鏂扮殑 best_model_para.pth
  4. 杈撳嚭鏂囦欢锛歠rag_importance_v2.npy / fragment_score_v2.png

杩愯屾柟寮忥紙鍦 Kaggle 鎴栨湰鍦帮級:
  cd Explainer_for_ionic_molecule
  python fragment_explain_v2.py --model_path ../checkpoints/best_model_para.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import DataLoader

# 鈹鈹 灏 GNN_for_property_prediction 鍔犲叆鎼滅储璺寰 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import GIN
from Explainer_v2 import IL_Explainer_v2
from Dataset_explain_v2 import IL_set_v2

# 鈹鈹 GIN 璁缁冩椂鐨 Args锛坘ey鍚嶅繀椤讳笌 GIN_Runner.py 鍜 Model.py 瀹屽叏涓鑷达級鈹鈹鈹鈹
# 娉ㄦ剰锛氳繖閲岀敤鐨勬槸鍒跺喎鍓備笁鍥句綋绯荤殑璁缁冨弬鏁帮紙GIN_Runner.py 涓鐨 Args锛
Args = {
    'num_gin_layer': 5,    # Model.py 璇诲彇 args['num_gin_layer']
    'emb_dim': 300,
    'feat_dim': 512,
    'drop_ratio': 0.2,
    'pool': 'mean',
}

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 鍒跺喎鍓備綋绯讳笓灞炵墖娈靛瓧鍏
# 姣忎釜 key 鏄涓涓 SMILES 瀛愮粨鏋勶紙鐢 RDKit SMARTS 涔熷彲锛
# 姣忎釜 value 鏄涓涓绌哄垪琛锛岃繍琛屾椂浼氭敹闆嗘瘡鏉℃暟鎹閲岃ョ墖娈电殑璐＄尞鍒嗘暟
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 鎸夊師瀛愬厓绱犵被鍨嬪垎缁勶紙鐢ㄥ師瀛愬簭鏁 x[:,0] 鐩存帴璇嗗埆锛屾棤闇 SMILES锛
# 鍘熷瓙搴忔暟瀵瑰簲: C=6, N=7, O=8, F=9, P=15, S=16, Cl=17, Br=35, B=5, I=53
# 杩欏楀垎缁勬柟娉曞湪鍒跺喎鍓-绂诲瓙娑蹭綋浣撶郴涓鎰忎箟鏄庣‘锛
#   F  鈫 鍒跺喎鍓傜殑鏍稿績姘熶唬鍩哄洟锛屽喅瀹 C-H 閰告у拰鍋舵瀬寮哄害
#   N  鈫 鍜鍞/鍚″暥闃崇诲瓙鐜蹇冿紝浠ュ強 Tf2N鈦 鐨勪簹姘ㄥ熀涓蹇
#   S  鈫 纾洪叞鍩猴紙Tf2N鈦, OTF鈦 绛夐槾绂诲瓙鐨勬椿鎬т綅鐐癸級
#   O  鈫 缇ч吀鏍广佺７閰搁叝鍩恒佺：閰告牴鐨勬哀鍘熷瓙锛堟阿閿鍙椾綋锛
#   P  鈫 瀛ｇ７闃崇诲瓙锛圥66614+, P4442+锛/ 纾烽吀閰闃寸诲瓙
#   C  鈫 鐑峰熀閾撅紙褰卞搷绌洪棿浣嶉樆鍜岃嚜鐢变綋绉锛
#   Cl 鈫 鍚姘鍒跺喎鍓傦紙R22, R114 绛 HCFC/CFC 绫伙級
#   Br 鈫 鍚婧村崵浠ｇ儍锛圧22B1 绛夛級
#   B  鈫 鍥涙盁纭奸吀鏍 BF4鈦
#   I  鈫 纰樼诲瓙 I鈦
ELEMENT_MAP = {
    5:  'B  (纭硷紝BF4鈦)',
    6:  'C  (纰筹紝鐑峰熀閾/鍒跺喎鍓傞ㄦ灦)',
    7:  'N  (姘锛屽挭鍞/鍚″暥/Tf2N鈦)',
    8:  'O  (姘э紝缇ч吀鏍/纾烽吀閰/纾洪吀鏍)',
    9:  'F  (姘燂紝姘熶唬鍒跺喎鍓傚叧閿鍏冪礌)',
    15: 'P  (纾凤紝瀛ｇ７闃崇诲瓙/纾烽吀閰闃寸诲瓙)',
    16: 'S  (纭锛岀：閰板熀闃寸诲瓙娲绘т腑蹇)',
    17: 'Cl (姘锛孒CFC/CFC 绫诲埗鍐峰墏)',
    35: 'Br (婧达紝鍚婧村崵浠ｇ儍)',
    53: 'I  (纰橈紝纰樼诲瓙闃寸诲瓙)',
}
# 鍒濆嬪寲姣忕嶅厓绱犵殑閲嶈佹ф敹闆嗗垪琛
Element_importance = {v: [] for v in ELEMENT_MAP.values()}


def plot_element_importance(result: dict, save_path: str):
    """
    缁樺埗鎸夊厓绱犲垎缁勭殑鍘熷瓙閲嶈佹ф按骞虫潯錮ef main(model_path: str, data_root: str, explainer_epochs: int = 100, num_samples: int = -1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[杩愯岃惧嘳 {device}")

    # 鈹鈹 1. 鍔犺浇妯″瀷 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    print(f"[妯″瀷] 鍔犺浇: {model_path}")
    model = GIN(Args).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 鈹鈹 2. 鍔犺浇鏁版嵁闆 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    data_npy  = os.path.join(data_root, 'data.npy')
    label_npy = os.path.join(data_root, 'label.npy')
    dataset = IL_set_v2(data_npy_path=data_npy, label_npy_path=label_npy)

    # batch_size=1锛氭瘡娆¤В閲婁竴涓鏍锋湰锛圙NNExplainer 鐨勫繀瑕佹潯浠讹級
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=IL_set_v2.collate_fn
    )

    # 鈹鈹 3. 鍒濆嬪寲 Explainer_v2锛堝洖褰掍笓鐢ㄧ増锛夆攢鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    # IL_Explainer_v2 閲嶅啓浜 __loss__锛氱敤 MSE 鏇夸唬鍘熺増鐨勫垎绫 NLL Loss
    # 杩欐牱姊搴︽柟鍚戞墠鑳芥ｇ‘鎸囧兼帺鐮佷紭鍖
    explainer = IL_Explainer_v2(
        model, epochs=explainer_epochs, lr=0.01
    )

    # 鈹鈹 4. 涓诲惊鐜锛氶愭牱鏈瑙ｉ噴 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    elem_imp  = {v: [] for v in ELEMENT_MAP.values()}  # 姣忕嶅厓绱犵殑鐩稿归噸瑕佹у垪琛
    node_feat_imp = np.zeros(5)                         # 5 绉嶈妭鐐圭壒寰佺殑绱鍔犻噸瑕佹

    total_to_run = len(loader) if num_samples <= 0 else min(num_samples, len(loader))
    print(f"[鏁版嵁瑙勬ā] 鎬绘暟鎹閲: {len(loader)} 鏉★紝鏈娆″垎鏋愭牱鏈鏁伴檺鍒: {total_to_run} 鏉")
    bar = tqdm(total=total_to_run, desc='Explaining', dynamic_ncols=True)

    explained_count = 0
    for G, cond, label, num_bonds_list in loader:
        if num_samples > 0 and explained_count >= num_samples:
            break

        G    = G.to(device)
        cond = cond.to(device)
        num_bond = num_bonds_list[0]

        try:
            node_feat_mask, edge_mask = explainer.explain_graph(G, cond)
        except Exception as e:
            bar.update()
            continue

        explained_count += 1
        node_feat_imp += node_feat_mask.cpu().numpy()

        # 鈹鈹 瑙ｆ瀽鍘熷瓙绾ф帺鐮 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
        # 鍏ㄥ眬铏氭嫙杈圭殑鎺╃爜鍙嶆槧浜嗘瘡涓鍘熷瓙瀵归勬祴鍊肩殑璐＄尞绋嬪害銆
        # 鍥犱负 add_global 缁欐瘡涓鐪熷疄鍘熷瓙 i 娣诲姞浜嗕竴鏉¤櫄鎷熻竟锛坕鈫抔lobal, global鈫抜锛夛紝
        # 鎵浠 edge_mask[num_bond:] 鐨勫墠鍗婃 = 姝ｅ悜鎺╃爜锛屽悗鍗婃 = 鍙嶅悜鎺╃爜
        global_mask = edge_mask[num_bond:].cpu()
        num_real_atom = global_mask.shape[0] // 2       # 涓嶅惈鍏ㄥ眬鑺傜偣鏈韬
        fwd = global_mask[:num_real_atom]
        bwd = global_mask[num_real_atom:]
        atom_imp = ((fwd + bwd) / 2).numpy()            # shape: [num_real_atom]
        mean_imp = atom_imp.mean()                      # 鍏ㄥ浘鍘熷瓙閲嶈佹у潎鍊硷紙鍩哄噯锛

        # 鈹鈹 鎸夊厓绱犵被鍨嬪垎缁勶紝璁＄畻鐩稿归噸瑕佹 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
        # x[:, 0] 瀛樼殑灏辨槸鍘熷瓙搴忔暟锛堟潵鑷 prepare_tri_graph_data.py 鐨勫浘鏋勫缓锛
        # 鏈鍚庝竴涓鑺傜偣鏄鍏ㄥ眬铏氭嫙鑺傜偣锛堝師瀛愬簭鏁=0锛夛紝璺宠繃
        atom_types = G.x[:num_real_atom, 0].cpu().numpy()   # shape: [num_real_atom]

        for atom_idx, (at, imp) in enumerate(zip(atom_types, atom_imp)):
            at = int(at)
            if at in ELEMENT_MAP:
                elem_label = ELEMENT_MAP[at]
                # 鐩稿归噸瑕佹 = 璇ュ師瀛愬緱鍒 - 鍏ㄥ浘鍧囧
                # 姝ｅ硷細璇ュ師瀛愭槸楂樹簬骞冲潎姘村钩鐨勫叧閿浣嶇偣
                # 璐熷硷細璇ュ師瀛愭槸鎯版х殑鑳屾櫙缁撴瀯
                elem_imp[elem_label].append(float(imp - mean_imp))

        bar.update()

    bar.close()

    # 鈹鈹 5. 姹囨绘瘡绉嶅厓绱犵殑骞冲潎鐩稿归噸瑕佹 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    result = {}
    for elem_label, scores in elem_imp.items():
        result[elem_label] = float(np.mean(scores)) if scores else float('nan')

    # 鈹鈹 6. 淇濆瓨鍘熷嬫暟鎹 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    out_dir = os.path.join(os.path.dirname(__file__), 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'element_importance_v2_raw.npy'),
            np.array(list(elem_imp.items()), dtype=object), allow_pickle=True)
    np.save(os.path.join(out_dir, 'node_feat_imp_v2.npy'),
            node_feat_imp, allow_pickle=False)

    # 鈹鈹 7. 缁樺浘 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    plot_element_importance(result, os.path.join(out_dir, 'element_score_v2.png'))

    # 鈹鈹 8. 鑺傜偣鐗瑰緛閲嶈佹у浘 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    feat_names = ['atomic_number', 'atomic_degree', 'charge', 'hybridization', 'Aromatic']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(feat_names, node_feat_imp, color='steelblue')
    ax.set_ylabel('Cumulative Importance Score')
    ax.set_title('Node Feature Importance\n(Tri-Graph Refrigerant V2)')
    plt.tight_layout()
    nf_path = os.path.join(out_dir, 'node_feature_v2.png')
    plt.savefig(nf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [鍥捐〃] 宸蹭繚瀛: {nf_path}")

    print("\n[瀹屾垚] 鐗囨甸噸瑕佹у垎鏋愬畬姣曪紒")
    print(f"  杈撳嚭鐩褰: {out_dir}")
    return result


# 鈹鈹 鍛戒护琛屽叆鍙 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='鍒跺喎鍓備綋绯 GNNExplainer 鐗囨甸噸瑕佹 V2')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join(ROOT, 'checkpoints', 'best_model_para.pth'),
        help='璁缁冨ソ鐨勬ā鍨嬫潈閲嶈矾寰勶紙榛樿: checkpoints/best_model_para.pth锛'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=os.path.join(ROOT, 'processed_tri_data'),
        help='processed_tri_data 鐩褰曡矾寰'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='GNNExplainer 浼樺寲杞鏁帮紙榛樿: 100锛岃秺澶ц秺绮剧‘浣嗚秺鎱锛'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='闄愬埗鍒嗘瀽鏍锋湰鐨勬暟閲忥紙榛樿: -1 琛ㄧず鍒嗘瀽鍏ㄩ噺鏁版嵁锛岃嫢鎸囧畾姝ｆ暟濡 500 鍒欓殢鏈哄彇鍓 500 涓鏍锋湰鍒嗘瀽锛屽缓璁 Colab 璁句负 500 浠ラ槻瓒呮椂锛'
    )
    args_cli = parser.parse_args()

    result = main(
        model_path    = args_cli.model_path,
        data_root     = args_cli.data_root,
        explainer_epochs = args_cli.epochs,
        num_samples   = args_cli.num_samples
    )

    print("\n鈹鈹 鏈缁堢墖娈甸噸瑕佹ф眹鎬 鈹鈹")
    for frag, score in sorted(result.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True):
        print(f"  {frag:<20s}  {score:.4f}")mp[elem_label].append(float(imp - mean_imp))

        bar.update()

    bar.close()

    # 鈹鈹 5. 姹囨绘瘡绉嶅厓绱犵殑骞冲潎鐩稿归噸瑕佹 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    result = {}
    for elem_label, scores in elem_imp.items():
        result[elem_label] = float(np.mean(scores)) if scores else float('nan')

    # 鈹鈹 6. 淇濆瓨鍘熷嬫暟鎹 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    out_dir = os.path.join(os.path.dirname(__file__), 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'element_importance_v2_raw.npy'),
            np.array(list(elem_imp.items()), dtype=object), allow_pickle=True)
    np.save(os.path.join(out_dir, 'node_feat_imp_v2.npy'),
            node_feat_imp, allow_pickle=False)

    # 鈹鈹 7. 缁樺浘 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    plot_element_importance(result, os.path.join(out_dir, 'element_score_v2.png'))

    # 鈹鈹 8. 鑺傜偣鐗瑰緛閲嶈佹у浘 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
    feat_names = ['atomic_number', 'atomic_degree', 'charge', 'hybridization', 'Aromatic']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(feat_names, node_feat_imp, color='steelblue')
    ax.set_ylabel('Cumulative Importance Score')
    ax.set_title('Node Feature Importance\n(Tri-Graph Refrigerant V2)')
    plt.tight_layout()
    nf_path = os.path.join(out_dir, 'node_feature_v2.png')
    plt.savefig(nf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [鍥捐〃] 宸蹭繚瀛: {nf_path}")

    print("\n[瀹屾垚] 鐗囨甸噸瑕佹у垎鏋愬畬姣曪紒")
    print(f"  杈撳嚭鐩褰: {out_dir}")
    return result


# 鈹鈹 鍛戒护琛屽叆鍙 鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹鈹
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='鍒跺喎鍓備綋绯 GNNExplainer 鐗囨甸噸瑕佹 V2')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join(ROOT, 'checkpoints', 'best_model_para.pth'),
        help='璁缁冨ソ鐨勬ā鍨嬫潈閲嶈矾寰勶紙榛樿: checkpoints/best_model_para.pth锛'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=os.path.join(ROOT, 'processed_tri_data'),
        help='processed_tri_data 鐩褰曡矾寰'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='GNNExplainer 浼樺寲杞鏁帮紙榛樿: 100锛岃秺澶ц秺绮剧‘浣嗚秺鎱锛'
    )
    args_cli = parser.parse_args()

    result = main(
        model_path    = args_cli.model_path,
        data_root     = args_cli.data_root,
        explainer_epochs = args_cli.epochs
    )

    print("\n鈹鈹 鏈缁堢墖娈甸噸瑕佹ф眹鎬 鈹鈹")
    for frag, score in sorted(result.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True):
        print(f"  {frag:<20s}  {score:.4f}")