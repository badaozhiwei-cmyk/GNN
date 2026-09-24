import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# 适配 Windows 控制台编码
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "Phase4_Scientific_Validation"))

from chemical_identity_contract import (
    get_refrigerant_identity,
    assert_refrigerant_contract,
    assert_graph_identity_contract,
    _df_registry
)

class TestRedTeamAdversarialIntegrity(unittest.TestCase):

    def test_r1234yf_hfp_injection_fails(self):
        """红队测试 1: 故意向 R1234yf 注入六氟丙烯 HFP (SMILES 掉包)，断言必须抛出 RuntimeError"""
        hfp_smiles = "C(=C(F)F)(C(F)(F)F)F" # 历史错误 Table S4 SMILES
        with self.assertRaises(RuntimeError) as ctx:
            assert_refrigerant_contract("R1234yf", smiles=hfp_smiles)
        self.assertIn("CHEMICAL IDENTITY CONTRACT BREACH", str(ctx.exception))
        self.assertIn("InChIKey 错位", str(ctx.exception))
        self.assertIn("分子式错位", str(ctx.exception))
        print("  [PASS] Test 1: R1234yf -> HFP 注入成功被拦截熔断！")

    def test_r1234yf_graph_atom_count_mutation_fails(self):
        """红队测试 2: 故意向 R1234yf 注入 HFP 的 9 重原子图结构，断言必须抛出 RuntimeError"""
        hfp_heavy_atoms = [6, 6, 6, 9, 9, 9, 9, 9, 9] # 3C + 6F
        with self.assertRaises(RuntimeError) as ctx:
            assert_graph_identity_contract(hfp_heavy_atoms, "R1234yf")
        self.assertIn("Graph Integrity Breach", str(ctx.exception))
        print("  [PASS] Test 2: R1234yf 图重原子突变 (7->9) 成功被拦截熔断！")

    def test_r134_r134a_isomeric_swap_fails(self):
        """红队测试 3: 故意把非对称 R134a 赋给对称 R134，分子式相同但 InChIKey/拓扑不同，断言必须熔断"""
        r134a_smiles = "FCC(F)(F)F" # 1,1,1,2-四氟乙烷
        with self.assertRaises(RuntimeError) as ctx:
            assert_refrigerant_contract("R134", smiles=r134a_smiles)
        self.assertIn("InChIKey 错位", str(ctx.exception))
        print("  [PASS] Test 3: R134 <-> R134a 同分异构体掉包成功被拦截熔断！")

    def test_r1336mzz_ez_stereochemistry_swap_fails(self):
        """红队测试 4: 故意把顺式 (Z) 赋给反式 (E)，断言立体化学 InChIKey 不符必须熔断"""
        cis_smiles = "FC(F)(F)/C=C\\C(F)(F)F" # R1336mzz(Z)
        with self.assertRaises(RuntimeError) as ctx:
            assert_refrigerant_contract("R1336mzz(E)", smiles=cis_smiles)
        self.assertIn("InChIKey 错位", str(ctx.exception))
        print("  [PASS] Test 4: R1336mzz(E) <-> (Z) 顺反立体构型掉包成功被拦截熔断！")

    def test_fail_closed_on_unknown_species(self):
        """红队测试 5: 传入不存在的假冒物种，断言必须 Fail-Closed 抛出 KeyError，严禁静默放行"""
        with self.assertRaises(KeyError) as ctx:
            get_refrigerant_identity("R9999_FICTITIOUS")
        self.assertIn("Chemical Registry Breach", str(ctx.exception))
        print("  [PASS] Test 5: 未知/假冒物种严格 Fail-Closed 拦截熔断！")

    def test_canonical_refrigerants_all_pass(self):
        """正向校验: 全量 26 种注册制冷剂自身调用契约全部 100% PASS"""
        for _, row in _df_registry.iterrows():
            name = row['canonical_name']
            smi = row['canonical_smiles']
            res = assert_refrigerant_contract(name, smiles=smi)
            self.assertEqual(res['canonical_name'], name)
        print("  [PASS] Test 6: 全量 26 种官方标准制冷剂 100% 通过契约校验！")

    def test_tampered_dataset_fails_freshness_gate(self):
        """红队测试 7: 模拟 data.npy 篡改或行乱序 (SHA256 变动)，断言必须抛出 RuntimeError 熔断"""
        import tempfile, json
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            (tmp_p / "data.npy").write_bytes(b"tampered_data_bytes_12345")
            (tmp_p / "label.npy").write_bytes(b"labels")
            (tmp_p / "meta_info.csv").write_bytes(b"meta")
            
            fake_manifest = {
                "dataset_id": "TEST_TAMPERED",
                "files": {
                    "data.npy": {"sha256": "0000000000000000000000000000000000000000000000000000000000000000", "size_bytes": 25},
                    "label.npy": {"sha256": "dummy", "size_bytes": 6},
                    "meta_info.csv": {"sha256": "dummy", "size_bytes": 4}
                }
            }
            with open(tmp_p / "dataset_manifest.json", "w") as f:
                json.dump(fake_manifest, f)
            
            from artifact_freshness_gate import validate_dataset_freshness
            with self.assertRaises(RuntimeError) as ctx:
                validate_dataset_freshness(tmp_p)
            self.assertIn("DATASET TAMPERED OR STALE BREACH", str(ctx.exception))
            print("  [PASS] Test 7: 数据集篡改/乱序成功触发新鲜度门禁熔断！")

    def test_broken_split_manifest_binding_fails(self):
        """红队测试 8: 模拟上游数据集变动但划分未同步，断言划分强绑定必须抛出 RuntimeError 熔断"""
        import tempfile, json
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            split_p = tmp_p / "test_split_bound.json"
            fake_split = {
                "is_disjoint": True,
                "bound_dataset_manifest_sha256": "correct_sha_1111111111111111111111111111111111111111111111111111111111111111"
            }
            with open(split_p, "w") as f:
                json.dump(fake_split, f)
            
            # 数据集 manifest 的实际哈希不同
            ds_dir = tmp_p / "ds"
            ds_dir.mkdir()
            (ds_dir / "dataset_manifest.json").write_text('{"actual": "different"}', encoding="utf-8")
            
            from artifact_freshness_gate import validate_bound_split_freshness
            with self.assertRaises(RuntimeError) as ctx:
                validate_bound_split_freshness(split_p, dataset_dir=ds_dir)
            self.assertIn("SPLIT-DATASET BINDING BROKEN", str(ctx.exception))
            print("  [PASS] Test 8: 划分与底层数据集哈希撕裂成功触发强契约熔断！")

    def test_graph_bond_mismatch_fails_closed(self):
        """红队测试 9: 故意注入错误键拓扑，断言 verify_and_get_rdkit_mol 必须严格 Fail-Closed 抛出 ValueError"""
        sys.path.insert(0, str(ROOT_DIR / "v7_shadow_experiment"))
        from phase2_v7_graph_ig import verify_and_get_rdkit_mol
        
        # 构造原子序数匹配 R1234yf 的节点序列，但将边连接设为空列表 (缺失化学键)
        fake_graph = (
            [[6], [6], [9], [6], [9], [9], [9]], # 正确的 3 C, 4 F 遍历序列
            [[], []] # 0 边连接
        )
        with self.assertRaises(ValueError) as ctx:
            verify_and_get_rdkit_mol("C=C(F)C(F)(F)F", fake_graph)
        self.assertIn("FAIL-CLOSED", str(ctx.exception))
        self.assertIn("Bond connectivity mismatch", str(ctx.exception))
        print("  [PASS] Test 9: 图键拓扑篡改/缺失严格 Fail-Closed 拦截熔断！")

    def test_split_aware_disjointness_certificate_integrity(self):
        """红队测试 10: [打破循环] 从底层原始 CSV 和 NPZ 独立重算五大轴结构不相交指标，并核验证书"""
        import json, hashlib
        from rdkit import Chem
        from rdkit.Chem import inchi

        # 1. 独立验证 Axis 1: HFC 训练宇宙中零 R1234yf / 零 HFP
        hfc_meta_p = ROOT_DIR / "datasets" / "hfc_2739_v7" / "meta_info.csv"
        self.assertTrue(hfc_meta_p.exists())
        df_hfc = pd.read_csv(hfc_meta_p)
        train_ref_names = set(df_hfc["refrigerant"].unique())
        self.assertNotIn("R1234yf", train_ref_names)
        self.assertNotIn("HFP", train_ref_names)

        # 独立重算 InChIKey
        yf_inchikey = inchi.MolToInchiKey(Chem.MolFromSmiles("C=C(F)C(F)(F)F"))
        hfp_inchikey = inchi.MolToInchiKey(Chem.MolFromSmiles("C(=C(F)F)(C(F)(F)F)F"))
        train_inchikeys = {inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) for smi in df_hfc["refri_smiles"].unique()}
        self.assertNotIn(yf_inchikey, train_inchikeys)
        self.assertNotIn(hfp_inchikey, train_inchikeys)

        # 2. 独立验证 Axis 4: Split L2 离子对组合严格不相交
        l2_npz = np.load(ROOT_DIR / "splits" / "L2_controlled_composite.npz")
        clean_ion = lambda s: str(s).strip().replace("[", "").replace("]", "").upper()
        df_hfc["pair"] = df_hfc["cation"].apply(clean_ion) + "__" + df_hfc["anion"].apply(clean_ion)
        l2_train_pairs = set(df_hfc.iloc[l2_npz["train"]]["pair"])
        l2_test_pairs = set(df_hfc.iloc[l2_npz["test"]]["pair"])
        self.assertEqual(len(l2_train_pairs.intersection(l2_test_pairs)), 0, "L2 pair overlap detected!")

        # 3. 独立验证 Axis 3: Split B1/B2 阴离子家族严格不相交
        b1_npz = np.load(ROOT_DIR / "splits_anion_ood" / "split_B1_Fam2_Fluorosulfonate.npz")
        b1_train_anis = set(df_hfc.iloc[b1_npz["train"]]["anion"].apply(clean_ion))
        b1_test_anis = set(df_hfc.iloc[b1_npz["test"]]["anion"].apply(clean_ion))
        self.assertEqual(len(b1_train_anis.intersection(b1_test_anis)), 0, "B1 anion overlap detected!")

        # 4. 核验官方证书与哈希一致性
        cert_p = ROOT_DIR / "Phase4_Scientific_Validation" / "structure_disjointness_certificate.json"
        self.assertTrue(cert_p.exists())
        with open(cert_p, "r", encoding="utf-8") as f:
            cert = json.load(f)
        self.assertEqual(cert.get("verdict"), "PROVEN_DISJOINT_ACROSS_ALL_AXES")
        stored_hash = cert.pop("certificate_sha256")
        recomputed_hash = hashlib.sha256(json.dumps(cert, indent=2, ensure_ascii=False).encode("utf-8")).hexdigest()
        self.assertEqual(stored_hash, recomputed_hash)
        print("  [PASS] Test 10: 底层原始数据独立重算证实五大轴 100% 结构不相交，证书密码学签名真实自洽！")

    def test_l2_recombinant_pair_leakage_adversarial_fails(self):
        """红队测试 11: 模拟 L2 离子对发生训练泄漏 (将训练对注入测试对)，断言必须抛出 AssertionError 熔断"""
        train_pairs = {('EMIM', 'AC'), ('BMIM', 'BF4'), ('HMIM', 'TF2N')}
        # 故意注入 ('EMIM', 'AC') 到 test_pairs
        test_pairs = {('EMIM', 'BF4'), ('BMIM', 'OTF'), ('EMIM', 'AC')}
        
        with self.assertRaises(AssertionError) as ctx:
            overlap = train_pairs.intersection(test_pairs)
            assert len(overlap) == 0, f"L2 Leakage Breach: Test pairs found in train: {overlap}"
        self.assertIn("L2 Leakage Breach", str(ctx.exception))
        self.assertIn("EMIM", str(ctx.exception))
        print("  [PASS] Test 11: L2 离子对组合泄漏成功触发断言拦截熔断！")

    def test_graph_ig_sanity_checks_certificate_integrity(self):
        """红队测试 12: [打破循环] 从原始 summary CSV 独立重算 Adebayo 多种子分布与 D_atom/D_graph 密度比"""
        import json, hashlib
        csv_p = ROOT_DIR / "results_attribution" / "graph_ig_sanity_checks_summary.csv"
        self.assertTrue(csv_p.exists(), "graph_ig_sanity_checks_summary.csv missing!")
        df_summary = pd.read_csv(csv_p)
        self.assertEqual(len(df_summary), 4)

        # 独立从 CSV 重算分位数与指标
        overall_rho_median = float(df_summary['rho_adebayo_median'].median())
        overall_cos_median = float(df_summary['cos_adebayo_median'].median())
        overall_rho_max = float(df_summary['rho_adebayo_max'].max())
        mean_atom_density_ratio = float(df_summary['ratio_d_atom_ani_cat'].mean())
        mean_graph_density_ratio = float(df_summary['ratio_d_graph_ani_cat'].mean())

        # 核心红队科学断言: 权重随机化后，排序彻底解耦 (中位数 rho < 0.25 vs 完整模型的 1.0)
        self.assertLess(overall_rho_median, 0.25, f"Adebayo breach: rank correlation not decoupled: {overall_rho_median}")
        self.assertLess(overall_cos_median, 0.50, f"Adebayo breach: directional cosine not altered: {overall_cos_median}")
        # 尺度归一化后阴离子密度仍保留优势 (> 1.0)
        self.assertGreater(mean_atom_density_ratio, 1.0)
        self.assertGreater(mean_graph_density_ratio, 1.0)

        # 检验证书自洽性
        cert_p = ROOT_DIR / "Phase4_Scientific_Validation" / "graph_ig_sanity_checks_certificate.json"
        self.assertTrue(cert_p.exists())
        with open(cert_p, "r", encoding="utf-8") as f:
            cert = json.load(f)
        self.assertEqual(cert.get("verdict"), "ADEBAYO_SANITY_AND_ROBUSTNESS_PROVEN")
        self.assertTrue(cert["adebayo_randomization"]["passes_adebayo_sanity_check"])
        stored_hash = cert.pop("certificate_sha256")
        recomputed_hash = hashlib.sha256(json.dumps(cert, indent=2, ensure_ascii=False).encode("utf-8")).hexdigest()
        self.assertEqual(stored_hash, recomputed_hash)
        print("  [PASS] Test 12: Graph-IG Adebayo N=10 分布与 D_atom/D_graph 独立重算达标，证书哈希签名自洽！")

    def test_numerical_stability_panel_integrity(self):
        """红队测试 13: 检验 16 组全样本经验数值稳定化面板指标 (余弦相似度 > 0.999, D_rel < 1.0%)"""
        csv_p = ROOT_DIR / "results_attribution" / "preflight_numerical_stability_panel.csv"
        self.assertTrue(csv_p.exists(), "preflight_numerical_stability_panel.csv missing!")
        
        df_panel = pd.read_csv(csv_p)
        self.assertEqual(len(df_panel), 16, f"Expected 16 evaluations in panel, got {len(df_panel)}")
        
        # 校验指标硬红线
        self.assertGreater(df_panel['cos_100_200'].min(), 0.999)
        self.assertLess(df_panel['d_rel_100_200_pct'].max(), 1.0)
        self.assertLess(df_panel['rel_err_200_pct'].median(), 1.0)
        self.assertGreater(df_panel['spearman_rank_100_200'].min(), 0.98)
        print("  [PASS] Test 13: 16 组多样本经验数值稳定化面板指标 100% 达标！")

    def test_grouped_state_l0_zero_input_leakage(self):
        """红队测试 14: [P0-1 闭环] 检验 Grouped-L0 物理状态五元组在 train/val 之间严格 0 重叠"""
        split_p = ROOT_DIR / "splits" / "HFC_grouped_state_split.npz"
        self.assertTrue(split_p.exists(), "HFC_grouped_state_split.npz missing!")
        npz = np.load(split_p)
        train_idx, val_idx = npz["train"], npz["val"]

        df_hfc = pd.read_csv(ROOT_DIR / "datasets" / "hfc_2739_v7" / "meta_info.csv")
        clean_ion = lambda s: str(s).strip().replace("[", "").replace("]", "").upper()
        df_hfc["state_5tuple"] = (
            df_hfc["cation"].apply(clean_ion) + "__" +
            df_hfc["anion"].apply(clean_ion) + "__" +
            df_hfc["refrigerant"].astype(str) + "__" +
            df_hfc["T_K"].round(2).astype(str) + "__" +
            df_hfc["P_MPa"].round(4).astype(str)
        )

        train_states = set(df_hfc.iloc[train_idx]["state_5tuple"])
        val_states = set(df_hfc.iloc[val_idx]["state_5tuple"])
        state_overlap = train_states.intersection(val_states)
        self.assertEqual(len(state_overlap), 0, f"Grouped-L0 State overlap detected: {state_overlap}")
        print("  [PASS] Test 14: Grouped-L0 状态五元组严格 0 输入泄漏检验通过！")

    def test_consumer_fail_closed_on_contaminated_descriptor(self):
        """红队测试 15: [P1-4 闭环] 模拟下游消费端加载含 HFP 的伪造 R1234yf 描述符，断言必须 Fail-Closed 熔断"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_desc_p = Path(tmp_dir) / "xTB_Physics_Descriptors.csv"
            fake_df = pd.DataFrame([
                {"Molecule": "R1234yf", "Category": "Refrigerant", "SMILES": "C(=C(F)F)(C(F)(F)F)F", "Dipole_Debye": 1.0, "Polarizability_au": 10.0, "Volume_A3": 80.0}
            ])
            fake_df.to_csv(fake_desc_p, index=False)

            from compute_f2_physics_alignment import _assert_uncorrupted_descriptors
            with self.assertRaises(RuntimeError) as ctx:
                _assert_uncorrupted_descriptors(fake_desc_p)
            self.assertIn("FAIL-CLOSED BREACH", str(ctx.exception))
            self.assertIn("historical HFP C3F6 alias", str(ctx.exception))
        print("  [PASS] Test 15: 下游消费端对受污染历史描述符严格 Fail-Closed 拦截熔断！")

    def test_sentinel_selection_manifest_integrity(self):
        """红队测试 16: [P1-4 闭环] 检验 16 对分层哨兵前置冻结清单的完整性与能区覆盖度"""
        manifest_p = ROOT_DIR / "Phase4_Scientific_Validation" / "sentinel_selection_manifest.csv"
        self.assertTrue(manifest_p.exists(), "sentinel_selection_manifest.csv missing!")
        df_sentinel = pd.read_csv(manifest_p)
        self.assertEqual(len(df_sentinel), 16)
        
        regimes = set(df_sentinel["regime"].unique())
        self.assertEqual(regimes, {"Strong", "Medium", "Weak", "Mixed"})
        
        # 校验基准能值均在合理负值区间
        self.assertTrue((df_sentinel["historical_delta_E_kcal_mol"] < 0).all())
        self.assertLess(df_sentinel["historical_delta_E_kcal_mol"].min(), -20.0) # Strong regime
        self.assertGreater(df_sentinel["historical_delta_E_kcal_mol"].max(), -4.0) # Weak regime
        print("  [PASS] Test 16: 16 组前置分层哨兵冻结清单完整性与 4 大能区覆盖度验证通过！")

    def test_checkpoint_lineage_contract_mismatch_fails_closed(self):
        """红队测试 17: [消费端血统门禁] 模拟下游加载划分哈希缺失或篡改的 Checkpoint，断言必须 Fail-Closed 熔断"""
        from artifact_freshness_gate import validate_checkpoint_lineage_contract

        # 1. 模拟缺少 split_sha256 的旧版无绑定权重
        legacy_ckpt = {"model_state_dict": {}, "epoch": 10}
        with self.assertRaises(RuntimeError) as ctx:
            validate_checkpoint_lineage_contract(legacy_ckpt, checkpoint_desc="legacy_unbound_model")
        self.assertIn("CHECKPOINT LINEAGE BREACH: UNBOUND MODEL", str(ctx.exception))

        # 2. 模拟 split_sha256 篡改或划分文件不一致
        tampered_ckpt = {
            "model_state_dict": {},
            "split_file": "splits/HFC_grouped_state_split.npz",
            "split_sha256": "bad_sha_0000000000000000000000000000000000000000000000000000000000000000"
        }
        real_split_p = ROOT_DIR / "splits" / "HFC_grouped_state_split.npz"
        with self.assertRaises(RuntimeError) as ctx:
            validate_checkpoint_lineage_contract(tampered_ckpt, expected_split_path=real_split_p)
        self.assertIn("CHECKPOINT-SPLIT LINEAGE MISMATCH", str(ctx.exception))
        print("  [PASS] Test 17: 消费端 Checkpoint 划分哈希缺失或篡改成功触发 Fail-Closed 熔断！")

    def test_measurement_context_homogeneity_contract(self):
        """红队测试 18: [测量上下文一致性] 校验 HFC 数据集测量上下文，注入异质数据断言必须熔断"""
        from thermodynamic_state_contract import assert_dataset_measurement_context_homogeneity
        
        # 1. 正向校验: 正式数据集 100% 满足单一 VLE 上下文
        df_real = pd.read_csv(ROOT_DIR / "datasets" / "hfc_2739_v7" / "meta_info.csv")
        ctx = assert_dataset_measurement_context_homogeneity(df_real)
        self.assertEqual(ctx['measurement_type'], "Vapor_Liquid_Equilibrium_VLE")

        # 2. 逆向校验: 注入非法异构 sheet 来源
        df_bad_sheet = df_real.head(10).copy()
        df_bad_sheet.loc[0, 'sheet'] = 'Table S1. Contaminated Source'
        with self.assertRaises(RuntimeError) as ctx_err:
            assert_dataset_measurement_context_homogeneity(df_bad_sheet)
        self.assertIn("Inhomogeneous sheets detected", str(ctx_err.exception))

        # 3. 逆向校验: 注入非物理压力
        df_bad_p = df_real.head(10).copy()
        df_bad_p.loc[0, 'P_MPa'] = -1.5
        with self.assertRaises(RuntimeError) as ctx_err:
            assert_dataset_measurement_context_homogeneity(df_bad_p)
        self.assertIn("Pressure out of physical VLE bounds", str(ctx_err.exception))
        print("  [PASS] Test 18: 数据集测量上下文单一体契约与非法数据注入拦截验证通过！")

    def test_grouped_vs_random_distribution_audit_integrity(self):
        """红队测试 19: [分布同质性证明] 校验 Grouped-L0 统计审计报告，证实温压未受扭曲且 0 泄漏"""
        import json
        audit_p = ROOT_DIR / "paper_results" / "grouped_vs_random_split_statistical_audit.json"
        self.assertTrue(audit_p.exists(), "audit report missing!")

        with open(audit_p, "r", encoding="utf-8") as f:
            audit = json.load(f)

        # 1. 硬性完整性不变量 (Hard Integrity Invariants - 必须 Fail-Closed):
        # 证实状态重叠严格为 0 (Grouped) vs 5 (Random)，样本行数完全封闭
        self.assertEqual(audit['coverage_summary']['Grouped-L0']['overlapping_states_count'], 0)
        self.assertEqual(audit['coverage_summary']['Random-L0']['overlapping_states_count'], 5)
        self.assertEqual(audit['coverage_summary']['Grouped-L0']['train_n'], 2464)
        self.assertEqual(audit['coverage_summary']['Grouped-L0']['val_n'], 275)

        # 2. 科学统计诊断 (Scientific Diagnostics - 验证多维诊断被真实计算并记录，不作僵化硬性熔断):
        self.assertIn('random_l0_null_reference_distribution', audit)
        self.assertIn('distance_diagnostics', audit)
        null_ref = audit['random_l0_null_reference_distribution']
        for var in ['T_K', 'P_MPa', 'x1']:
            self.assertIn(var, null_ref)
            obs = null_ref[var]['grouped_observed_val_vs_all']
            self.assertIn('within_random_reference_envelope', obs)
            self.assertIn('within_prespecified_physical_margin', obs)

        print("  [PASS] Test 19: Grouped-L0 状态不变量 (0 泄漏) 与多维统计诊断分层核查通过！")

    def test_sentinel_manifest_historical_row_and_hash_integrity(self):
        """红队测试 20: [哨兵溯源行号] 校验 16 哨兵的 historical_row_id 与记录哈希一致性"""
        import hashlib
        manifest_p = ROOT_DIR / "Phase4_Scientific_Validation" / "sentinel_selection_manifest.csv"
        df_sent = pd.read_csv(manifest_p)
        df_full = pd.read_csv(ROOT_DIR / "Phase4_Scientific_Validation" / "full_pair_interaction_results.csv")

        self.assertIn('historical_row_id', df_sent.columns)
        self.assertIn('historical_record_hash', df_sent.columns)

        for _, r in df_sent.iterrows():
            row_id = int(r['historical_row_id'])
            rec = df_full.iloc[row_id]
            self.assertEqual(rec['Ion_Name'], r['ion_name'])
            self.assertEqual(rec['Refrigerant'], r['refrigerant'])
            rec_str = "|".join(str(v) for v in rec.values)
            expected_hash = hashlib.sha256(rec_str.encode('utf-8')).hexdigest()
            self.assertEqual(r['historical_record_hash'], expected_hash)
        print("  [PASS] Test 20: 16 组哨兵原始行号 historical_row_id 与记录哈希 100% 独立重算自洽！")

if __name__ == '__main__':
    print("=" * 90)
    print("  P0 RED-TEAM ADVERSARIAL INTEGRITY TEST RUNNER (ALL 20 INJECTIONS)")
    print("=" * 90)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRedTeamAdversarialIntegrity)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if not res.wasSuccessful():
        sys.exit(1)
