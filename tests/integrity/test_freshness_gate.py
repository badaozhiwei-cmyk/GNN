import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# 适配 Windows 控制台编码
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "Phase4_Scientific_Validation"))

from artifact_freshness_gate import (
    validate_dataset_freshness,
    validate_bound_split_freshness,
    assert_clean_pipeline_environment
)

class TestArtifactFreshnessGate(unittest.TestCase):

    def test_canonical_datasets_pass(self):
        """测试 1: 真实的不可变数据集目录校验必须全部通过"""
        hfc_dir = ROOT_DIR / "datasets" / "hfc_2739_v7"
        full_dir = ROOT_DIR / "datasets" / "full_4444_v7"
        
        m_hfc = validate_dataset_freshness(hfc_dir, expected_dataset_id="HFC_2739_V7_CANONICAL")
        self.assertEqual(m_hfc['dataset_id'], "HFC_2739_V7_CANONICAL")
        
        m_full = validate_dataset_freshness(full_dir, expected_dataset_id="FULL_4444_V7_CANONICAL")
        self.assertEqual(m_full['dataset_id'], "FULL_4444_V7_CANONICAL")
        print("  [PASS] Test 1: Canonical dataset manifests verified.")

    def test_tampered_file_fails_closed(self):
        """测试 2: 模拟数据文件被篡改 (SHA256 改变)，必须 Fail-Closed 抛出 RuntimeError"""
        hfc_dir = ROOT_DIR / "datasets" / "hfc_2739_v7"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # 复制一份数据集
            for f in ['dataset_manifest.json', 'data.npy', 'label.npy', 'meta_info.csv']:
                shutil.copy2(hfc_dir / f, tmp_path / f)
            
            # 篡改 label.npy
            with open(tmp_path / 'label.npy', 'ab') as f:
                f.write(b"tampered_bytes_corruption")

            with self.assertRaises(RuntimeError) as ctx:
                validate_dataset_freshness(tmp_path)
            self.assertIn("DATASET TAMPERED OR STALE BREACH", str(ctx.exception))
            self.assertIn("SHA256 摘要不匹配", str(ctx.exception))
            print("  [PASS] Test 2: Tampered data file successfully intercepted by hash mismatch!")

    def test_missing_manifest_fails_closed(self):
        """测试 3: 缺少 manifest 契约锁的目录必须被拦截"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(RuntimeError) as ctx:
                validate_dataset_freshness(tmp_dir)
            self.assertIn("缺少 dataset_manifest.json 契约锁", str(ctx.exception))
            print("  [PASS] Test 3: Unmanifested directory fail-closed verified.")

    def test_stale_split_hash_binding_fails(self):
        """测试 4: 底层数据集更新导致 SHA256 改变时，旧划分契约必须自动失效"""
        bound_sp = ROOT_DIR / "splits" / "HFC_all_split_bound.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sp_copy = tmp_path / "split_bound.json"
            shutil.copy2(bound_sp, sp_copy)
            
            # 创建一个假冒的修改版数据集
            mock_ds = tmp_path / "mock_ds"
            mock_ds.mkdir()
            mock_manifest = mock_ds / "dataset_manifest.json"
            mock_manifest.write_text('{"dataset_id": "MODIFIED"}', encoding='utf-8')

            with self.assertRaises(RuntimeError) as ctx:
                validate_bound_split_freshness(sp_copy, dataset_dir=mock_ds)
            self.assertIn("SPLIT-DATASET BINDING BROKEN", str(ctx.exception))
            print("  [PASS] Test 4: Stale split hash binding successfully invalidated!")

if __name__ == '__main__':
    unittest.main()
