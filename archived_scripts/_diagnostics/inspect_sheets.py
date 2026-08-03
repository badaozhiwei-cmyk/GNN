"""
检查 ZLJ_DATA.xlsx 的三个关键Sheet的结构
"""
import pandas as pd

excel_path = '../ZLJ_DATA.xlsx'
sheets_to_check = ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']

print("="*80)
print("检查 ZLJ_DATA.xlsx 中用于训练的三个 Sheet 的数据结构")
print("="*80)

total_rows = 0

for sheet_name in sheets_to_check:
    print(f"\n【{sheet_name}】")
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=2)
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        print(f"  列名: {list(df.columns)}")
        print(f"  前3行:")
        print(df.head(3).to_string())
        total_rows += len(df)
    except Exception as e:
        print(f"  ❌ 错误: {e}")

print(f"\n{'='*80}")
print(f"总行数: {total_rows}")
print(f"{'='*80}")
