# step1_clean
# 清洗数据集，过滤非法 SMILES，去盐

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

#关闭 RDKit 的日志输出，防刷屏
RDLogger.DisableLog('rdApp.*')

def main():
    print("="*40)
    print("开始数据清洗流程...")
    print("="*40)

    #读取原始数据
    input_file = "trail_work/01_my_work/Lipophilicity.csv"
    try:
        df = pd.read_csv(input_file)
        print(f"[1/5] 成功读取原始数据，总行数: {len(df)}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}，请检查路径。")
        return

    # 处理空值与格式转换
    # 过滤 exp 和 smiles 为空的行
    df = df.dropna(subset=['exp', 'smiles'])
    # 强制把 exp 列转为浮点数
    df['exp'] = pd.to_numeric(df['exp'], errors='coerce')
    df = df.dropna(subset=['exp'])
    print(f"[2/5] 去除空值及非数字格式后，行数: {len(df)}")

    #过滤非法 SMILES
    def is_valid_smiles(smiles):
        if not isinstance(smiles, str): return False
        return Chem.MolFromSmiles(smiles) is not None

    df = df[df['smiles'].apply(is_valid_smiles)]
    print(f"[3/5] 过滤非法 SMILES 后，行数: {len(df)}")

    #排雷与安全去盐
    #删掉故意埋雷的 HUGE 分子和超长分子
    df = df[~df['CMPD_CHEMBLID'].str.contains("HUGE", na=False)]
    df = df[df['smiles'].str.len() < 300]
    print(f"[4/5] 排除超长异常分子后，行数: {len(df)}")

    #安全去盐并规范化
    def safe_clean_smiles(smiles):
        parts = smiles.split('.')
        main_part = max(parts, key=len) # 保留最长片段
        mol = Chem.MolFromSmiles(main_part)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True) # 返回规范 SMILES
        return None

    df['std_smiles'] = df['smiles'].apply(safe_clean_smiles)

    #去重并保存
    df_final = df.dropna(subset=['std_smiles']).copy()
    df_final = df_final.drop_duplicates(subset=['std_smiles'], keep='first').reset_index(drop=True)
    print(f"[5/5] 去盐及去重后，最终有效行数: {len(df_final)}")

    output_file = "trail_work/01_my_work/s1_Lipophilicity_clean_final.csv"
    df_final.to_csv(output_file, index=False)
    print("="*40)
    print(f"清洗完成！文件已保存为: {output_file}")
    print("="*40)

if __name__ == "__main__":
    main()