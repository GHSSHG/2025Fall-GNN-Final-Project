import pandas as pd
import os

def main():
    # 先定义路径
    input_path = os.path.join('data', 'raw_data.tsv')
    output_path = os.path.join('data', 'data.tsv')

    if not os.path.exists(input_path):
        print(f"错误：在 {input_path} 未找到输入文件。")
        return

    print(f"正在读取文件: {input_path} ...")
    
    try:
        # 读取 TSV 文件
        df = pd.read_csv(input_path, sep='\t')
        
        # 检查必要的列是否存在
        required_source_cols = ['CDR3', 'Epitope', 'Score']
        missing_cols = [col for col in required_source_cols if col not in df.columns]
        if missing_cols:
            print(f"错误：源数据中缺少以下列: {missing_cols}")
            return

        # 生成 label 列
        # 逻辑：如果 'Score' == 0，则 label=0，否则 label=1
        # 使用 lambda 函数处理每一行
        df['label'] = df['Score'].apply(lambda x: 0 if x == 0 else 1)

        # 只保留需要的列 (CDR3, Epitope, label)
        final_columns = ['CDR3', 'Epitope', 'label']
        df_subset = df[final_columns]
        
        print(f"原始数据行数: {len(df_subset)}")

        # 去重
        df_clean = df_subset.drop_duplicates()
        
        print(f"去重后行数: {len(df_clean)}")

        # 保存结果
        df_clean.to_csv(output_path, sep='\t', index=False)
        print(f"处理完成！结果已保存至: {output_path}")
        # 打印前几行看看结果
        print("前5行预览:")
        print(df_clean.head())

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()