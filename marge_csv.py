import pandas as pd
import os

# CSVファイルが格納されているディレクトリ
directory_path = './csv' 

# すべてのCSVファイルをリスト
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# データフレームのリスト
df_list = []

# 各CSVファイルを読み込み
for csv_file in csv_files:
    file_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(file_path)
    df_list.append(df)

# データフレームを結合
combined_df = pd.concat(df_list, ignore_index=True)

# 結合したデータフレームを新しいCSVファイルに書き出し
combined_df.to_csv('combined_output.csv', index=False, encoding='utf-8')

print('All CSV files have been combined into combined_output.csv')
