import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# シード設定（再現可能な結果のため）
np.random.seed(42)
random.seed(42)

def generate_repair_test_data(n_repairs=1000):
    """
    修理データのテストデータを生成する関数
    
    Parameters:
    n_repairs: 修理件数（IF_IDのユニーク数）
    
    Returns:
    DataFrame: 修理データ
    """
    
    # 基本設定
    models = ['M100', 'M150', 'M200', 'M250']
    areas = ['JP', 'CN', 'USA', 'EUR']
    
    # 機種別の部品番号リスト（機種間で重複なし）
    parts_by_model = {
        'M100': [f'10{str(i).zfill(6)}' for i in range(1, 21)],  # 10000001-10000020
        'M150': [f'15{str(i).zfill(6)}' for i in range(1, 21)],  # 15000001-15000020
        'M200': [f'20{str(i).zfill(6)}' for i in range(1, 21)],  # 20000001-20000020
        'M250': [f'25{str(i).zfill(6)}' for i in range(1, 21)]   # 25000001-25000020
    }
    
    # 機種別の製造開始年月（HEADが'01'の時の年月）
    model_start_dates = {
        'M100': '2023-02',  # HEAD'01' = 2023/2
        'M150': '2022-06',  # HEAD'01' = 2022/6
        'M200': '2021-10',  # HEAD'01' = 2021/10
        'M250': '2020-04'   # HEAD'01' = 2020/4
    }
    
    def get_prod_month_from_head(model, head_num):
        """HEADから製造年月を計算"""
        start_date = pd.to_datetime(model_start_dates[model])
        months_to_add = head_num - 1
        prod_month = start_date + pd.DateOffset(months=months_to_add)
        return prod_month.to_period('M')
    
    def get_head_from_prod_month(model, prod_month):
        """製造年月からHEADを計算"""
        start_date = pd.to_datetime(model_start_dates[model]).to_period('M')
        months_diff = (prod_month.year - start_date.year) * 12 + (prod_month.month - start_date.month)
        return f"{months_diff + 1:02d}"
    
    # データ生成
    data = []
    
    for i in range(n_repairs):
        # 修理入庫日を生成（2023/1から2025/5まで）
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 5, 31)
        random_days = random.randint(0, (end_date - start_date).days)
        repair_date = start_date + timedelta(days=random_days)
        
        # 機種と地域をランダム選択
        model = random.choice(models)
        area = random.choice(areas)
        
        # IF_IDを生成（8桁の数字）
        if_id = f"{10000000 + i:08d}"
        
        # 製造年月を修理入庫日より前に設定
        # 修理入庫日から6ヶ月～36ヶ月前の範囲で製造年月を設定
        months_before = random.randint(6, 36)
        prod_month = (pd.to_datetime(repair_date) - pd.DateOffset(months=months_before)).to_period('M')
        
        # HEADを製造年月から計算
        head = get_head_from_prod_month(model, prod_month)
        
        # この修理で使用する部品数（1～5個）
        num_parts = random.randint(1, 5)
        selected_parts = random.sample(parts_by_model[model], num_parts)
        
        # 各部品に対して行を作成
        for part_no in selected_parts:
            data.append({
                'date': repair_date,
                'prod_month': prod_month,
                'HEAD': head,
                'Model': model,
                'Area': area,
                'parts_no': part_no,
                'IF_ID': if_id
            })
    
    # DataFrameを作成
    df = pd.DataFrame(data)
    
    # データ型を設定
    df['date'] = pd.to_datetime(df['date'])
    df['prod_month'] = df['prod_month'].astype('period[M]')
    df['HEAD'] = df['HEAD'].astype('object')
    df['Model'] = df['Model'].astype('object')
    df['Area'] = df['Area'].astype('object')
    df['parts_no'] = df['parts_no'].astype('object')
    df['IF_ID'] = df['IF_ID'].astype('object')
    
    # dateでソート
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

# テストデータ生成
test_data = generate_repair_test_data(n_repairs=1000)
test_data.to_pickle('test_data.pkl')

# データ確認
print("生成されたテストデータの概要:")
print(f"総行数: {len(test_data)}")
print(f"修理件数（IF_IDのユニーク数）: {test_data['IF_ID'].nunique()}")
print(f"期間: {test_data['date'].min().strftime('%Y/%m')} ～ {test_data['date'].max().strftime('%Y/%m')}")
print()

print("データ型:")
print(test_data.dtypes)
print()

print("最初の10行:")
print(test_data.head(10))
print()

print("各列のユニーク値数:")
for col in test_data.columns:
    print(f"{col}: {test_data[col].nunique()}")
print()

print("機種別の統計:")
print(test_data.groupby('Model').agg({
    'IF_ID': 'nunique',  # 修理件数
    'parts_no': 'count'  # 部品使用総数
}).rename(columns={'IF_ID': '修理件数', 'parts_no': '部品使用総数'}))
print()

print("1回の修理で使用する部品数の分布:")
parts_per_repair = test_data.groupby('IF_ID').size()
print(parts_per_repair.value_counts().sort_index())
print()

print("HEADと製造年月の対応例（M100）:")
m100_data = test_data[test_data['Model'] == 'M100'][['HEAD', 'prod_month']].drop_duplicates().sort_values('HEAD')
print(m100_data.head(10))