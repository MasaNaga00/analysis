import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans']

# ===== 1. データ前処理 =====
# K480機種のデータを抽出
df_k480 = df_parts[df_parts['Model'] == 'K480'].copy()

print(f"K480機種の修理データ件数: {len(df_k480)}")
print(f"製造月の範囲: {df_k480['prod_month'].min()} - {df_k480['prod_month'].max()}")
print(f"修理入庫日の範囲: {df_k480['date'].min()} - {df_k480['date'].max()}")

# 製品寿命を計算（月単位）
df_k480['prod_month_start'] = df_k480['prod_month'].dt.start_time
df_k480['lifetime_days'] = (df_k480['date'] - df_k480['prod_month_start']).dt.days
df_k480['lifetime_months'] = df_k480['lifetime_days'] / 30.44  # 平均月日数

print(f"\n製品寿命統計:")
print(df_k480['lifetime_months'].describe())

# ===== 2. 製造月別寿命分析 =====
print("\n===== 製造月別製品寿命分析 =====")

# 製造月別寿命の統計
lifetime_by_month = df_k480.groupby('prod_month')['lifetime_months'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(2)
print("\n製造月別寿命統計:")
print(lifetime_by_month)

# 製造月別寿命の箱ひげ図
plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)
df_k480.boxplot(column='lifetime_months', by='prod_month', ax=plt.gca())
plt.title('製造月別製品寿命分布')
plt.xlabel('製造月')
plt.ylabel('寿命（月）')
plt.xticks(rotation=45)

# 製造月別寿命の平均値プロット
plt.subplot(2, 2, 2)
monthly_mean = df_k480.groupby('prod_month')['lifetime_months'].mean()
plt.plot(monthly_mean.index.astype(str), monthly_mean.values, 'o-')
plt.title('製造月別平均寿命推移')
plt.xlabel('製造月')
plt.ylabel('平均寿命（月）')
plt.xticks(rotation=45)
plt.grid(True)

# 全体平均からの乖離
overall_mean = df_k480['lifetime_months'].mean()
plt.axhline(y=overall_mean, color='r', linestyle='--', label=f'全体平均({overall_mean:.1f}月)')
plt.legend()

plt.tight_layout()
plt.show()

# ===== 3. 統計的検定 =====
print("\n===== 統計的検定 =====")

# Kruskal-Wallis検定（製造月間の寿命分布の差）
month_groups = [group['lifetime_months'].values for name, group in df_k480.groupby('prod_month')]
kruskal_stat, kruskal_p = kruskal(*month_groups)
print(f"Kruskal-Wallis検定:")
print(f"  統計量: {kruskal_stat:.4f}")
print(f"  p値: {kruskal_p:.6f}")
print(f"  結果: {'有意差あり' if kruskal_p < 0.05 else '有意差なし'}")

# 各製造月と全体平均の比較（Z検定）
print(f"\n製造月別異常検知（全体平均 {overall_mean:.2f}月 との比較）:")
overall_std = df_k480['lifetime_months'].std()

for month, group in df_k480.groupby('prod_month'):
    n = len(group)
    if n >= 5:  # サンプルサイズが小さすぎる場合は除外
        month_mean = group['lifetime_months'].mean()
        # Z統計量の計算
        z_stat = (month_mean - overall_mean) / (overall_std / np.sqrt(n))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # 両側検定
        
        status = "異常" if p_value < 0.05 and month_mean < overall_mean else "正常"
        print(f"  {month}: 平均{month_mean:.2f}月 (n={n}) Z={z_stat:.2f} p={p_value:.4f} [{status}]")

# ===== 4. 部品別故障パターン分析 =====
print("\n===== 部品別故障パターン分析 =====")

# 製造月×部品のクロス集計
cross_tab = pd.crosstab(df_k480['prod_month'], df_k480['parts_no'])
print("\n製造月×交換部品クロス集計:")
print(cross_tab)

# 部品別故障率の時系列プロット
plt.figure(figsize=(15, 10))

# 主要部品（故障件数上位5位）を抽出
top_parts = df_k480['parts_no'].value_counts().head(5)
print(f"\n主要故障部品Top5:")
print(top_parts)

# 製造月別の各部品故障率
plt.subplot(2, 2, 3)
for part in top_parts.index:
    part_data = df_k480[df_k480['parts_no'] == part]
    monthly_count = part_data.groupby('prod_month').size()
    total_monthly = df_k480.groupby('prod_month').size()
    part_rate = (monthly_count / total_monthly * 100).fillna(0)
    
    plt.plot(part_rate.index.astype(str), part_rate.values, 'o-', label=part)

plt.title('製造月別主要部品故障率推移')
plt.xlabel('製造月')
plt.ylabel('故障率（%）')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# 部品×製造月のヒートマップ
plt.subplot(2, 2, 4)
# 上位部品のみでヒートマップ作成
top_parts_data = cross_tab[top_parts.index]
sns.heatmap(top_parts_data.T, annot=True, fmt='d', cmap='Reds')
plt.title('製造月×主要部品 故障件数')
plt.xlabel('製造月')
plt.ylabel('部品番号')

plt.tight_layout()
plt.show()

# カイ二乗検定（製造月と部品の独立性）
chi2_stat, chi2_p, dof, expected = chi2_contingency(cross_tab)
print(f"\nカイ二乗検定（製造月と交換部品の独立性）:")
print(f"  統計量: {chi2_stat:.4f}")
print(f"  p値: {chi2_p:.6f}")
print(f"  結果: {'製造月と部品に関連あり' if chi2_p < 0.05 else '製造月と部品は独立'}")

# ===== 5. 地域要因の分析 =====
print("\n===== 地域要因分析 =====")

# 地域別寿命統計
region_stats = df_k480.groupby('Aria')['lifetime_months'].agg([
    'count', 'mean', 'median', 'std'
]).round(2)
print("\n地域別寿命統計:")
print(region_stats)

# 地域×製造月の二元配置分散分析的分析
print(f"\n地域×製造月の組み合わせ分析:")
region_month_stats = df_k480.groupby(['Aria', 'prod_month'])['lifetime_months'].agg([
    'count', 'mean'
]).round(2)
print(region_month_stats)

# ===== 6. 異常製造月の最終判定 =====
print("\n===== 異常製造月の最終判定 =====")

# 複数指標による総合判定
monthly_summary = df_k480.groupby('prod_month').agg({
    'lifetime_months': ['count', 'mean', 'std'],
    'parts_no': lambda x: len(x.unique())  # 故障部品種類数
}).round(2)

monthly_summary.columns = ['件数', '平均寿命', '寿命標準偏差', '故障部品種類数']
monthly_summary['寿命偏差'] = (monthly_summary['平均寿命'] - overall_mean).round(2)
monthly_summary['異常度'] = np.where(
    (monthly_summary['平均寿命'] < overall_mean - overall_std) & 
    (monthly_summary['件数'] >= 5),
    '要注意', '正常'
)

print("\n製造月別総合評価:")
print(monthly_summary)

# 要注意製造月の詳細分析
warning_months = monthly_summary[monthly_summary['異常度'] == '要注意'].index
if len(warning_months) > 0:
    print(f"\n要注意製造月: {list(warning_months)}")
    for month in warning_months:
        month_data = df_k480[df_k480['prod_month'] == month]
        print(f"\n{month}の詳細:")
        print(f"  修理件数: {len(month_data)}")
        print(f"  平均寿命: {month_data['lifetime_months'].mean():.2f}月")
        print(f"  主要故障部品: {month_data['parts_no'].value_counts().head(3).to_dict()}")
        print(f"  地域分布: {month_data['Aria'].value_counts().to_dict()}")
else:
    print("\n統計的に有意な異常製造月は検出されませんでした。")

print("\n===== 分析完了 =====")
