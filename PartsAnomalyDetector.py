import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import median_abs_deviation
import warnings
warnings.filterwarnings('ignore')
import matplotlib_fontja

# 日本語フォント設定
#plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
matplotlib_fontja.japanize()

class PartsAnomalyDetector:
    """
    部品異常検出クラス - Phase 1実装
    """
    
    def __init__(self, df_parts):
        """
        初期化
        Args:
            df_parts: 修理データのDataFrame
        """
        self.df_parts = df_parts.copy()
        self.results = {}
        self.prepare_data()
    
    def prepare_data(self):
        """
        データの前処理
        """
        print("=== データ前処理開始 ===")
        
        # 基本統計情報
        print(f"総レコード数: {len(self.df_parts):,}")
        print(f"ユニーク修理件数: {self.df_parts['IF_ID'].nunique():,}")
        print(f"機種数: {self.df_parts['Model'].nunique()}")
        print(f"部品種類数: {self.df_parts['parts_no'].nunique()}")
        
        # 月次データの準備
        self.df_parts['year_month'] = self.df_parts['date'].dt.to_period('M')
        
        # 部品使用数の集計（基本集計）
        self.parts_usage = self.df_parts.groupby(['Model', 'prod_month', 'Area', 'parts_no']).agg({
            'IF_ID': 'count',  # 部品使用回数
            'date': ['min', 'max']  # 最初と最後の修理日
        }).reset_index()
        
        self.parts_usage.columns = ['Model', 'prod_month', 'Area', 'parts_no', 'usage_count', 'first_repair', 'last_repair']
        
        # 修理件数の集計
        self.repair_counts = self.df_parts.groupby(['Model', 'prod_month', 'Area'])['IF_ID'].nunique().reset_index()
        self.repair_counts.columns = ['Model', 'prod_month', 'Area', 'repair_count']
        
        # 部品使用数と修理件数をマージ
        self.parts_usage = self.parts_usage.merge(self.repair_counts, on=['Model', 'prod_month', 'Area'])
        
        # 部品使用率の計算（修理件数あたりの部品使用数）
        self.parts_usage['usage_rate'] = self.parts_usage['usage_count'] / self.parts_usage['repair_count']
        
        print("=== データ前処理完了 ===\n")
    
    def detect_anomalies_mad_zscore(self, threshold=3.5):
        """
        修正Z-Score（MAD基準）による異常検出
        Args:
            threshold: 異常値判定閾値（デフォルト3.5）
        """
        print("=== 修正Z-Score（MAD基準）異常検出開始 ===")
        
        anomalies = []
        
        # 機種・部品ごとにグループ化して異常検出
        for (model, parts_no), group in self.parts_usage.groupby(['Model', 'parts_no']):
            if len(group) < 3:  # データが少なすぎる場合はスキップ
                continue
            
            usage_rates = group['usage_rate'].values
            
            # MAD（中央絶対偏差）計算
            median_usage = np.median(usage_rates)
            mad = median_abs_deviation(usage_rates)
            
            if mad == 0:  # MADが0の場合（全て同じ値）
                continue
            
            # 修正Z-Score計算
            modified_z_scores = 0.6745 * (usage_rates - median_usage) / mad
            
            # 異常値の検出
            anomaly_mask = np.abs(modified_z_scores) > threshold
            
            if np.any(anomaly_mask):
                group_with_scores = group.copy()
                group_with_scores['modified_z_score'] = modified_z_scores
                group_with_scores['is_anomaly_mad'] = anomaly_mask
                group_with_scores['median_usage_rate'] = median_usage
                group_with_scores['mad_value'] = mad
                
                anomalies.append(group_with_scores[anomaly_mask])
        
        if anomalies:
            self.results['mad_anomalies'] = pd.concat(anomalies, ignore_index=True)
            print(f"MAD基準で検出された異常データ数: {len(self.results['mad_anomalies'])}")
        else:
            self.results['mad_anomalies'] = pd.DataFrame()
            print("MAD基準で異常データは検出されませんでした")
        
        print("=== 修正Z-Score（MAD基準）異常検出完了 ===\n")
    
    def detect_relative_anomalies(self, threshold_percentile=95):
        """
        製造年月・地域別相対比較異常検出
        Args:
            threshold_percentile: 異常値判定パーセンタイル（デフォルト95%）
        """
        print("=== 相対比較異常検出開始 ===")
        
        relative_anomalies = []
        
        # 機種・部品ごとに相対比較
        for (model, parts_no), group in self.parts_usage.groupby(['Model', 'parts_no']):
            if len(group) < 5:  # データが少なすぎる場合はスキップ
                continue
            
            # パーセンタイル基準での異常検出
            threshold_value = np.percentile(group['usage_rate'], threshold_percentile)
            anomaly_mask = group['usage_rate'] > threshold_value
            
            if np.any(anomaly_mask):
                group_with_stats = group.copy()
                group_with_stats['percentile_threshold'] = threshold_value
                group_with_stats['is_anomaly_relative'] = anomaly_mask
                group_with_stats['relative_rank'] = group['usage_rate'].rank(pct=True)
                
                relative_anomalies.append(group_with_stats[anomaly_mask])
        
        if relative_anomalies:
            self.results['relative_anomalies'] = pd.concat(relative_anomalies, ignore_index=True)
            print(f"相対比較で検出された異常データ数: {len(self.results['relative_anomalies'])}")
        else:
            self.results['relative_anomalies'] = pd.DataFrame()
            print("相対比較で異常データは検出されませんでした")
        
        print("=== 相対比較異常検出完了 ===\n")
    
    def detect_trend_anomalies(self, window_size=3):
        """
        時系列トレンド異常検出
        Args:
            window_size: 移動平均の窓サイズ（月）
        """
        print("=== 時系列トレンド異常検出開始 ===")
        
        # 月次集計データの準備
        monthly_usage = self.df_parts.groupby(['Model', 'parts_no', 'year_month']).agg({
            'IF_ID': 'count'
        }).reset_index()
        monthly_usage.columns = ['Model', 'parts_no', 'year_month', 'monthly_usage']
        
        trend_anomalies = []
        
        # 機種・部品ごとに時系列分析
        for (model, parts_no), group in monthly_usage.groupby(['Model', 'parts_no']):
            if len(group) < window_size * 2:  # 窓サイズの2倍以上のデータが必要
                continue
            
            # 時系列順にソート
            group = group.sort_values('year_month').reset_index(drop=True)
            
            # Modelカラムを追加（後で使用するため）
            group['Model'] = model
            
            # 移動平均計算
            group['moving_avg'] = group['monthly_usage'].rolling(window=window_size, center=True).mean()
            
            # 変化率計算
            group['change_rate'] = group['monthly_usage'].pct_change()
            
            # 急激な変化の検出（変化率の標準偏差の2倍以上）
            if len(group['change_rate'].dropna()) > 2:
                change_std = group['change_rate'].std()
                change_mean = group['change_rate'].mean()
                
                # 異常な変化率を検出
                anomaly_mask = np.abs(group['change_rate'] - change_mean) > (2 * change_std)
                
                if np.any(anomaly_mask):
                    group['is_anomaly_trend'] = anomaly_mask
                    group['change_threshold'] = 2 * change_std
                    trend_anomalies.append(group[anomaly_mask])
        
        if trend_anomalies:
            self.results['trend_anomalies'] = pd.concat(trend_anomalies, ignore_index=True)
            print(f"トレンド分析で検出された異常データ数: {len(self.results['trend_anomalies'])}")
        else:
            self.results['trend_anomalies'] = pd.DataFrame()
            print("トレンド分析で異常データは検出されませんでした")
        
        print("=== 時系列トレンド異常検出完了 ===\n")
    
    def run_all_detections(self, mad_threshold=3.5, percentile_threshold=95, trend_window=3):
        """
        全ての異常検出を実行
        """
        print("=== Phase 1 異常検出開始 ===\n")
        
        self.detect_anomalies_mad_zscore(mad_threshold)
        self.detect_relative_anomalies(percentile_threshold)
        self.detect_trend_anomalies(trend_window)
        
        # 結果サマリー
        self.generate_summary()
        
        print("=== Phase 1 異常検出完了 ===\n")
    
    def generate_summary(self):
        """
        検出結果のサマリー生成
        """
        print("=== 異常検出結果サマリー ===")
        
        summary = {
            'MAD異常検出': len(self.results.get('mad_anomalies', [])),
            '相対比較異常検出': len(self.results.get('relative_anomalies', [])),
            'トレンド異常検出': len(self.results.get('trend_anomalies', []))
        }
        
        for method, count in summary.items():
            print(f"{method}: {count}件")
        
        # 機種別異常数
        if not self.results.get('mad_anomalies', pd.DataFrame()).empty:
            print("\n=== 機種別異常数（MAD基準） ===")
            model_counts = self.results['mad_anomalies']['Model'].value_counts().head(10)
            for model, count in model_counts.items():
                print(f"{model}: {count}件")
        
        print("=" * 30 + "\n")
    
    def get_model_anomalies(self, model_name):
        """
        指定機種の異常データを取得
        Args:
            model_name: 機種名
        Returns:
            dict: 各手法での異常検出結果
        """
        model_results = {}
        
        for method, data in self.results.items():
            if not data.empty and 'Model' in data.columns:
                model_data = data[data['Model'] == model_name]
                if not model_data.empty:
                    model_results[method] = model_data
        
        return model_results

# 使用例
if __name__ == "__main__":
    # サンプルデータでの実行例
    # detector = PartsAnomalyDetector(df_parts)
    # detector.run_all_detections()
    # 
    # # 特定機種の結果取得
    # model_results = detector.get_model_anomalies('Camera_Model_A')
    pass