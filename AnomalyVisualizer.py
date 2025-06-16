import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib_fontja
# 日本語フォント設定
#plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")
matplotlib_fontja.japanize()

class AnomalyVisualizer:
    """
    異常値可視化クラス
    """
    
    def __init__(self, detector):
        """
        初期化
        Args:
            detector: PartsAnomalyDetectorのインスタンス
        """
        self.detector = detector
        self.df_parts = detector.df_parts
        self.results = detector.results
    
    def plot_model_overview(self, model_name, figsize=(15, 10)):
        """
        指定機種の異常値概要を可視化
        Args:
            model_name: 機種名
            figsize: 図のサイズ
        """
        # 機種データの取得
        model_data = self.df_parts[self.df_parts['Model'] == model_name]
        model_results = self.detector.get_model_anomalies(model_name)
        
        if model_data.empty:
            print(f"機種 '{model_name}' のデータが見つかりません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'機種: {model_name} - 異常検出結果概要', fontsize=16, fontweight='bold')
        
        # 1. 部品使用数分布
        parts_usage_model = self.detector.parts_usage[
            self.detector.parts_usage['Model'] == model_name
        ]
        
        if not parts_usage_model.empty:
            ax1 = axes[0, 0]
            ax1.hist(parts_usage_model['usage_rate'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('部品使用率分布')
            ax1.set_xlabel('使用率（修理件数あたり）')
            ax1.set_ylabel('頻度')
            ax1.grid(True, alpha=0.3)
        
        # 2. 異常検出結果サマリー
        ax2 = axes[0, 1]
        anomaly_counts = []
        method_names = []
        
        for method, data in model_results.items():
            if not data.empty:
                anomaly_counts.append(len(data))
                method_names.append(method.replace('_anomalies', '').replace('_', ' ').title())
        
        if anomaly_counts:
            bars = ax2.bar(method_names, anomaly_counts, color=['red', 'orange', 'yellow'][:len(anomaly_counts)])
            ax2.set_title('異常検出数（手法別）')
            ax2.set_ylabel('異常検出数')
            ax2.tick_params(axis='x', rotation=45)
            
            # 数値をバーの上に表示
            for bar, count in zip(bars, anomaly_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom')
        
        # 3. 製造年月別異常数
        ax3 = axes[1, 0]
        if 'mad_anomalies' in model_results and not model_results['mad_anomalies'].empty:
            mad_data = model_results['mad_anomalies']
            prod_month_counts = mad_data.groupby('prod_month').size()
            
            if not prod_month_counts.empty:
                prod_month_counts.plot(kind='bar', ax=ax3, color='lightcoral')
                ax3.set_title('製造年月別異常数（MAD基準）')
                ax3.set_xlabel('製造年月')
                ax3.set_ylabel('異常数')
                ax3.tick_params(axis='x', rotation=45)
        
        # 4. 地域別異常数
        ax4 = axes[1, 1]
        if 'mad_anomalies' in model_results and not model_results['mad_anomalies'].empty:
            mad_data = model_results['mad_anomalies']
            area_counts = mad_data.groupby('Area').size()
            
            if not area_counts.empty:
                area_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%')
                ax4.set_title('地域別異常数（MAD基準）')
                ax4.set_ylabel('')
        
        plt.tight_layout()
        plt.show()
    
    def plot_parts_anomaly_heatmap(self, model_name, method='mad_anomalies', figsize=(12, 8)):
        """
        部品・製造年月別異常値ヒートマップ
        Args:
            model_name: 機種名
            method: 異常検出手法
            figsize: 図のサイズ
        """
        model_results = self.detector.get_model_anomalies(model_name)
        
        if method not in model_results:
            print(f"機種 '{model_name}' に {method} の結果がありません")
            return
        
        anomaly_data = model_results[method]
        
        # ピボットテーブル作成
        heatmap_data = anomaly_data.pivot_table(
            values='usage_rate', 
            index='parts_no', 
            columns='prod_month', 
            aggfunc='mean'
        )
        
        if heatmap_data.empty:
            print("ヒートマップ用のデータがありません")
            return
        
        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='Reds', 
                   cbar_kws={'label': '部品使用率'})
        plt.title(f'機種: {model_name} - 部品・製造年月別異常使用率 ({method})')
        plt.xlabel('製造年月')
        plt.ylabel('部品番号')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_anomalies(self, model_name, parts_no=None, figsize=(15, 8)):
        """
        時系列異常値プロット
        Args:
            model_name: 機種名
            parts_no: 部品番号（指定しない場合は全部品）
            figsize: 図のサイズ
        """
        # 月次データの準備
        model_data = self.df_parts[self.df_parts['Model'] == model_name]
        
        if parts_no:
            model_data = model_data[model_data['parts_no'] == parts_no]
            title_suffix = f" - 部品: {parts_no}"
        else:
            title_suffix = " - 全部品"
        
        monthly_data = model_data.groupby(['parts_no', 'year_month']).agg({
            'IF_ID': 'count'
        }).reset_index()
        monthly_data.columns = ['parts_no', 'year_month', 'usage_count']
        
        if monthly_data.empty:
            print("時系列データがありません")
            return
        
        # 異常データの取得
        trend_anomalies = self.results.get('trend_anomalies', pd.DataFrame())
        if not trend_anomalies.empty:
            trend_anomalies = trend_anomalies[trend_anomalies['Model'] == model_name]
            if parts_no:
                trend_anomalies = trend_anomalies[trend_anomalies['parts_no'] == parts_no]
        
        # プロット
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f'機種: {model_name}{title_suffix} - 時系列異常検出', fontsize=14, fontweight='bold')
        
        # 上段：時系列プロット
        ax1 = axes[0]
        
        # 部品ごとに時系列プロット
        for parts, group in monthly_data.groupby('parts_no'):
            if parts_no and parts != parts_no:
                continue
            
            group = group.sort_values('year_month')
            ax1.plot(group['year_month'].astype(str), group['usage_count'], 
                    marker='o', label=f'部品: {parts}', alpha=0.7)
            
            # 異常点をハイライト
            if not trend_anomalies.empty:
                anomaly_points = trend_anomalies[trend_anomalies['parts_no'] == parts]
                if not anomaly_points.empty:
                    ax1.scatter(anomaly_points['year_month'].astype(str), 
                              anomaly_points['monthly_usage'],
                              color='red', s=100, marker='x', label='異常点')
        
        ax1.set_title('月次部品使用数推移')
        ax1.set_xlabel('年月')
        ax1.set_ylabel('使用数')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 下段：変化率プロット
        ax2 = axes[1]
        
        for parts, group in monthly_data.groupby('parts_no'):
            if parts_no and parts != parts_no:
                continue
            
            group = group.sort_values('year_month')
            change_rate = group['usage_count'].pct_change()
            ax2.plot(group['year_month'].astype(str), change_rate, 
                    marker='o', label=f'部品: {parts}', alpha=0.7)
        
        ax2.set_title('月次変化率')
        ax2.set_xlabel('年月')
        ax2.set_ylabel('変化率')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_anomaly_scatter(self, model_name, x_axis='usage_rate', y_axis='repair_count', 
                           method='mad_anomalies', figsize=(12, 8)):
        """
        異常値散布図
        Args:
            model_name: 機種名
            x_axis: X軸の項目
            y_axis: Y軸の項目
            method: 異常検出手法
            figsize: 図のサイズ
        """
        # 機種データの取得
        model_usage = self.detector.parts_usage[
            self.detector.parts_usage['Model'] == model_name
        ]
        
        if model_usage.empty:
            print(f"機種 '{model_name}' のデータがありません")
            return
        
        # 異常データの取得
        model_results = self.detector.get_model_anomalies(model_name)
        anomaly_data = model_results.get(method, pd.DataFrame())
        
        plt.figure(figsize=figsize)
        
        # 正常データをプロット
        plt.scatter(model_usage[x_axis], model_usage[y_axis], 
                   alpha=0.6, c='blue', label='正常データ', s=50)
        
        # 異常データをプロット
        if not anomaly_data.empty:
            plt.scatter(anomaly_data[x_axis], anomaly_data[y_axis], 
                       alpha=0.8, c='red', label='異常データ', s=100, marker='x')
        
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'機種: {model_name} - 異常値散布図 ({method})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_anomaly_report(self, model_name, save_path=None):
        """
        異常検出レポート生成
        Args:
            model_name: 機種名
            save_path: 保存パス（指定しない場合は表示のみ）
        """
        model_results = self.detector.get_model_anomalies(model_name)
        
        if not model_results:
            print(f"機種 '{model_name}' の異常データがありません")
            return
        
        report = f"=== 異常検出レポート: {model_name} ===\n"
        report += f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for method, data in model_results.items():
            if data.empty:
                continue
            
            report += f"--- {method.replace('_', ' ').title()} ---\n"
            report += f"異常検出数: {len(data)}\n"
            
            # デバッグ用：データ構造の確認
            print(f"\nデバッグ - {method} のカラム: {list(data.columns)}")
            
            # 上位異常値（データ構造に応じて処理）
            if 'modified_z_score' in data.columns:
                # MAD異常検出の場合
                top_anomalies = data.nlargest(5, 'modified_z_score')[
                    ['parts_no', 'prod_month', 'Area', 'usage_rate', 'modified_z_score']
                ]
            elif 'relative_rank' in data.columns:
                # 相対比較異常検出の場合
                top_anomalies = data.nlargest(5, 'relative_rank')[
                    ['parts_no', 'prod_month', 'Area', 'usage_rate', 'relative_rank']
                ]
            elif 'change_rate' in data.columns:
                # トレンド異常検出の場合（usage_rateがない）
                available_cols = ['parts_no', 'year_month']
                if 'monthly_usage' in data.columns:
                    available_cols.append('monthly_usage')
                if 'change_rate' in data.columns:
                    available_cols.append('change_rate')
                
                top_anomalies = data.nlargest(5, 'monthly_usage')[available_cols] if 'monthly_usage' in data.columns else data.head(5)[available_cols]
            else:
                # その他の場合（利用可能なカラムを動的に選択）
                available_cols = ['parts_no']
                if 'prod_month' in data.columns:
                    available_cols.append('prod_month')
                if 'Area' in data.columns:
                    available_cols.append('Area')
                if 'usage_rate' in data.columns:
                    available_cols.append('usage_rate')
                    top_anomalies = data.nlargest(5, 'usage_rate')[available_cols]
                elif 'monthly_usage' in data.columns:
                    available_cols.append('monthly_usage')
                    top_anomalies = data.nlargest(5, 'monthly_usage')[available_cols]
                else:
                    top_anomalies = data.head(5)[available_cols]
            
            report += "上位異常値:\n"
            report += top_anomalies.to_string(index=False)
            report += "\n\n"
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"レポートを保存しました: {save_path}")
        else:
            print(report)

# 使用例
def visualize_model_anomalies(detector, model_name):
    """
    指定機種の異常値を包括的に可視化
    Args:
        detector: PartsAnomalyDetectorのインスタンス
        model_name: 機種名
    """
    visualizer = AnomalyVisualizer(detector)
    
    print(f"=== 機種 '{model_name}' の異常値可視化開始 ===")
    
    # 1. 概要プロット
    visualizer.plot_model_overview(model_name)
    
    # 2. ヒートマップ
    visualizer.plot_parts_anomaly_heatmap(model_name)
    
    # 3. 散布図
    visualizer.plot_anomaly_scatter(model_name)
    
    # 4. 時系列プロット
    visualizer.plot_time_series_anomalies(model_name)
    
    # 5. レポート生成
    visualizer.generate_anomaly_report(model_name)
    
    print(f"=== 機種 '{model_name}' の異常値可視化完了 ===")

# 使用例
if __name__ == "__main__":
    # detector = PartsAnomalyDetector(df_parts)
    # detector.run_all_detections()
    # 
    # # 特定機種の可視化
    # visualize_model_anomalies(detector, 'Camera_Model_A')
    pass