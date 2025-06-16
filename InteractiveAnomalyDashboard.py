import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed

class InteractiveAnomalyDashboard:
    """
    対話的異常検索ダッシュボード
    """
    
    def __init__(self, exporter):
        """
        初期化
        Args:
            exporter: AnomalyDataExporterのインスタンス
        """
        self.exporter = exporter
        if self.exporter.comprehensive_data is None:
            self.exporter.create_comprehensive_anomaly_table()
        if self.exporter.summary_data is None:
            self.exporter.create_summary_table()
    
    def create_search_widgets(self):
        """
        検索用ウィジェットを作成
        """
        # データから選択肢を取得
        models = [''] + sorted(self.exporter.comprehensive_data['Model'].unique().tolist())
        parts = [''] + sorted(self.exporter.comprehensive_data['parts_no'].unique().tolist())
        areas = [''] + sorted(self.exporter.comprehensive_data['Area'].unique().tolist())
        levels = [''] + ['Critical', 'High', 'Medium', 'Low', 'Normal']
        
        # ウィジェット作成
        model_widget = widgets.Dropdown(
            options=models,
            value='',
            description='機種:',
            style={'description_width': 'initial'}
        )
        
        parts_widget = widgets.Dropdown(
            options=parts,
            value='',
            description='部品番号:',
            style={'description_width': 'initial'}
        )
        
        area_widget = widgets.Dropdown(
            options=areas,
            value='',
            description='地域:',
            style={'description_width': 'initial'}
        )
        
        level_widget = widgets.Dropdown(
            options=levels,
            value='',
            description='異常レベル:',
            style={'description_width': 'initial'}
        )
        
        min_score_widget = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=1.0,
            step=0.1,
            description='最小スコア:',
            style={'description_width': 'initial'}
        )
        
        top_n_widget = widgets.IntSlider(
            value=20,
            min=1,
            max=100,
            step=1,
            description='表示件数:',
            style={'description_width': 'initial'}
        )
        
        return {
            'model': model_widget,
            'parts_no': parts_widget,
            'area': area_widget,
            'anomaly_level': level_widget,
            'min_score': min_score_widget,
            'top_n': top_n_widget
        }
    
    def interactive_search(self):
        """
        対話的検索インターface
        """
        widgets_dict = self.create_search_widgets()
        
        def search_and_display(model, parts_no, area, anomaly_level, min_score, top_n):
            # 検索条件の準備
            search_kwargs = {'top_n': top_n}
            
            if model:
                search_kwargs['model'] = model
            if parts_no:
                search_kwargs['parts_no'] = parts_no
            if area:
                search_kwargs['area'] = area
            if anomaly_level:
                search_kwargs['anomaly_level'] = anomaly_level
            if min_score > 0:
                search_kwargs['min_score'] = min_score
            
            # 検索実行
            result = self.exporter.search_anomalies(**search_kwargs)
            
            if result.empty:
                print("検索条件に該当するデータがありません")
                return
            
            # 結果表示用のカラム選択
            display_cols = [
                'Model', 'parts_no', 'Area', 'anomaly_level',
                'comprehensive_anomaly_score', 'usage_rate',
                'usage_count', 'repair_count'
            ]
            
            display_data = result[display_cols].round(4)
            
            # 検索条件の表示
            print("=== 検索条件 ===")
            for key, value in search_kwargs.items():
                print(f"{key}: {value}")
            
            print(f"\n=== 検索結果: {len(result)}件 ===")
            
            # Jupyter環境での表示
            try:
                display(HTML(display_data.to_html(index=False, escape=False)))
            except:
                # 通常のコンソール表示
                print(display_data.to_string(index=False))
            
            # 統計サマリー
            print(f"\n=== 結果統計 ===")
            print(f"平均異常スコア: {result['comprehensive_anomaly_score'].mean():.4f}")
            print(f"最大異常スコア: {result['comprehensive_anomaly_score'].max():.4f}")
            print(f"異常レベル分布:")
            level_counts = result['anomaly_level'].value_counts()
            for level, count in level_counts.items():
                print(f"  {level}: {count}件")
        
        # インタラクティブウィジェット表示
        return interact(
            search_and_display,
            model=widgets_dict['model'],
            parts_no=widgets_dict['parts_no'],
            area=widgets_dict['area'],
            anomaly_level=widgets_dict['anomaly_level'],
            min_score=widgets_dict['min_score'],
            top_n=widgets_dict['top_n']
        )
    
    def create_summary_dashboard(self):
        """
        サマリーダッシュボードを作成
        """
        print("=== 異常検出サマリーダッシュボード ===")
        
        # 基本統計
        total_records = len(self.exporter.comprehensive_data)
        total_models = self.exporter.comprehensive_data['Model'].nunique()
        total_parts = self.exporter.comprehensive_data['parts_no'].nunique()
        
        print(f"総データ数: {total_records:,}件")
        print(f"機種数: {total_models}機種")
        print(f"部品種類数: {total_parts}種類")
        
        # 異常レベル分布
        print(f"\n=== 異常レベル分布 ===")
        level_dist = self.exporter.comprehensive_data['anomaly_level'].value_counts()
        for level, count in level_dist.items():
            percentage = (count / total_records) * 100
            print(f"{level}: {count}件 ({percentage:.1f}%)")
        
        # 機種別異常数 Top 10
        print(f"\n=== 機種別異常数 Top 10 ===")
        model_anomalies = self.exporter.comprehensive_data[
            self.exporter.comprehensive_data['anomaly_level'] != 'Normal'
        ]['Model'].value_counts().head(10)
        
        for model, count in model_anomalies.items():
            print(f"{model}: {count}件")
        
        # 部品別異常数 Top 10
        print(f"\n=== 部品別異常数 Top 10 ===")
        parts_anomalies = self.exporter.comprehensive_data[
            self.exporter.comprehensive_data['anomaly_level'] != 'Normal'
        ]['parts_no'].value_counts().head(10)
        
        for parts, count in parts_anomalies.items():
            print(f"{parts}: {count}件")
        
        # 地域別異常数
        print(f"\n=== 地域別異常数 ===")
        area_anomalies = self.exporter.comprehensive_data[
            self.exporter.comprehensive_data['anomaly_level'] != 'Normal'
        ]['Area'].value_counts()
        
        for area, count in area_anomalies.items():
            print(f"{area}: {count}件")
    
    def plot_anomaly_overview(self, figsize=(15, 12)):
        """
        異常検出概要の可視化
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('異常検出概要ダッシュボード', fontsize=16, fontweight='bold')
        
        # 1. 異常レベル分布（円グラフ）
        ax1 = axes[0, 0]
        level_dist = self.exporter.comprehensive_data['anomaly_level'].value_counts()
        colors = ['red', 'orange', 'yellow', 'lightblue', 'lightgreen'][:len(level_dist)]
        level_dist.plot(kind='pie', ax=ax1, autopct='%1.1f%%', colors=colors)
        ax1.set_title('異常レベル分布')
        ax1.set_ylabel('')
        
        # 2. 異常スコア分布（ヒストグラム）
        ax2 = axes[0, 1]
        self.exporter.comprehensive_data['comprehensive_anomaly_score'].hist(
            bins=50, ax=ax2, alpha=0.7, color='skyblue', edgecolor='black'
        )
        ax2.set_title('異常スコア分布')
        ax2.set_xlabel('異常スコア')
        ax2.set_ylabel('頻度')
        ax2.grid(True, alpha=0.3)
        
        # 3. 機種別異常数（上位10機種）
        ax3 = axes[0, 2]
        model_anomalies = self.exporter.comprehensive_data[
            self.exporter.comprehensive_data['anomaly_level'] != 'Normal'
        ]['Model'].value_counts().head(10)
        
        model_anomalies.plot(kind='bar', ax=ax3, color='lightcoral')
        ax3.set_title('機種別異常数 Top 10')
        ax3.set_xlabel('機種')
        ax3.set_ylabel('異常数')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 部品別異常数（上位10部品）
        ax4 = axes[1, 0]
        parts_anomalies = self.exporter.comprehensive_data[
            self.exporter.comprehensive_data['anomaly_level'] != 'Normal'
        ]['parts_no'].value_counts().head(10)
        
        parts_anomalies.plot(kind='bar', ax=ax4, color='lightgreen')
        ax4.set_title('部品別異常数 Top 10')
        ax4.set_xlabel('部品番号')
        ax4.set_ylabel('異常数')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 地域別異常数
        ax5 = axes[1, 1]
        area_anomalies = self.exporter.comprehensive_data[
            self.exporter.comprehensive_data['anomaly_level'] != 'Normal'
        ]['Area'].value_counts()
        
        area_anomalies.plot(kind='bar', ax=ax5, color='gold')
        ax5.set_title('地域別異常数')
        ax5.set_xlabel('地域')
        ax5.set_ylabel('異常数')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. 異常スコア vs 使用率 散布図
        ax6 = axes[1, 2]
        scatter_data = self.exporter.comprehensive_data[
            self.exporter.comprehensive_data['anomaly_level'] != 'Normal'
        ]
        
        colors_map = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'lightblue'}
        
        for level, color in colors_map.items():
            level_data = scatter_data[scatter_data['anomaly_level'] == level]
            if not level_data.empty:
                ax6.scatter(level_data['usage_rate'], level_data['comprehensive_anomaly_score'], 
                           c=color, label=level, alpha=0.6)
        
        ax6.set_title('異常スコア vs 使用率')
        ax6.set_xlabel('使用率')
        ax6.set_ylabel('異常スコア')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_model_comparison_chart(self, models_list, metric='max_anomaly_score'):
        """
        機種間比較チャート
        Args:
            models_list: 比較対象機種のリスト
            metric: 比較指標
        """
        comparison_data = []
        
        for model in models_list:
            model_summary = self.exporter.summary_data[
                self.exporter.summary_data['Model'] == model
            ]
            
            if not model_summary.empty:
                comparison_data.append({
                    'Model': model,
                    'max_anomaly_score': model_summary['max_anomaly_score'].max(),
                    'avg_anomaly_score': model_summary['avg_anomaly_score'].mean(),
                    'total_anomaly_count': model_summary['total_anomaly_count'].sum(),
                    'unique_parts': len(model_summary),
                    'critical_parts': len(model_summary[model_summary['overall_anomaly_level'] == 'Critical']),
                    'high_parts': len(model_summary[model_summary['overall_anomaly_level'] == 'High'])
                })
        
        if not comparison_data:
            print("比較データがありません")
            return
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('機種間比較チャート', fontsize=16, fontweight='bold')
        
        # 1. 最大異常スコア比較
        ax1 = axes[0, 0]
        comparison_df.set_index('Model')['max_anomaly_score'].plot(kind='bar', ax=ax1, color='red', alpha=0.7)
        ax1.set_title('最大異常スコア比較')
        ax1.set_ylabel('スコア')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 平均異常スコア比較
        ax2 = axes[0, 1]
        comparison_df.set_index('Model')['avg_anomaly_score'].plot(kind='bar', ax=ax2, color='orange', alpha=0.7)
        ax2.set_title('平均異常スコア比較')
        ax2.set_ylabel('スコア')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 異常検出数比較
        ax3 = axes[1, 0]
        comparison_df.set_index('Model')['total_anomaly_count'].plot(kind='bar', ax=ax3, color='gold', alpha=0.7)
        ax3.set_title('総異常検出数比較')
        ax3.set_ylabel('検出数')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 高リスク部品数比較
        ax4 = axes[1, 1]
        risk_data = comparison_df.set_index('Model')[['critical_parts', 'high_parts']]
        risk_data.plot(kind='bar', ax=ax4, color=['red', 'orange'], alpha=0.7)
        ax4.set_title('高リスク部品数比較')
        ax4.set_ylabel('部品数')
        ax4.legend(['Critical', 'High'])
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 数値表も表示
        print("=== 機種間比較表 ===")
        display_df = comparison_df.round(4)
        print(display_df.to_string(index=False))
        
        return comparison_df

class AnomalyReportGenerator:
    """
    自動レポート生成クラス
    """
    
    def __init__(self, exporter):
        self.exporter = exporter
    
    def generate_executive_summary(self, output_path=None):
        """
        エグゼクティブサマリーレポート生成
        """
        if self.exporter.comprehensive_data is None:
            self.exporter.create_comprehensive_anomaly_table()
        
        data = self.exporter.comprehensive_data
        
        # 基本統計
        total_records = len(data)
        anomaly_records = len(data[data['anomaly_level'] != 'Normal'])
        anomaly_rate = (anomaly_records / total_records) * 100
        
        # 高リスク事項
        critical_count = len(data[data['anomaly_level'] == 'Critical'])
        high_count = len(data[data['anomaly_level'] == 'High'])
        
        # 機種別リスク
        model_risk = data[data['anomaly_level'].isin(['Critical', 'High'])]['Model'].value_counts().head(5)
        
        # 部品別リスク
        parts_risk = data[data['anomaly_level'].isin(['Critical', 'High'])]['parts_no'].value_counts().head(5)
        
        # レポート作成
        report = f"""
===========================================
       品質異常検出 エグゼクティブサマリー
===========================================

生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

【全体概要】
・総分析データ数: {total_records:,}件
・異常検出数: {anomaly_records:,}件
・異常検出率: {anomaly_rate:.1f}%

【リスクレベル別内訳】
・Critical (最重要): {critical_count}件
・High (重要): {high_count}件
・Medium (中程度): {len(data[data['anomaly_level'] == 'Medium'])}件
・Low (軽微): {len(data[data['anomaly_level'] == 'Low'])}件

【要注意機種 Top 5】
"""
        
        for i, (model, count) in enumerate(model_risk.items(), 1):
            report += f"{i}. {model}: {count}件\n"
        
        report += f"""
【要注意部品 Top 5】
"""
        
        for i, (parts, count) in enumerate(parts_risk.items(), 1):
            report += f"{i}. {parts}: {count}件\n"
        
        # 推奨アクション
        report += f"""
【推奨アクション】
・Critical/Highレベルの異常({critical_count + high_count}件)の即座な確認が必要
・要注意機種「{model_risk.index[0] if not model_risk.empty else 'N/A'}」の詳細調査を推奨
・要注意部品「{parts_risk.index[0] if not parts_risk.empty else 'N/A'}」の設計・製造プロセス見直しを検討

【次回分析推奨】
・{datetime.now().strftime('%Y年%m月%d日')}から1週間後
・新たな修理データ追加後の再分析

===========================================
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"エグゼクティブサマリーを保存: {output_path}")
        else:
            print(report)
        
        return report

# 統合使用例
def complete_anomaly_analysis_workflow(df_parts, output_dir='./anomaly_reports'):
    """
    完全な異常分析ワークフロー
    """
    print("=== 完全異常分析ワークフロー開始 ===")
    
    # 1. 異常検出実行
    from .PartsAnomalyDetector import PartsAnomalyDetector  # 適切なインポートパスに修正
    detector = PartsAnomalyDetector(df_parts)
    detector.run_all_detections()
    
    # 2. データ出力
    exporter = export_all_anomaly_data(detector, output_dir)
    
    # 3. ダッシュボード作成
    dashboard = InteractiveAnomalyDashboard(exporter)
    dashboard.create_summary_dashboard()
    dashboard.plot_anomaly_overview()
    
    # 4. エグゼクティブサマリー生成
    report_generator = AnomalyReportGenerator(exporter)
    summary_path = os.path.join(output_dir, f'executive_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    report_generator.generate_executive_summary(summary_path)
    
    print("=== 完全異常分析ワークフロー完了 ===")
    
    return {
        'detector': detector,
        'exporter': exporter,
        'dashboard': dashboard,
        'report_generator': report_generator
    }

# 使用例
if __name__ == "__main__":
    # 基本的な使用例
    # detector = PartsAnomalyDetector(df_parts)
    # detector.run_all_detections()
    # 
    # # データ出力
    # exporter = export_all_anomaly_data(detector)
    # 
    # # 対話的検索（Jupyter環境）
    # dashboard = InteractiveAnomalyDashboard(exporter)
    # dashboard.interactive_search()
    # 
    # # 個別検索例
    # # 特定機種の部品ランキング
    # ranking = exporter.get_model_parts_ranking('M150')
    # 
    # # 高異常スコアの検索
    # high_anomalies = exporter.search_anomalies(min_score=0.7, top_n=20)
    # 
    # # クイック検索
    # info = exporter.quick_lookup('M150', 'PART001')
    pass