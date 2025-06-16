# =============================================================================
# 実際の使用例とワークフロー
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PartsAnomalyDetector import PartsAnomalyDetector
from AnomalyVisualizer import AnomalyVisualizer

# 1. データの読み込みとセットアップ
def setup_analysis(df_parts):
    """
    分析環境のセットアップ
    """
    print("=== 分析環境セットアップ ===")
    
    # データの基本チェック
    print(f"データ期間: {df_parts['date'].min()} ～ {df_parts['date'].max()}")
    print(f"機種数: {df_parts['Model'].nunique()}")
    print(f"部品種類: {df_parts['parts_no'].nunique()}")
    print(f"地域数: {df_parts['Area'].nunique()}")
    
    # 検出器の初期化
    detector = PartsAnomalyDetector(df_parts)
    
    return detector

# 2. 基本的な異常検出ワークフロー
def basic_anomaly_workflow(df_parts, target_models=None):
    """
    基本的な異常検出ワークフロー
    Args:
        df_parts: 修理データ
        target_models: 対象機種リスト（Noneの場合は全機種）
    """
    print("=== 基本異常検出ワークフロー開始 ===")
    
    # 1. 検出器セットアップ
    detector = setup_analysis(df_parts)
    
    # 2. 異常検出実行
    detector.run_all_detections(
        mad_threshold=3.5,      # MAD閾値（調整可能）
        percentile_threshold=95, # パーセンタイル閾値
        trend_window=3          # トレンド分析窓サイズ
    )
    
    # 3. 結果の確認
    print("\n=== 検出結果概要 ===")
    all_models = df_parts['Model'].unique()
    
    if target_models is None:
        target_models = all_models[:5]  # デフォルトで上位5機種
    
    # 各機種の異常数を集計
    model_anomaly_summary = []
    for model in all_models:
        model_results = detector.get_model_anomalies(model)
        total_anomalies = sum(len(data) for data in model_results.values())
        model_anomaly_summary.append({
            'Model': model,
            'Total_Anomalies': total_anomalies,
            'MAD_Anomalies': len(model_results.get('mad_anomalies', [])),
            'Relative_Anomalies': len(model_results.get('relative_anomalies', [])),
            'Trend_Anomalies': len(model_results.get('trend_anomalies', []))
        })
    
    summary_df = pd.DataFrame(model_anomaly_summary).sort_values('Total_Anomalies', ascending=False)
    print("\n機種別異常検出数 Top 10:")
    print(summary_df.head(10).to_string(index=False))
    
    return detector, summary_df

# 3. 詳細分析用関数
def detailed_model_analysis(detector, model_name, analysis_type='comprehensive'):
    """
    特定機種の詳細分析
    Args:
        detector: PartsAnomalyDetectorインスタンス
        model_name: 分析対象機種
        analysis_type: 分析タイプ（'comprehensive', 'focused', 'timeline'）
    """
    print(f"=== 機種 '{model_name}' 詳細分析開始 ===")
    
    # 可視化器の初期化
    visualizer = AnomalyVisualizer(detector)
    
    if analysis_type == 'comprehensive':
        # 包括的分析
        print("包括的分析を実行中...")
        visualizer.plot_model_overview(model_name)
        visualizer.plot_parts_anomaly_heatmap(model_name)
        visualizer.plot_anomaly_scatter(model_name)
        visualizer.plot_time_series_anomalies(model_name)
        
    elif analysis_type == 'focused':
        # 重要な異常に焦点を当てた分析
        print("重要異常に焦点を当てた分析を実行中...")
        model_results = detector.get_model_anomalies(model_name)
        
        # 最も深刻な異常部品を特定
        if 'mad_anomalies' in model_results and not model_results['mad_anomalies'].empty:
            top_anomaly = model_results['mad_anomalies'].loc[
                model_results['mad_anomalies']['modified_z_score'].idxmax()
            ]
            critical_parts = top_anomaly['parts_no']
            
            print(f"最重要異常部品: {critical_parts}")
            visualizer.plot_time_series_anomalies(model_name, critical_parts)
            visualizer.plot_anomaly_scatter(model_name, method='mad_anomalies')
        
    elif analysis_type == 'timeline':
        # 時系列中心の分析
        print("時系列分析を実行中...")
        visualizer.plot_time_series_anomalies(model_name)
        
        # 月次トレンドの詳細分析
        model_data = detector.df_parts[detector.df_parts['Model'] == model_name]
        monthly_trend = model_data.groupby(['year_month', 'parts_no']).size().unstack(fill_value=0)
        
        if not monthly_trend.empty:
            print("\n月次部品使用数トレンド（上位5部品）:")
            top_parts = monthly_trend.sum().nlargest(5).index
            print(monthly_trend[top_parts].tail(6).to_string())
    
    # レポート生成
    visualizer.generate_anomaly_report(model_name)
    
    print(f"=== 機種 '{model_name}' 詳細分析完了 ===")

# 4. 異常値の深掘り分析
def drill_down_anomaly(detector, model_name, parts_no, analysis_date_range=None):
    """
    特定の異常値の深掘り分析
    Args:
        detector: PartsAnomalyDetectorインスタンス
        model_name: 機種名
        parts_no: 部品番号
        analysis_date_range: 分析期間（tuple: (start_date, end_date)）
    """
    print(f"=== 深掘り分析: {model_name} - {parts_no} ===")
    
    # 対象データの抽出
    target_data = detector.df_parts[
        (detector.df_parts['Model'] == model_name) & 
        (detector.df_parts['parts_no'] == parts_no)
    ]
    
    if analysis_date_range:
        start_date, end_date = analysis_date_range
        target_data = target_data[
            (target_data['date'] >= start_date) & 
            (target_data['date'] <= end_date)
        ]
    
    if target_data.empty:
        print("対象データが見つかりません")
        return
    
    # 詳細統計
    print("\n=== 基本統計 ===")
    print(f"総修理件数: {target_data['IF_ID'].nunique()}")
    print(f"総部品使用数: {len(target_data)}")
    print(f"期間: {target_data['date'].min()} ～ {target_data['date'].max()}")
    
    # 地域別分析
    print("\n=== 地域別統計 ===")
    area_stats = target_data.groupby('Area').agg({
        'IF_ID': 'nunique',
        'date': ['min', 'max']
    })
    area_stats.columns = ['修理件数', '最初の修理', '最後の修理']
    print(area_stats.to_string())
    
    # 製造年月別分析
    print("\n=== 製造年月別統計 ===")
    prod_stats = target_data.groupby('prod_month').agg({
        'IF_ID': 'nunique',
        'date': ['min', 'max']
    })
    prod_stats.columns = ['修理件数', '最初の修理', '最後の修理']
    print(prod_stats.to_string())
    
    # 時系列可視化
    visualizer = AnomalyVisualizer(detector)
    visualizer.plot_time_series_anomalies(model_name, parts_no)
    
    # 同時交換部品の分析
    print("\n=== 同時交換部品分析 ===")
    concurrent_repairs = target_data.groupby('IF_ID')['parts_no'].apply(list)
    concurrent_parts = {}
    
    for repair_parts in concurrent_repairs:
        if len(repair_parts) > 1:  # 複数部品の同時交換
            other_parts = [p for p in repair_parts if p != parts_no]
            for other_part in other_parts:
                concurrent_parts[other_part] = concurrent_parts.get(other_part, 0) + 1
    
    if concurrent_parts:
        concurrent_df = pd.DataFrame(list(concurrent_parts.items()), 
                                   columns=['併用部品', '併用回数'])
        concurrent_df = concurrent_df.sort_values('併用回数', ascending=False)
        print("よく同時交換される部品 Top 5:")
        print(concurrent_df.head().to_string(index=False))

# 5. 定期監視用の自動化関数
def automated_monitoring(df_parts, alert_threshold=5, output_dir='./reports'):
    """
    定期監視用の自動化関数
    Args:
        df_parts: 修理データ
        alert_threshold: アラート閾値（異常数）
        output_dir: 出力ディレクトリ
    """
    print("=== 自動監視開始 ===")
    
    # 検出器実行
    detector = PartsAnomalyDetector(df_parts)
    detector.run_all_detections()
    
    # アラート対象機種の特定
    alert_models = []
    for model in df_parts['Model'].unique():
        model_results = detector.get_model_anomalies(model)
        total_anomalies = sum(len(data) for data in model_results.values())
        
        if total_anomalies >= alert_threshold:
            alert_models.append({
                'model': model,
                'anomaly_count': total_anomalies,
                'priority': 'High' if total_anomalies >= alert_threshold * 2 else 'Medium'
            })
    
    # アラートレポート生成
    if alert_models:
        print(f"\n=== アラート: {len(alert_models)}機種で異常検出 ===")
        
        alert_df = pd.DataFrame(alert_models).sort_values('anomaly_count', ascending=False)
        print(alert_df.to_string(index=False))
        
        # 高優先度機種の詳細レポート生成
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for alert in alert_models:
            if alert['priority'] == 'High':
                model_name = alert['model']
                visualizer = AnomalyVisualizer(detector)
                
                # レポートファイル生成
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_path = f"{output_dir}/alert_{model_name}_{timestamp}.txt"
                visualizer.generate_anomaly_report(model_name, report_path)
                
                print(f"高優先度アラートレポート生成: {report_path}")
    
    else:
        print("アラート対象の異常は検出されませんでした")
    
    return detector, alert_models

# 6. 使用例の統合
def main_workflow_example():
    """
    メインワークフローの使用例
    """
    # サンプルコード（実際のデータに置き換えて使用）
    
    # Step 1: データ読み込み
    df_parts = pd.read_pickle('test_data.pkl')
    df_parts['date'] = pd.to_datetime(df_parts['date'])
    #df_parts['prod_month'] = pd.to_datetime(df_parts['prod_month']).dt.to_period('M')
    
    # Step 2: 基本的な異常検出
    detector, summary = basic_anomaly_workflow(df_parts)
    
    # Step 3: 上位異常機種の詳細分析
    top_models = summary.head(3)['Model'].tolist()
    for model in top_models:
        detailed_model_analysis(detector, model, 'comprehensive')
    
    # Step 4: 特定異常の深掘り
    # 例：最も異常スコアの高い機種・部品を分析
    if not detector.results['mad_anomalies'].empty:
        top_anomaly = detector.results['mad_anomalies'].nlargest(1, 'modified_z_score').iloc[0]
        drill_down_anomaly(detector, top_anomaly['Model'], top_anomaly['parts_no'])
    
    # Step 5: 定期監視
    detector, alerts = automated_monitoring(df_parts, alert_threshold=5)
    
    print("ワークフロー例を確認してください。実際のデータでコメントアウトを解除して実行してください。")

if __name__ == "__main__":
    main_workflow_example()