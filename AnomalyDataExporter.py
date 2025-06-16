import pandas as pd
import numpy as np
from datetime import datetime
import os

class AnomalyDataExporter:
    """
    異常度データの出力・検索クラス
    """
    
    def __init__(self, detector):
        """
        初期化
        Args:
            detector: PartsAnomalyDetectorのインスタンス
        """
        self.detector = detector
        self.comprehensive_data = None
        self.summary_data = None
        
    def create_comprehensive_anomaly_table(self):
        """
        全データの包括的な異常度テーブルを作成
        """
        print("=== 包括的異常度テーブル作成中 ===")
        
        # ベースとなる部品使用データを取得
        base_data = self.detector.parts_usage.copy()
        
        # 各異常検出結果をマージ
        # 1. MAD異常検出結果
        if 'mad_anomalies' in self.detector.results and not self.detector.results['mad_anomalies'].empty:
            mad_data = self.detector.results['mad_anomalies'][
                ['Model', 'prod_month', 'Area', 'parts_no', 'modified_z_score', 'is_anomaly_mad']
            ].copy()
            base_data = base_data.merge(
                mad_data, 
                on=['Model', 'prod_month', 'Area', 'parts_no'], 
                how='left'
            )
        else:
            base_data['modified_z_score'] = np.nan
            base_data['is_anomaly_mad'] = False
        
        # 2. 相対比較異常検出結果
        if 'relative_anomalies' in self.detector.results and not self.detector.results['relative_anomalies'].empty:
            relative_data = self.detector.results['relative_anomalies'][
                ['Model', 'prod_month', 'Area', 'parts_no', 'relative_rank', 'is_anomaly_relative']
            ].copy()
            base_data = base_data.merge(
                relative_data, 
                on=['Model', 'prod_month', 'Area', 'parts_no'], 
                how='left'
            )
        else:
            base_data['relative_rank'] = np.nan
            base_data['is_anomaly_relative'] = False
        
        # 3. 総合異常スコアの計算
        base_data = self._calculate_comprehensive_score(base_data)
        
        # 4. 追加の分析情報を計算
        base_data = self._add_analysis_metrics(base_data)
        
        # 5. カラムの整理と並び替え
        base_data = self._organize_columns(base_data)
        
        self.comprehensive_data = base_data
        print(f"包括的異常度テーブル作成完了: {len(base_data)}件")
        
        return base_data
    
    def _calculate_comprehensive_score(self, data):
        """
        総合異常スコアを計算
        """
        # 正規化されたスコアを計算
        data['mad_score_normalized'] = 0
        data['relative_score_normalized'] = 0
        
        # MADスコアの正規化（0-1スケール）
        valid_mad = data['modified_z_score'].notna()
        if valid_mad.any():
            mad_values = data.loc[valid_mad, 'modified_z_score'].abs()
            max_mad = mad_values.max()
            if max_mad > 0:
                data.loc[valid_mad, 'mad_score_normalized'] = (mad_values / max_mad).clip(0, 1)
        
        # 相対ランクスコアの正規化
        valid_relative = data['relative_rank'].notna()
        if valid_relative.any():
            data.loc[valid_relative, 'relative_score_normalized'] = data.loc[valid_relative, 'relative_rank']
        
        # 総合異常スコア（重み付き平均）
        data['comprehensive_anomaly_score'] = (
            data['mad_score_normalized'] * 0.5 + 
            data['relative_score_normalized'] * 0.3 +
            (data['is_anomaly_mad'].astype(float) * 0.2)
        )
        
        # 異常レベルの分類
        data['anomaly_level'] = 'Normal'
        data.loc[data['comprehensive_anomaly_score'] >= 0.3, 'anomaly_level'] = 'Low'
        data.loc[data['comprehensive_anomaly_score'] >= 0.5, 'anomaly_level'] = 'Medium'
        data.loc[data['comprehensive_anomaly_score'] >= 0.7, 'anomaly_level'] = 'High'
        data.loc[data['comprehensive_anomaly_score'] >= 0.9, 'anomaly_level'] = 'Critical'
        
        return data
    
    def _add_analysis_metrics(self, data):
        """
        追加の分析指標を計算
        """
        # 同一機種・部品内でのランキング
        data['rank_in_model_parts'] = data.groupby(['Model', 'parts_no'])['usage_rate'].rank(
            method='dense', ascending=False
        )
        
        # 同一機種内でのランキング
        data['rank_in_model'] = data.groupby('Model')['usage_rate'].rank(
            method='dense', ascending=False
        )
        
        # 平均使用率との比較
        model_parts_avg = data.groupby(['Model', 'parts_no'])['usage_rate'].transform('mean')
        data['usage_rate_vs_avg'] = (data['usage_rate'] - model_parts_avg) / model_parts_avg
        
        # 最新修理日からの経過日数
        data['days_since_last_repair'] = (
            datetime.now().date() - pd.to_datetime(data['last_repair']).dt.date
        ).dt.days
        
        return data
    
    def _organize_columns(self, data):
        """
        カラムの整理と並び替え
        """
        # 主要カラムの順序定義
        primary_cols = [
            'Model', 'parts_no', 'prod_month', 'Area',
            'comprehensive_anomaly_score', 'anomaly_level',
            'usage_count', 'usage_rate', 'repair_count'
        ]
        
        # 異常検出関連カラム
        anomaly_cols = [
            'is_anomaly_mad', 'modified_z_score', 'mad_score_normalized',
            'is_anomaly_relative', 'relative_rank', 'relative_score_normalized'
        ]
        
        # 分析指標カラム
        analysis_cols = [
            'rank_in_model_parts', 'rank_in_model', 'usage_rate_vs_avg',
            'first_repair', 'last_repair', 'days_since_last_repair'
        ]
        
        # 存在するカラムのみを選択
        available_cols = []
        for col_group in [primary_cols, anomaly_cols, analysis_cols]:
            for col in col_group:
                if col in data.columns:
                    available_cols.append(col)
        
        return data[available_cols].sort_values(
            ['comprehensive_anomaly_score', 'Model', 'parts_no'], 
            ascending=[False, True, True]
        )
    
    def create_summary_table(self):
        """
        機種・部品別サマリーテーブルを作成
        """
        print("=== サマリーテーブル作成中 ===")
        
        if self.comprehensive_data is None:
            self.create_comprehensive_anomaly_table()
        
        # 機種・部品別の集計
        summary = self.comprehensive_data.groupby(['Model', 'parts_no']).agg({
            'comprehensive_anomaly_score': ['max', 'mean', 'count'],
            'usage_rate': ['max', 'mean', 'std'],
            'usage_count': 'sum',
            'repair_count': 'sum',
            'is_anomaly_mad': 'sum',
            'is_anomaly_relative': 'sum',
            'anomaly_level': lambda x: (x != 'Normal').sum(),
            'Area': 'nunique',
            'prod_month': 'nunique',
            'last_repair': 'max'
        }).reset_index()
        
        # カラム名の整理
        summary.columns = [
            'Model', 'parts_no',
            'max_anomaly_score', 'avg_anomaly_score', 'data_points',
            'max_usage_rate', 'avg_usage_rate', 'usage_rate_std',
            'total_usage_count', 'total_repair_count',
            'mad_anomaly_count', 'relative_anomaly_count', 'total_anomaly_count',
            'area_count', 'prod_month_count', 'latest_repair'
        ]
        
        # 異常度レベルの判定
        summary['overall_anomaly_level'] = 'Normal'
        summary.loc[summary['max_anomaly_score'] >= 0.3, 'overall_anomaly_level'] = 'Low'
        summary.loc[summary['max_anomaly_score'] >= 0.5, 'overall_anomaly_level'] = 'Medium'
        summary.loc[summary['max_anomaly_score'] >= 0.7, 'overall_anomaly_level'] = 'High'
        summary.loc[summary['max_anomaly_score'] >= 0.9, 'overall_anomaly_level'] = 'Critical'
        
        # 最新修理日からの経過日数
        summary['days_since_latest_repair'] = (
            datetime.now().date() - pd.to_datetime(summary['latest_repair']).dt.date
        ).dt.days
        
        # ソート
        summary = summary.sort_values(
            ['max_anomaly_score', 'total_anomaly_count'], 
            ascending=[False, False]
        )
        
        self.summary_data = summary
        print(f"サマリーテーブル作成完了: {len(summary)}件")
        
        return summary
    
    def export_to_files(self, output_dir='./anomaly_reports', timestamp=True):
        """
        ファイルに出力
        Args:
            output_dir: 出力ディレクトリ
            timestamp: ファイル名にタイムスタンプを付けるか
        """
        # ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # タイムスタンプ
        ts = datetime.now().strftime('%Y%m%d_%H%M%S') if timestamp else ''
        ts_suffix = f'_{ts}' if ts else ''
        
        print(f"=== ファイル出力開始: {output_dir} ===")
        
        # 1. 包括データの出力
        if self.comprehensive_data is not None:
            # CSV出力
            csv_path = os.path.join(output_dir, f'comprehensive_anomaly_data{ts_suffix}.csv')
            self.comprehensive_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"包括データCSV出力: {csv_path}")
            
            # Excel出力（複数シート）
            excel_path = os.path.join(output_dir, f'anomaly_analysis{ts_suffix}.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # 全データ
                self.comprehensive_data.to_excel(writer, sheet_name='全データ', index=False)
                
                # 異常度レベル別
                for level in ['Critical', 'High', 'Medium', 'Low']:
                    level_data = self.comprehensive_data[
                        self.comprehensive_data['anomaly_level'] == level
                    ]
                    if not level_data.empty:
                        level_data.to_excel(writer, sheet_name=f'{level}異常', index=False)
                
                # 機種別（上位10機種）
                top_models = self.comprehensive_data['Model'].value_counts().head(10).index
                for model in top_models:
                    model_data = self.comprehensive_data[
                        self.comprehensive_data['Model'] == model
                    ].sort_values('comprehensive_anomaly_score', ascending=False)
                    
                    # シート名の長さ制限対応
                    sheet_name = model[:30] if len(model) > 30 else model
                    model_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"包括データExcel出力: {excel_path}")
        
        # 2. サマリーデータの出力
        if self.summary_data is not None:
            summary_csv_path = os.path.join(output_dir, f'anomaly_summary{ts_suffix}.csv')
            self.summary_data.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"サマリーCSV出力: {summary_csv_path}")
        
        print("=== ファイル出力完了 ===")
        
        return {
            'comprehensive_csv': csv_path if self.comprehensive_data is not None else None,
            'comprehensive_excel': excel_path if self.comprehensive_data is not None else None,
            'summary_csv': summary_csv_path if self.summary_data is not None else None
        }
    
    def search_anomalies(self, model=None, parts_no=None, anomaly_level=None, 
                        min_score=None, max_score=None, area=None, top_n=None):
        """
        異常データの検索・フィルタリング
        Args:
            model: 機種名（部分一致）
            parts_no: 部品番号（部分一致）
            anomaly_level: 異常レベル（'Critical', 'High', 'Medium', 'Low', 'Normal'）
            min_score: 最小異常スコア
            max_score: 最大異常スコア
            area: 地域（部分一致）
            top_n: 上位N件
        Returns:
            pd.DataFrame: フィルタリング結果
        """
        if self.comprehensive_data is None:
            print("まず create_comprehensive_anomaly_table() を実行してください")
            return pd.DataFrame()
        
        result = self.comprehensive_data.copy()
        
        # フィルタリング
        if model:
            result = result[result['Model'].str.contains(model, case=False, na=False)]
        
        if parts_no:
            result = result[result['parts_no'].str.contains(parts_no, case=False, na=False)]
        
        if anomaly_level:
            if isinstance(anomaly_level, str):
                result = result[result['anomaly_level'] == anomaly_level]
            elif isinstance(anomaly_level, list):
                result = result[result['anomaly_level'].isin(anomaly_level)]
        
        if min_score is not None:
            result = result[result['comprehensive_anomaly_score'] >= min_score]
        
        if max_score is not None:
            result = result[result['comprehensive_anomaly_score'] <= max_score]
        
        if area:
            result = result[result['Area'].str.contains(area, case=False, na=False)]
        
        # ソート
        result = result.sort_values('comprehensive_anomaly_score', ascending=False)
        
        # 上位N件
        if top_n:
            result = result.head(top_n)
        
        print(f"検索結果: {len(result)}件")
        return result
    
    def get_model_parts_ranking(self, model, top_n=20):
        """
        指定機種の部品別異常度ランキング
        Args:
            model: 機種名
            top_n: 上位N件
        Returns:
            pd.DataFrame: ランキング結果
        """
        if self.summary_data is None:
            self.create_summary_table()
        
        model_data = self.summary_data[
            self.summary_data['Model'].str.contains(model, case=False, na=False)
        ].copy()
        
        if model_data.empty:
            print(f"機種 '{model}' のデータが見つかりません")
            return pd.DataFrame()
        
        # ランキング用の表示カラム選択
        display_cols = [
            'Model', 'parts_no', 'overall_anomaly_level',
            'max_anomaly_score', 'avg_anomaly_score',
            'total_anomaly_count', 'max_usage_rate', 'avg_usage_rate'
        ]
        
        result = model_data[display_cols].head(top_n)
        result['ranking'] = range(1, len(result) + 1)
        
        # カラム順序調整
        result = result[['ranking'] + display_cols]
        
        print(f"機種 '{model}' の部品異常度ランキング Top {len(result)}:")
        print(result.to_string(index=False))
        
        return result
    
    def quick_lookup(self, model, parts_no):
        """
        特定機種・部品の異常度クイック検索
        Args:
            model: 機種名
            parts_no: 部品番号
        Returns:
            dict: 異常度情報
        """
        if self.comprehensive_data is None:
            self.create_comprehensive_anomaly_table()
        
        # 完全一致検索
        exact_match = self.comprehensive_data[
            (self.comprehensive_data['Model'] == model) & 
            (self.comprehensive_data['parts_no'] == parts_no)
        ]
        
        if exact_match.empty:
            # 部分一致検索
            partial_match = self.comprehensive_data[
                (self.comprehensive_data['Model'].str.contains(model, case=False, na=False)) & 
                (self.comprehensive_data['parts_no'].str.contains(parts_no, case=False, na=False))
            ]
            
            if partial_match.empty:
                print(f"機種 '{model}' - 部品 '{parts_no}' のデータが見つかりません")
                return {}
            else:
                print(f"部分一致結果: {len(partial_match)}件")
                data = partial_match
        else:
            data = exact_match
        
        # サマリー情報作成
        summary_info = {
            '機種': model,
            '部品番号': parts_no,
            'データ件数': len(data),
            '最大異常スコア': data['comprehensive_anomaly_score'].max(),
            '平均異常スコア': data['comprehensive_anomaly_score'].mean(),
            '最高異常レベル': data['anomaly_level'].mode().iloc[0] if not data.empty else 'Unknown',
            'MAD異常検出回数': data['is_anomaly_mad'].sum(),
            '相対異常検出回数': data['is_anomaly_relative'].sum(),
            '最大使用率': data['usage_rate'].max(),
            '平均使用率': data['usage_rate'].mean(),
            '最新修理日': data['last_repair'].max()
        }
        
        print(f"\n=== クイック検索結果: {model} - {parts_no} ===")
        for key, value in summary_info.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        return summary_info

# 使用例関数
def export_all_anomaly_data(detector, output_dir='./anomaly_reports'):
    """
    全ての異常データを出力
    Args:
        detector: PartsAnomalyDetectorのインスタンス
        output_dir: 出力ディレクトリ
    Returns:
        AnomalyDataExporter: エクスポーターインスタンス
    """
    exporter = AnomalyDataExporter(detector)
    
    # 1. 包括的異常度テーブル作成
    exporter.create_comprehensive_anomaly_table()
    
    # 2. サマリーテーブル作成
    exporter.create_summary_table()
    
    # 3. ファイル出力
    output_files = exporter.export_to_files(output_dir)
    
    print(f"\n=== 出力ファイル一覧 ===")
    for file_type, path in output_files.items():
        if path:
            print(f"{file_type}: {path}")
    
    return exporter

# 使用例
if __name__ == "__main__":
    # 使用例
    # detector = PartsAnomalyDetector(df_parts)
    # detector.run_all_detections()
    # 
    # # 全データ出力
    # exporter = export_all_anomaly_data(detector)
    # 
    # # 検索例
    # # 高異常レベルの検索
    # high_anomalies = exporter.search_anomalies(anomaly_level=['Critical', 'High'], top_n=10)
    # 
    # # 特定機種の部品ランキング
    # ranking = exporter.get_model_parts_ranking('M150', top_n=10)
    # 
    # # クイック検索
    # info = exporter.quick_lookup('M150', 'PART001')
    pass