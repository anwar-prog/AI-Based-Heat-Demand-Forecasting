import sys
import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.base import clone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class FastSVRForecasting:

    def __init__(self, data_path: str = None, fast_mode: bool = True):
        if data_path is None:
            self.data_path = Path("/workspace/Thesis/01_data/processed_data")
        else:
            self.data_path = Path(data_path)

        self.results_dir = Path("/workspace/Thesis/04_svr_implementation/results/33features")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.horizons = [1, 6, 24, 48, 72]
        self.kernels = ['rbf', 'linear'] if fast_mode else ['rbf', 'poly', 'linear', 'sigmoid']
        self.fast_mode = fast_mode

        self.results = {}

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        print("Fast SVR Forecasting System Initialized")
        print(f"Target Horizons: {self.horizons}")
        print(f"Fast Mode: {fast_mode}")
        print(f"Data path: {self.data_path}")
        print(f"Results will be saved to: {self.results_dir}")

    def load_data(self) -> pd.DataFrame:
        try:
            file_path = self.data_path / "merged_dataset.csv"

            print(f"Loading dataset: {file_path}")

            if not file_path.exists():
                available_files = list(self.data_path.glob("*.csv"))
                if available_files:
                    file_path = available_files[0]
                    print(f"Using: {file_path.name}")
                else:
                    raise FileNotFoundError(f"No CSV files found in {self.data_path}")

            if self.fast_mode:
                sample_df = pd.read_csv(file_path, nrows=5)
                key_patterns = ['temp', 'V1', 'N1', 'N2', 'hdd', 'hour', 'datetime', 'timestamp', 'demand', 'load']
                cols_to_keep = []

                for col in sample_df.columns:
                    if any(pattern in col.lower() for pattern in key_patterns):
                        cols_to_keep.append(col)

                numeric_cols = []
                for col in sample_df.columns:
                    try:
                        pd.to_numeric(sample_df[col], errors='raise')
                        numeric_cols.append(col)
                        if len(numeric_cols) >= 25:
                            break
                    except:
                        continue

                final_cols = list(set(cols_to_keep + numeric_cols))
                df = pd.read_csv(file_path, usecols=final_cols)
                print(f"Fast mode: Loaded {len(final_cols)} key columns")
            else:
                df = pd.read_csv(file_path)

            print(f"Data loaded: {df.shape}")

            date_columns = ['datetime', 'timestamp', 'date', 'time']
            date_col = None

            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break

            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                print(f"Date range: {df.index[0]} to {df.index[-1]}")
            else:
                df.index = pd.date_range(start='2021-01-01', periods=len(df), freq='H')
                print("Created synthetic datetime index")

            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        print("Preparing feature set...")

        numeric_columns = []
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                continue

        print(f"Found {len(numeric_columns)} numeric columns")
        df_numeric = df[numeric_columns].copy()

        if self.fast_mode:
            priority_patterns = [
                'V1', 'N1', 'N2', 'V6', 'W1', 'F1_Sud', 'F1_Nord', 'B1_B2', 'V2', 'ZN',
                'temp', 'hdd', 'hour', 'pressure', 'humidity', 'solar', 'wind'
            ]

            selected_features = []

            for pattern in priority_patterns:
                matching_cols = [col for col in numeric_columns if pattern.lower() in col.lower()]
                selected_features.extend(matching_cols[:2])
                if len(selected_features) >= 30:
                    break

            remaining_cols = [col for col in numeric_columns if col not in selected_features]
            selected_features.extend(remaining_cols[:max(0, 33-len(selected_features))])

            features = selected_features[:33]
            print(f"Fast mode: Using {len(features)} features")
        else:
            features = numeric_columns[:33]

        if 'hour' not in df_numeric.columns:
            df_numeric['hour'] = df_numeric.index.hour
            if 'hour' not in features:
                features.append('hour')

        if len(features) < 33:
            if 'hour_sin' not in df_numeric.columns:
                df_numeric['hour_sin'] = np.sin(2 * np.pi * df_numeric['hour'] / 24)
                df_numeric['hour_cos'] = np.cos(2 * np.pi * df_numeric['hour'] / 24)
                features.extend(['hour_sin', 'hour_cos'])

        valid_features = [f for f in features if f in df_numeric.columns and not df_numeric[f].isna().all()]

        print(f"Final feature set: {len(valid_features)} features")

        return df_numeric[valid_features], valid_features

    def create_synthetic_heat_demand(self, df: pd.DataFrame) -> pd.Series:
        print("Creating synthetic heat demand target...")

        temp_patterns = ['hdd', 'temp']
        temp_col = None

        for pattern in temp_patterns:
            matching_cols = [col for col in df.columns if pattern in col.lower()]
            if matching_cols:
                temp_col = matching_cols[0]
                break

        if temp_col:
            if 'hdd' in temp_col.lower():
                base_demand = df[temp_col] * 3.5
            else:
                base_demand = np.maximum(0, 15.5 - df[temp_col]) * 3.5
            print(f"Using {temp_col} for base demand")
        else:
            hours = np.arange(len(df))
            synthetic_temp = 10 + 8 * np.sin(2 * np.pi * hours / (24 * 365)) + 5 * np.sin(2 * np.pi * hours / 24)
            base_demand = np.maximum(0, 15.5 - synthetic_temp) * 3.5
            print("Created synthetic temperature for base demand")

        hour = df.index.hour if hasattr(df.index, 'hour') else np.arange(len(df)) % 24
        time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24)

        if hasattr(df.index, 'dayofyear'):
            day_of_year = df.index.dayofyear
        else:
            day_of_year = np.arange(len(df)) % 365

        seasonal_factor = 1.0 + 0.3 * np.cos(2 * np.pi * day_of_year / 365)

        heat_demand = base_demand * time_factor * seasonal_factor
        heat_demand = np.maximum(heat_demand, 0.1)

        print(f"Synthetic heat demand range: {heat_demand.min():.2f} to {heat_demand.max():.2f} MWh")
        return heat_demand

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        targets = pd.DataFrame(index=df.index)

        target_cols = ['heat_demand', 'demand', 'load', 'consumption']
        heat_demand = None

        for col in target_cols:
            if col in df.columns:
                heat_demand = df[col]
                print(f"Using existing {col} column")
                break

        if heat_demand is None:
            heat_demand = self.create_synthetic_heat_demand(df)

        for horizon in self.horizons:
            targets[f'target_{horizon}h'] = heat_demand.shift(-horizon)

        print(f"Created targets for horizons: {self.horizons}")
        return targets

    def get_fast_param_grids(self, kernel: str) -> Dict[str, List]:
        if kernel == 'rbf':
            return {
                'C': [1, 10, 100],
                'gamma': ['scale', 0.1],
                'epsilon': [0.1]
            }
        elif kernel == 'linear':
            return {
                'C': [1, 100],
                'epsilon': [0.1]
            }
        elif kernel == 'poly':
            return {
                'C': [1, 100],
                'gamma': ['scale'],
                'degree': [2],
                'epsilon': [0.1]
            }
        else:
            return {
                'C': [1, 100],
                'gamma': ['scale'],
                'epsilon': [0.1]
            }

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 kernel: str, horizon: int) -> Dict[str, Any]:
        param_grid = self.get_fast_param_grids(kernel)

        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)

        print(f"  {kernel}: Testing {total_combinations} combinations")

        tscv = TimeSeriesSplit(n_splits=3)

        svr = SVR(kernel=kernel, cache_size=2000)

        search = GridSearchCV(
            svr, param_grid,
            cv=tscv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )

        search.fit(X_train, y_train)

        print(f"  {kernel}: Best score = {search.best_score_:.4f}")
        return search.best_params_

    def train_kernel_comparison(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                horizon: int) -> Dict[str, Dict]:
        print(f"Kernel comparison for {horizon}h horizon:")

        kernel_results = {}

        for kernel in self.kernels:
            try:
                best_params = self.optimize_hyperparameters(X_train, y_train, kernel, horizon)

                svr = SVR(kernel=kernel, **best_params, cache_size=2000)
                svr.fit(X_train, y_train)

                val_pred = svr.predict(X_val)

                val_r2 = r2_score(y_val, val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

                kernel_results[kernel] = {
                    'model': svr,
                    'params': best_params,
                    'val_r2': val_r2,
                    'val_rmse': val_rmse,
                    'predictions': val_pred,
                    'actuals': y_val
                }

                print(f"  {kernel}: R2={val_r2:.4f}, RMSE={val_rmse:.4f}")

                if self.fast_mode and val_r2 > 0.85:
                    print(f"  Early stopping: {kernel} achieved good performance")
                    break

            except Exception as e:
                print(f"  {kernel} failed: {e}")
                continue

        return kernel_results

    def select_best_model(self, kernel_results: Dict[str, Dict], horizon: int) -> Tuple[Any, str]:
        if not kernel_results:
            raise ValueError("No successful kernel results")

        best_kernel = max(kernel_results.keys(), key=lambda k: kernel_results[k]['val_r2'])
        best_score = kernel_results[best_kernel]['val_r2']

        print(f"  Best kernel: {best_kernel} (R2={best_score:.4f})")

        return kernel_results[best_kernel]['model'], best_kernel

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                       horizon: int) -> Dict[str, float]:
        pred = model.predict(X_test)

        metrics = {
            'horizon': horizon,
            'r2_score': r2_score(y_test, pred),
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'mae': mean_absolute_error(y_test, pred),
            'mape': mean_absolute_percentage_error(y_test, pred),
            'predictions': pred,
            'actuals': y_test
        }

        return metrics

    def create_visualizations(self, results, all_kernel_results, timestamp):
        print("Creating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('SVR Comprehensive Performance Analysis (33 Features)', fontsize=16, fontweight='bold')

        horizons = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        mape_scores = []
        best_kernels = []

        for horizon in self.horizons:
            if horizon in results:
                horizons.append(f"{horizon}h")
                metrics = results[horizon]['metrics']
                r2_scores.append(metrics['r2_score'])
                rmse_scores.append(metrics['rmse'])
                mae_scores.append(metrics['mae'])
                mape_scores.append(metrics['mape'])
                best_kernels.append(results[horizon]['model_type'].split('(')[1].split(')')[0])

        bars1 = axes[0,0].bar(horizons, r2_scores, color='skyblue', alpha=0.8)
        axes[0,0].set_title('R² Score by Horizon')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_ylim(0, 1)
        for bar, v, kernel in zip(bars1, r2_scores, best_kernels):
            axes[0,0].text(bar.get_x() + bar.get_width()/2., v + 0.01,
                           f'{v:.3f}\n({kernel})', ha='center', fontweight='bold')

        axes[0,1].bar(horizons, rmse_scores, color='lightcoral', alpha=0.8)
        axes[0,1].set_title('RMSE by Horizon')
        axes[0,1].set_ylabel('RMSE')
        for i, v in enumerate(rmse_scores):
            axes[0,1].text(i, v + max(rmse_scores)*0.01, f'{v:.2f}', ha='center', fontweight='bold')

        axes[0,2].bar(horizons, mae_scores, color='lightgreen', alpha=0.8)
        axes[0,2].set_title('MAE by Horizon')
        axes[0,2].set_ylabel('MAE')
        for i, v in enumerate(mae_scores):
            axes[0,2].text(i, v + max(mae_scores)*0.01, f'{v:.2f}', ha='center', fontweight='bold')

        axes[1,0].bar(horizons, mape_scores, color='gold', alpha=0.8)
        axes[1,0].set_title('MAPE by Horizon')
        axes[1,0].set_ylabel('MAPE (%)')
        for i, v in enumerate(mape_scores):
            axes[1,0].text(i, v + max(mape_scores)*0.01, f'{v:.1f}', ha='center', fontweight='bold')

        axes[1,1].plot(horizons, r2_scores, marker='o', linewidth=2, markersize=8, label='R² Score')
        axes[1,1].set_title('Performance Degradation Trend')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        kernel_counts = {}
        for kernel in best_kernels:
            kernel_counts[kernel] = kernel_counts.get(kernel, 0) + 1

        axes[1,2].pie(kernel_counts.values(), labels=kernel_counts.keys(), autopct='%1.0f%%',
                      startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen', 'gold'][:len(kernel_counts)])
        axes[1,2].set_title('Best Kernel Distribution')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'comprehensive_performance_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        if all_kernel_results:
            n_horizons = len([h for h in self.horizons if h in all_kernel_results])
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('SVR Kernel Comparison Analysis', fontsize=16, fontweight='bold')

            plot_idx = 0
            for horizon in self.horizons[:6]:
                if horizon in all_kernel_results:
                    row = plot_idx // 3
                    col = plot_idx % 3

                    kernel_names = list(all_kernel_results[horizon].keys())
                    r2_values = [all_kernel_results[horizon][k]['val_r2'] for k in kernel_names]

                    bars = axes[row, col].bar(kernel_names, r2_values, alpha=0.8)
                    axes[row, col].set_title(f'{horizon}h Horizon Kernel Comparison')
                    axes[row, col].set_ylabel('Validation R²')
                    axes[row, col].set_ylim(0, 1)

                    best_idx = r2_values.index(max(r2_values))
                    bars[best_idx].set_color('gold')

                    for bar, v in zip(bars, r2_values):
                        axes[row, col].text(bar.get_x() + bar.get_width()/2., v + 0.01,
                                            f'{v:.3f}', ha='center', fontweight='bold')
                    plot_idx += 1

            for idx in range(plot_idx, 6):
                row = idx // 3
                col = idx % 3
                fig.delaxes(axes[row, col])

            plt.tight_layout()
            plt.savefig(self.results_dir / f'kernel_comparison_analysis_{timestamp}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('SVR Time Series Forecasts', fontsize=16, fontweight='bold')

        for idx, horizon in enumerate(self.horizons):
            if horizon in results and idx < 6:
                row = idx // 2
                col = idx % 2

                metrics = results[horizon]['metrics']
                predictions = metrics['predictions']
                actuals = metrics['actuals']

                sample_size = min(500, len(predictions))
                sample_pred = predictions[:sample_size]
                sample_actual = actuals[:sample_size]

                axes[row, col].plot(sample_actual, label='Actual', alpha=0.8, linewidth=1)
                axes[row, col].plot(sample_pred, label='Predicted', alpha=0.8, linewidth=1)
                axes[row, col].set_title(f'{horizon}h Horizon (R²={metrics["r2_score"]:.3f})')
                axes[row, col].set_xlabel('Time Steps')
                axes[row, col].set_ylabel('Heat Demand (MWh)')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        if len(self.horizons) < 6:
            for idx in range(len(self.horizons), 6):
                row = idx // 2
                col = idx % 2
                fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.savefig(self.results_dir / f'time_series_forecasts_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('SVR Prediction Accuracy Analysis', fontsize=16, fontweight='bold')

        for idx, horizon in enumerate(self.horizons[:6]):
            if horizon in results:
                row = idx // 3
                col = idx % 3

                metrics = results[horizon]['metrics']
                predictions = metrics['predictions']
                actuals = metrics['actuals']

                axes[row, col].scatter(actuals, predictions, alpha=0.6, s=10)

                min_val = min(np.min(actuals), np.min(predictions))
                max_val = max(np.max(actuals), np.max(predictions))
                axes[row, col].plot([min_val, max_val], [min_val, max_val],
                                    'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')

                axes[row, col].set_title(f'{horizon}h Horizon\nR²={metrics["r2_score"]:.3f}, RMSE={metrics["rmse"]:.2f}')
                axes[row, col].set_xlabel('Actual Heat Demand (MWh)')
                axes[row, col].set_ylabel('Predicted Heat Demand (MWh)')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        for idx in range(len(self.horizons), 6):
            if idx < 6:
                row = idx // 3
                col = idx % 3
                fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.savefig(self.results_dir / f'prediction_accuracy_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SVR Hyperparameter Analysis', fontsize=16, fontweight='bold')

        c_values = []
        gamma_values = []
        epsilon_values = []
        horizons_hp = []

        for horizon in self.horizons:
            if horizon in results:
                model = results[horizon]['model']
                horizons_hp.append(f"{horizon}h")
                c_values.append(model.C)

                if hasattr(model, 'gamma'):
                    if isinstance(model.gamma, str):
                        gamma_values.append(0.1)
                    else:
                        gamma_values.append(model.gamma)
                else:
                    gamma_values.append(0)

                epsilon_values.append(model.epsilon)

        axes[0,0].plot(horizons_hp, c_values, marker='o', linewidth=2, markersize=8)
        axes[0,0].set_title('C Parameter by Horizon')
        axes[0,0].set_ylabel('C Value')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')

        axes[0,1].plot(horizons_hp, gamma_values, marker='s', linewidth=2, markersize=8, color='orange')
        axes[0,1].set_title('Gamma Parameter by Horizon')
        axes[0,1].set_ylabel('Gamma Value')
        axes[0,1].grid(True, alpha=0.3)

        axes[1,0].plot(horizons_hp, epsilon_values, marker='^', linewidth=2, markersize=8, color='green')
        axes[1,0].set_title('Epsilon Parameter by Horizon')
        axes[1,0].set_ylabel('Epsilon Value')
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].scatter(c_values, r2_scores[:len(c_values)], s=100, alpha=0.7, label='C vs R²')
        axes[1,1].set_title('Hyperparameter vs Performance')
        axes[1,1].set_xlabel('C Value (log scale)')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].set_xscale('log')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / f'hyperparameter_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"All visualizations saved to {self.results_dir}")

    def run_fast_svr(self) -> Dict[str, Any]:
        print("Starting Fast SVR Implementation")
        print("="*50)

        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        df = self.load_data()
        X_df, feature_names = self.prepare_features(df)
        y_df = self.create_targets(df)

        valid_idx = ~(X_df.isna().any(axis=1) | y_df.isna().any(axis=1))
        X_df = X_df[valid_idx]
        y_df = y_df[valid_idx]

        print(f"Final dataset shape: {X_df.shape}")
        print(f"Features: {len(feature_names)}")

        n_train = int(len(X_df) * 0.7)
        n_val = int(len(X_df) * 0.15)

        X_train = X_df.iloc[:n_train].values
        X_val = X_df.iloc[n_train:n_train+n_val].values
        X_test = X_df.iloc[n_train+n_val:].values

        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        results = {}
        all_kernel_results = {}

        for horizon in self.horizons:
            print(f"\nProcessing {horizon}h horizon")
            print("-" * 30)

            y_train = y_df.iloc[:n_train][f'target_{horizon}h'].dropna().values
            y_val = y_df.iloc[n_train:n_train+n_val][f'target_{horizon}h'].dropna().values
            y_test = y_df.iloc[n_train+n_val:][f'target_{horizon}h'].dropna().values

            min_len = min(len(X_train_scaled), len(y_train))
            X_train_h = X_train_scaled[:min_len]
            y_train_h = y_train[:min_len]

            min_len = min(len(X_val_scaled), len(y_val))
            X_val_h = X_val_scaled[:min_len]
            y_val_h = y_val[:min_len]

            min_len = min(len(X_test_scaled), len(y_test))
            X_test_h = X_test_scaled[:min_len]
            y_test_h = y_test[:min_len]

            if len(y_train_h) < 50:
                print(f"Insufficient data for {horizon}h horizon")
                continue

            kernel_results = self.train_kernel_comparison(
                X_train_h, y_train_h, X_val_h, y_val_h, horizon
            )

            if not kernel_results:
                print(f"No successful models for {horizon}h horizon")
                continue

            all_kernel_results[horizon] = kernel_results

            best_model, best_kernel = self.select_best_model(kernel_results, horizon)

            final_metrics = self.evaluate_model(best_model, X_test_h, y_test_h, horizon)

            results[horizon] = {
                'model': best_model,
                'model_type': f"SVR ({best_kernel})",
                'metrics': final_metrics,
                'features': feature_names,
                'scaler': scaler,
                'kernel_results': kernel_results
            }

            metrics = final_metrics
            print(f"Results for {horizon}h:")
            print(f"  R2 Score: {metrics['r2_score']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")

        end_time = datetime.now()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time}")

        self.create_visualizations(results, all_kernel_results, timestamp)

        return results, timestamp

    def save_results(self, results: Dict[str, Any], timestamp: str):
        print(f"Saving results to {self.results_dir}...")

        save_results = {}
        for horizon, result in results.items():
            save_results[horizon] = {
                'model_type': result['model_type'],
                'metrics': {k: v for k, v in result['metrics'].items() if k not in ['predictions', 'actuals']},
                'features': result['features'],
                'kernel_results': {k: {kk: vv for kk, vv in v.items() if kk not in ['model', 'predictions', 'actuals']}
                                   for k, v in result['kernel_results'].items()}
            }

        results_file = self.results_dir / f"svr_enhanced_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(save_results, f)

        for horizon, result in results.items():
            model_file = self.results_dir / f"svr_model_{horizon}h_{timestamp}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)

        if results:
            scaler_file = self.results_dir / f"svr_scaler_{timestamp}.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(results[list(results.keys())[0]]['scaler'], f)

        predictions_data = {}
        for horizon, result in results.items():
            metrics = result['metrics']
            predictions_data[f'{horizon}h_predictions'] = metrics['predictions'].tolist()
            predictions_data[f'{horizon}h_actuals'] = metrics['actuals'].tolist()

        predictions_file = self.results_dir / f"svr_predictions_{timestamp}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        summary_data = []
        for horizon, result in results.items():
            metrics = result['metrics']
            model = result['model']
            summary_data.append({
                'timestamp': timestamp,
                'horizon': horizon,
                'model_type': result['model_type'],
                'kernel': model.kernel,
                'C': model.C,
                'gamma': getattr(model, 'gamma', 'N/A'),
                'epsilon': model.epsilon,
                'features_count': len(result['features']),
                'r2_score': metrics['r2_score'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / f"svr_enhanced_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        kernel_comparison_data = []
        for horizon, result in results.items():
            for kernel, kernel_result in result['kernel_results'].items():
                kernel_comparison_data.append({
                    'horizon': horizon,
                    'kernel': kernel,
                    'val_r2': kernel_result['val_r2'],
                    'val_rmse': kernel_result['val_rmse'],
                    'params': str(kernel_result['params'])
                })

        kernel_df = pd.DataFrame(kernel_comparison_data)
        kernel_file = self.results_dir / f"svr_kernel_comparison_{timestamp}.csv"
        kernel_df.to_csv(kernel_file, index=False)

        print(f"Results saved successfully!")
        print(f"- Main results: svr_enhanced_results_{timestamp}.pkl")
        print(f"- Model states: svr_model_[horizon]h_{timestamp}.pkl")
        print(f"- Summary: svr_enhanced_summary_{timestamp}.csv")
        print(f"- Predictions: svr_predictions_{timestamp}.json")
        print(f"- Kernel comparison: svr_kernel_comparison_{timestamp}.csv")
        print(f"- Visualizations: [multiple PNG files]")

    def print_final_summary(self, results: Dict[str, Any]):
        print("\nSVR RESULTS SUMMARY (33 Features)")
        print("=" * 50)

        print(f"{'Horizon':>8} {'Kernel':>8} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
        print("-" * 56)

        for horizon in self.horizons:
            if horizon in results:
                metrics = results[horizon]['metrics']
                kernel = results[horizon]['model'].kernel
                print(f"{horizon:>7}h {kernel:>8} {metrics['r2_score']:>7.3f} {metrics['rmse']:>7.3f} {metrics['mae']:>7.3f} {metrics['mape']:>7.1f}")

    def compare_with_baselines(self, results):
        print("\nCOMPARISON WITH BASELINES")
        print("=" * 50)

        ffnn_baseline = {1: 0.988, 6: 0.943, 24: 0.861, 48: 0.800, 72: 0.743}

        print(f"{'Horizon':>8} {'FFNN-33':>8} {'SVR-33':>8} {'Improvement':>12}")
        print("-" * 40)

        total_improvement = 0
        valid_comparisons = 0

        for horizon in self.horizons:
            if horizon in results and horizon in ffnn_baseline:
                ffnn_score = ffnn_baseline[horizon]
                svr_score = results[horizon]['metrics']['r2_score']
                improvement = ((svr_score - ffnn_score) / ffnn_score) * 100
                total_improvement += improvement
                valid_comparisons += 1

                print(f"{horizon:>7}h {ffnn_score:>7.3f} {svr_score:>7.3f} {improvement:>+10.1f}%")

        if valid_comparisons > 0:
            avg_improvement = total_improvement / valid_comparisons
            print("-" * 40)
            print(f"Average improvement: {avg_improvement:+.1f}%")


def main():
    print("SVR Implementation - HPC Execution (33 Features)")
    print("Schweinfurt District Heating Network Forecasting")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    svr_system = FastSVRForecasting(fast_mode=True)

    start_time = datetime.now()

    results, timestamp = svr_system.run_fast_svr()
    svr_system.save_results(results, timestamp)
    svr_system.print_final_summary(results)
    svr_system.compare_with_baselines(results)

    end_time = datetime.now()
    execution_time = end_time - start_time

    print(f"\nSVR Implementation Complete!")
    print(f"Total Execution Time: {execution_time}")
    print(f"Results saved in: {svr_system.results_dir}")
    print(f"Timestamp: {timestamp}")

if __name__ == "__main__":
    main()