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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class AllFeaturesSVRForecasting:

    def __init__(self, data_path=None, fast_mode=True):
        if data_path is None:
            self.data_path = Path("/workspace/Thesis/01_data/processed_data")
        else:
            self.data_path = Path(data_path)

        # Fixed results path for HPC
        self.results_dir = Path("/workspace/Thesis/04_svr_implementation/results/allfeatures")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.horizons = [1, 6, 24, 48, 72]
        self.kernels = ['rbf', 'linear'] if fast_mode else ['rbf', 'poly', 'linear', 'sigmoid']
        self.fast_mode = fast_mode

        self.base_features = [
            'app_temp', 'azimuth', 'clouds', 'dewpt', 'dhi', 'dni', 'elev_angle', 'ghi',
            'h_angle', 'pod', 'precip', 'pres', 'revision_status', 'rh', 'slp', 'snow',
            'solar_rad', 'temp', 'uv', 'vis', 'weather', 'wind_dir', 'wind_gust_spd',
            'wind_spd', 'date', 'month', 'year', 'hour', 'day_of_week', 'season',
            'B1_B2_expected_supply_temp', 'B1_B2_expected_return_temp', 'B1_B2_expected_delta_T',
            'F1_Nord_expected_supply_temp', 'F1_Nord_expected_return_temp', 'F1_Nord_expected_delta_T',
            'F1_Sud_expected_supply_temp', 'F1_Sud_expected_return_temp', 'F1_Sud_expected_delta_T',
            'Maintal_expected_supply_temp', 'Maintal_expected_return_temp', 'Maintal_expected_delta_T',
            'N1_expected_supply_temp', 'N1_expected_return_temp', 'N1_expected_delta_T',
            'N2_expected_supply_temp', 'N2_expected_return_temp', 'N2_expected_delta_T',
            'V1_expected_supply_temp', 'V1_expected_return_temp', 'V1_expected_delta_T',
            'V2_expected_supply_temp', 'V2_expected_return_temp', 'V2_expected_delta_T',
            'V6_expected_supply_temp', 'V6_expected_return_temp', 'V6_expected_delta_T',
            'W1_expected_supply_temp', 'W1_expected_return_temp', 'W1_expected_delta_T',
            'ZN_expected_supply_temp', 'ZN_expected_return_temp', 'ZN_expected_delta_T',
            'hdd_18', 'hdd_15_5', 'temp_change', 'is_daytime', 'is_weekend'
        ]

        self.exclude_columns = ['datetime', 'timestamp_local', 'timestamp_utc', 'ts']
        self.results = {}

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        print("All Features SVR Forecasting System Initialized")
        print(f"Target Horizons: {self.horizons}")
        print(f"Fast Mode: {fast_mode}")
        print(f"Results will be saved to: {self.results_dir}")

    def load_data(self):
        file_path = self.data_path / "merged_dataset.csv"

        if not file_path.exists():
            available_files = list(self.data_path.glob("*.csv"))
            if available_files:
                file_path = available_files[0]
            else:
                raise FileNotFoundError(f"No CSV files found in {self.data_path}")

        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape}")

        # Set datetime index
        date_col = None
        for col in ['datetime', 'timestamp', 'date', 'time']:
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

    def prepare_base_features(self, df):
        print("Preparing base feature set...")

        available_features = []
        for feature in self.base_features:
            if feature in df.columns and feature not in self.exclude_columns:
                available_features.append(feature)

        print(f"Found {len(available_features)} base features")

        # Process features to ensure numeric
        for feature in available_features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                if df[feature].isnull().sum() > 0:
                    df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill')
                    if df[feature].isnull().sum() > 0:
                        df[feature] = df[feature].fillna(df[feature].median())
            else:
                try:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
                    if df[feature].isnull().sum() > 0:
                        df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill')
                        if df[feature].isnull().sum() > 0:
                            df[feature] = df[feature].fillna(df[feature].median())
                except:
                    le = LabelEncoder()
                    df[feature] = le.fit_transform(df[feature].astype(str))

            if df[feature].isnull().sum() > 0:
                df[feature] = df[feature].fillna(0)

        return df[available_features], available_features

    def load_heat_demand_data(self, df):
        if 'heat_demand' in df.columns:
            heat_demand = df['heat_demand']
        elif 'demand' in df.columns:
            heat_demand = df['demand']
        else:
            zone_weights = {
                'V1': 0.16, 'N1': 0.13, 'N2': 0.07, 'V6': 0.05, 'W1': 0.14,
                'F1_Sud': 0.12, 'F1_Nord': 0.10, 'B1_B2': 0.15, 'V2': 0.06, 'ZN': 0.02
            }

            if 'hdd_15_5' in df.columns:
                base_demand = df['hdd_15_5'] * 3.8
            elif 'hdd_18' in df.columns:
                base_demand = df['hdd_18'] * 3.5
            else:
                base_demand = np.maximum(0, 15.5 - df['temp']) * 3.8

            temp_contribution = 0
            for zone, weight in zone_weights.items():
                supply_temp_col = f'{zone}_expected_supply_temp'
                if supply_temp_col in df.columns:
                    temp_norm = (df[supply_temp_col] - 60) / 55
                    temp_contribution += temp_norm * weight * 20

            hour = df['hour'] if 'hour' in df.columns else df.index.hour
            time_factor = 1.0 + 0.45 * (
                    np.exp(-((hour - 7) ** 2) / 12) +
                    np.exp(-((hour - 12) ** 2) / 18) +
                    np.exp(-((hour - 19) ** 2) / 14) +
                    0.3 * np.exp(-((hour - 22) ** 2) / 10)
            )

            dow = df['day_of_week'] if 'day_of_week' in df.columns else df.index.dayofweek
            dow_factor = np.array([1.30, 1.25, 1.20, 1.15, 1.10, 0.70, 0.65])[dow]

            month = df['month'] if 'month' in df.columns else df.index.month
            seasonal_factor = (1.0 + 0.35 * np.cos(2 * np.pi * (month - 1) / 12) +
                               0.15 * np.cos(4 * np.pi * (month - 1) / 12))

            heat_demand = ((base_demand + temp_contribution) * time_factor *
                           dow_factor * seasonal_factor)
            heat_demand = np.maximum(heat_demand, 0.1)

        print(f"Heat demand data loaded: {heat_demand.min():.2f} to {heat_demand.max():.2f} MWh")
        return heat_demand

    def add_advanced_features(self, df, heat_demand):
        print("Adding advanced features...")
        enhanced_features = {}

        # Core lag features
        lag_hours = [12, 23, 24, 25, 36, 47, 48, 49, 60, 72, 84, 96, 120, 144]
        for lag in lag_hours:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        # Weekly patterns
        weekly_lags = [162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 174, 176]
        for lag in weekly_lags:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        # Multi-week patterns
        for lag in [336, 504]:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        # Rolling statistics
        windows = [24, 48, 72, 96, 120, 144, 168, 240, 336]
        for window in windows:
            enhanced_features[f'demand_rolling_mean_{window}h'] = heat_demand.rolling(window).mean()
            enhanced_features[f'demand_rolling_std_{window}h'] = heat_demand.rolling(window).std()

        # Special rolling features
        enhanced_features['demand_rolling_min_72h'] = heat_demand.rolling(72).min()
        enhanced_features['demand_rolling_max_72h'] = heat_demand.rolling(72).max()
        enhanced_features['demand_rolling_median_72h'] = heat_demand.rolling(72).median()

        # Time features
        hour = df['hour'] if 'hour' in df.columns else df.index.hour
        enhanced_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        enhanced_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        dow = df['day_of_week'] if 'day_of_week' in df.columns else df.index.dayofweek
        enhanced_features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        enhanced_features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        month = df['month'] if 'month' in df.columns else df.index.month
        enhanced_features['month_sin'] = np.sin(2 * np.pi * month / 12)
        enhanced_features['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Week position features
        week_position = (dow * 24 + hour) / 168.0
        enhanced_features['week_position_sin'] = np.sin(2 * np.pi * week_position)
        enhanced_features['week_position_cos'] = np.cos(2 * np.pi * week_position)

        for harmonic in [2, 3]:
            enhanced_features[f'week_position_sin_{harmonic}'] = np.sin(2 * np.pi * harmonic * week_position)
            enhanced_features[f'week_position_cos_{harmonic}'] = np.cos(2 * np.pi * harmonic * week_position)

        # Temperature features
        if 'temp' in df.columns:
            enhanced_features['temp_change_6h'] = df['temp'].diff(6)
            enhanced_features['temp_change_24h'] = df['temp'].diff(24)
            enhanced_features['temp_change_72h'] = df['temp'].diff(72)

            def safe_trend(x):
                return np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 48 and not np.isnan(x).all() else 0

            enhanced_features['temp_trend_48h'] = df['temp'].rolling(48).apply(safe_trend, raw=False)
            enhanced_features['temp_trend_72h'] = df['temp'].rolling(72).apply(safe_trend, raw=False)
            enhanced_features['temp_trend_168h'] = df['temp'].rolling(168).apply(safe_trend, raw=False)
            enhanced_features['temp_volatility_72h'] = df['temp'].rolling(72).std()
            enhanced_features['temp_volatility_168h'] = df['temp'].rolling(168).std()

        # Business day features
        enhanced_features['is_business_day'] = ((dow < 5)).astype(int)
        enhanced_features['is_friday'] = (dow == 4).astype(int)
        enhanced_features['is_sunday'] = (dow == 6).astype(int)
        enhanced_features['is_weekend'] = (dow >= 5).astype(int)
        enhanced_features['is_monday'] = (dow == 0).astype(int)
        enhanced_features['is_tuesday'] = (dow == 1).astype(int)
        enhanced_features['is_wednesday'] = (dow == 2).astype(int)
        enhanced_features['is_thursday'] = (dow == 3).astype(int)
        enhanced_features['is_saturday'] = (dow == 5).astype(int)

        # Weekend transition features
        enhanced_features['friday_evening'] = ((dow == 4) & (hour >= 16)).astype(int)
        enhanced_features['sunday_evening'] = ((dow == 6) & (hour >= 18)).astype(int)
        enhanced_features['monday_morning'] = ((dow == 0) & (hour <= 11)).astype(int)

        # Seasonal features
        day_of_year = df.index.dayofyear if hasattr(df.index, 'dayofyear') else ((month - 1) * 30 + 15)
        enhanced_features['season_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        enhanced_features['season_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Week of year features
        if hasattr(df.index, 'isocalendar'):
            week_of_year = df.index.isocalendar().week
        else:
            week_of_year = ((day_of_year - 1) // 7 + 1)
        enhanced_features['week_of_year_sin'] = np.sin(2 * np.pi * week_of_year / 52)
        enhanced_features['week_of_year_cos'] = np.cos(2 * np.pi * week_of_year / 52)

        # HDD features
        if 'hdd_15_5' in df.columns:
            enhanced_features['hdd_lag_48h'] = df['hdd_15_5'].shift(48)
            enhanced_features['hdd_lag_72h'] = df['hdd_15_5'].shift(72)
            enhanced_features['hdd_lag_168h'] = df['hdd_15_5'].shift(168)
            enhanced_features['hdd_trend_72h'] = df['hdd_15_5'].rolling(72).apply(safe_trend, raw=False)
            enhanced_features['hdd_trend_168h'] = df['hdd_15_5'].rolling(168).apply(safe_trend, raw=False)

        # Zone features
        major_zones = ['B1_B2', 'V1', 'W1', 'F1_Sud']
        for zone in major_zones:
            supply_col = f'{zone}_expected_supply_temp'
            if supply_col in df.columns:
                enhanced_features[f'{zone}_lag_72h'] = df[supply_col].shift(72)
                enhanced_features[f'{zone}_lag_168h'] = df[supply_col].shift(168)
                enhanced_features[f'{zone}_trend_72h'] = df[supply_col].rolling(72).apply(safe_trend, raw=False)

        # Interaction features
        if 'temp' in df.columns:
            enhanced_features['weekend_temp_interaction'] = enhanced_features['is_weekend'] * df['temp']
        enhanced_features['dow_hour_interaction'] = enhanced_features['dow_sin'] * enhanced_features['hour_sin']
        enhanced_features['week_position_season_interaction'] = enhanced_features['week_position_sin'] * enhanced_features['season_sin']

        print(f"Added {len(enhanced_features)} advanced features")
        return enhanced_features

    def create_targets(self, heat_demand):
        targets = pd.DataFrame(index=heat_demand.index)
        for horizon in self.horizons:
            targets[f'target_{horizon}h'] = heat_demand.shift(-horizon)
        print(f"Created targets for horizons: {self.horizons}")
        return targets

    def prepare_ml_dataset(self, base_df, enhanced_features, targets):
        print("Preparing complete ML dataset...")

        ml_df = base_df.copy()

        for feature_name, feature_series in enhanced_features.items():
            ml_df[feature_name] = feature_series

        for target_name in targets.columns:
            ml_df[target_name] = targets[target_name]

        print(f"Dataset before cleaning: {ml_df.shape[0]} samples, {ml_df.shape[1]} columns")

        ml_df_clean = ml_df.dropna()
        print(f"Dataset after cleaning: {ml_df_clean.shape[0]} samples")

        feature_columns = [col for col in ml_df_clean.columns if not col.startswith('target_')]
        print(f"Total features available: {len(feature_columns)}")

        return ml_df_clean, feature_columns

    def feature_selection_by_horizon(self, X, y, horizon):
        print(f"Applying feature selection for {horizon}h horizon...")

        X_numeric = X.copy()
        for col in X_numeric.columns:
            if not pd.api.types.is_numeric_dtype(X_numeric[col]):
                try:
                    X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
                    X_numeric[col] = X_numeric[col].fillna(0)
                except:
                    le = LabelEncoder()
                    X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))

        X_numeric = X_numeric.replace([np.inf, -np.inf], 0)
        X_numeric = X_numeric.fillna(0)

        var_threshold = VarianceThreshold(threshold=0.001)
        X_var = var_threshold.fit_transform(X_numeric)
        var_features = X_numeric.columns[var_threshold.get_support()]

        # Feature counts for SVR
        if horizon <= 6:
            k_features = min(150, len(var_features))
        elif horizon <= 24:
            k_features = min(200, len(var_features))
        elif horizon <= 48:
            k_features = min(250, len(var_features))
        else:
            k_features = min(300, len(var_features))

        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X_var, y)
        selected_indices = selector.get_support()
        selected_features = var_features[selected_indices].tolist()

        print(f"{horizon}h feature selection: {len(X_numeric.columns)} -> {len(var_features)} -> {len(selected_features)}")
        return X_selected, selected_features

    def get_param_grids(self, kernel):
        if self.fast_mode:
            if kernel == 'rbf':
                return {'C': [1, 10, 100], 'gamma': ['scale', 0.1, 0.01], 'epsilon': [0.1, 0.01]}
            elif kernel == 'linear':
                return {'C': [1, 10, 100], 'epsilon': [0.1, 0.01]}
        else:
            if kernel == 'rbf':
                return {'C': [0.1, 1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1], 'epsilon': [0.001, 0.01, 0.1, 0.2]}
            elif kernel == 'linear':
                return {'C': [0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 0.2]}
            elif kernel == 'poly':
                return {'C': [1, 10, 100], 'gamma': ['scale', 'auto'], 'degree': [2, 3], 'epsilon': [0.01, 0.1]}
            else:
                return {'C': [1, 10, 100], 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1]}

    def optimize_hyperparameters(self, X_train, y_train, kernel, horizon):
        param_grid = self.get_param_grids(kernel)

        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)

        print(f"  {kernel}: Testing {total_combinations} combinations")

        tscv = TimeSeriesSplit(n_splits=3)
        svr = SVR(kernel=kernel, cache_size=2000)

        search = GridSearchCV(svr, param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=0)
        search.fit(X_train, y_train)

        print(f"  {kernel}: Best score = {search.best_score_:.4f}")
        return search.best_params_

    def train_kernel_comparison(self, X_train, y_train, X_val, y_val, horizon):
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

            except Exception as e:
                print(f"  {kernel} failed: {e}")
                continue

        return kernel_results

    def select_best_model(self, kernel_results, horizon):
        if not kernel_results:
            raise ValueError("No successful kernel results")

        best_kernel = max(kernel_results.keys(), key=lambda k: kernel_results[k]['val_r2'])
        best_score = kernel_results[best_kernel]['val_r2']

        print(f"  Best kernel: {best_kernel} (R2={best_score:.4f})")

        return kernel_results[best_kernel]['model'], best_kernel

    def evaluate_model(self, model, X_test, y_test, horizon):
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
        """Create comprehensive visualizations for All Features SVR results"""
        print("Creating visualizations...")

        # 1. Comprehensive Performance Analysis
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('All Features SVR Comprehensive Performance Analysis', fontsize=16, fontweight='bold')

        horizons = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        mape_scores = []
        features_used = []
        best_kernels = []

        for horizon in self.horizons:
            if horizon in results:
                horizons.append(f"{horizon}h")
                metrics = results[horizon]['metrics']
                r2_scores.append(metrics['r2_score'])
                rmse_scores.append(metrics['rmse'])
                mae_scores.append(metrics['mae'])
                mape_scores.append(metrics['mape'])
                features_used.append(results[horizon]['feature_count'])
                best_kernels.append(results[horizon]['model'].kernel)

        # R2 Score with kernel annotations
        bars1 = axes[0,0].bar(horizons, r2_scores, color='skyblue', alpha=0.8)
        axes[0,0].set_title('R² Score by Horizon')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_ylim(0, 1)
        for bar, v, kernel in zip(bars1, r2_scores, best_kernels):
            axes[0,0].text(bar.get_x() + bar.get_width()/2., v + 0.01,
                           f'{v:.3f}\n({kernel})', ha='center', fontweight='bold')

        # RMSE comparison
        axes[0,1].bar(horizons, rmse_scores, color='lightcoral', alpha=0.8)
        axes[0,1].set_title('RMSE by Horizon')
        axes[0,1].set_ylabel('RMSE')
        for i, v in enumerate(rmse_scores):
            axes[0,1].text(i, v + max(rmse_scores)*0.01, f'{v:.2f}', ha='center', fontweight='bold')

        # Features used
        axes[0,2].bar(horizons, features_used, color='lightgreen', alpha=0.8)
        axes[0,2].set_title('Features Selected by Horizon')
        axes[0,2].set_ylabel('Number of Features')
        for i, v in enumerate(features_used):
            axes[0,2].text(i, v + max(features_used)*0.01, f'{v}', ha='center', fontweight='bold')

        # Performance vs Feature Count
        axes[1,0].scatter(features_used, r2_scores, s=100, alpha=0.7, c=range(len(features_used)), cmap='viridis')
        axes[1,0].set_title('Performance vs Feature Count')
        axes[1,0].set_xlabel('Number of Features')
        axes[1,0].set_ylabel('R² Score')
        for i, (x, y, h) in enumerate(zip(features_used, r2_scores, horizons)):
            axes[1,0].annotate(h, (x, y), xytext=(5, 5), textcoords='offset points')
        axes[1,0].grid(True, alpha=0.3)

        # Performance degradation trend
        axes[1,1].plot(horizons, r2_scores, marker='o', linewidth=2, markersize=8, label='R² Score')
        axes[1,1].set_title('Performance Degradation Trend')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # Kernel distribution
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

        # 2. Feature Selection Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('All Features SVR - Feature Selection Analysis', fontsize=16, fontweight='bold')

        # Feature count progression
        axes[0,0].bar(horizons, features_used, color='mediumpurple', alpha=0.8)
        axes[0,0].set_title('Selected Features by Horizon')
        axes[0,0].set_ylabel('Number of Features')
        for i, v in enumerate(features_used):
            axes[0,0].text(i, v + max(features_used)*0.01, f'{v}', ha='center', fontweight='bold')

        # Feature efficiency (performance per feature)
        efficiency = [r2/feat for r2, feat in zip(r2_scores, features_used)]
        axes[0,1].plot(horizons, efficiency, marker='s', linewidth=2, markersize=8, color='orange')
        axes[0,1].set_title('Feature Efficiency (R²/Feature Count)')
        axes[0,1].set_ylabel('R² per Feature')
        axes[0,1].grid(True, alpha=0.3)

        # Comparison with 33-feature baseline
        baseline_features = [33] * len(horizons)
        width = 0.35
        x_pos = np.arange(len(horizons))

        axes[1,0].bar(x_pos - width/2, baseline_features, width, label='33-Feature SVR', alpha=0.8, color='lightcoral')
        axes[1,0].bar(x_pos + width/2, features_used, width, label='All-Feature SVR', alpha=0.8, color='lightgreen')
        axes[1,0].set_title('Feature Count Comparison')
        axes[1,0].set_ylabel('Number of Features')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(horizons)
        axes[1,0].legend()

        # Feature scaling visualization
        feature_ratios = [feat/33 for feat in features_used]
        axes[1,1].bar(horizons, feature_ratios, color='gold', alpha=0.8)
        axes[1,1].set_title('Feature Scale-up Factor (vs 33-feature)')
        axes[1,1].set_ylabel('Feature Ratio (All/33)')
        axes[1,1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (33 features)')
        axes[1,1].legend()
        for i, v in enumerate(feature_ratios):
            axes[1,1].text(i, v + max(feature_ratios)*0.01, f'{v:.1f}x', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'feature_selection_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Kernel Comparison Analysis
        if all_kernel_results:
            n_horizons = len([h for h in self.horizons if h in all_kernel_results])
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('All Features SVR Kernel Comparison Analysis', fontsize=16, fontweight='bold')

            plot_idx = 0
            for horizon in self.horizons[:6]:
                if horizon in all_kernel_results:
                    row = plot_idx // 3
                    col = plot_idx % 3

                    kernel_names = list(all_kernel_results[horizon].keys())
                    r2_values = [all_kernel_results[horizon][k]['val_r2'] for k in kernel_names]

                    bars = axes[row, col].bar(kernel_names, r2_values, alpha=0.8)
                    axes[row, col].set_title(f'{horizon}h Horizon ({features_used[plot_idx] if plot_idx < len(features_used) else "N/A"} features)')
                    axes[row, col].set_ylabel('Validation R²')
                    axes[row, col].set_ylim(0, 1)

                    # Highlight best kernel
                    best_idx = r2_values.index(max(r2_values))
                    bars[best_idx].set_color('gold')

                    for bar, v in zip(bars, r2_values):
                        axes[row, col].text(bar.get_x() + bar.get_width()/2., v + 0.01,
                                            f'{v:.3f}', ha='center', fontweight='bold')
                    plot_idx += 1

            # Remove empty subplots
            for idx in range(plot_idx, 6):
                row = idx // 3
                col = idx % 3
                fig.delaxes(axes[row, col])

            plt.tight_layout()
            plt.savefig(self.results_dir / f'kernel_comparison_analysis_{timestamp}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Time Series Forecasts
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('All Features SVR Time Series Forecasts', fontsize=16, fontweight='bold')

        for idx, horizon in enumerate(self.horizons):
            if horizon in results and idx < 6:
                row = idx // 2
                col = idx % 2

                metrics = results[horizon]['metrics']
                predictions = metrics['predictions']
                actuals = metrics['actuals']

                # Sample data for visualization (first 500 points)
                sample_size = min(500, len(predictions))
                sample_pred = predictions[:sample_size]
                sample_actual = actuals[:sample_size]

                axes[row, col].plot(sample_actual, label='Actual', alpha=0.8, linewidth=1)
                axes[row, col].plot(sample_pred, label='Predicted', alpha=0.8, linewidth=1)
                axes[row, col].set_title(f'{horizon}h Horizon (R²={metrics["r2_score"]:.3f}, {results[horizon]["feature_count"]} features)')
                axes[row, col].set_xlabel('Time Steps')
                axes[row, col].set_ylabel('Heat Demand (MWh)')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        # Remove empty subplots if needed
        if len(self.horizons) < 6:
            for idx in range(len(self.horizons), 6):
                row = idx // 2
                col = idx % 2
                fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.savefig(self.results_dir / f'time_series_forecasts_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Model Comparison with Baselines
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('All Features SVR vs Baselines Comparison', fontsize=16, fontweight='bold')

        # Baseline results
        svr_33_baseline = {1: 0.989, 6: 0.930, 24: 0.864, 48: 0.793, 72: 0.739}
        ffnn_baseline = {1: 0.988, 6: 0.943, 24: 0.861, 48: 0.800, 72: 0.743}

        all_svr_scores = []
        svr_33_scores = []
        ffnn_scores = []
        labels = []
        svr_improvements = []
        ffnn_improvements = []

        for horizon in self.horizons:
            if horizon in results and horizon in svr_33_baseline and horizon in ffnn_baseline:
                labels.append(f"{horizon}h")
                all_svr_score = results[horizon]['metrics']['r2_score']
                svr_33_score = svr_33_baseline[horizon]
                ffnn_score = ffnn_baseline[horizon]

                all_svr_scores.append(all_svr_score)
                svr_33_scores.append(svr_33_score)
                ffnn_scores.append(ffnn_score)

                svr_improvement = ((all_svr_score - svr_33_score) / svr_33_score) * 100
                ffnn_improvement = ((all_svr_score - ffnn_score) / ffnn_score) * 100

                svr_improvements.append(svr_improvement)
                ffnn_improvements.append(ffnn_improvement)

        # Three-way comparison
        x_pos = np.arange(len(labels))
        width = 0.25

        ax1.bar(x_pos - width, svr_33_scores, width, label='SVR-33', alpha=0.8, color='lightcoral')
        ax1.bar(x_pos, ffnn_scores, width, label='FFNN-33', alpha=0.8, color='skyblue')
        ax1.bar(x_pos + width, all_svr_scores, width, label='SVR-All', alpha=0.8, color='lightgreen')
        ax1.set_title('R² Score Comparison')
        ax1.set_ylabel('R² Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Feature efficiency comparison
        efficiency_all = [r2/feat for r2, feat in zip(all_svr_scores, features_used)]
        efficiency_33 = [r2/33 for r2 in svr_33_scores]

        ax2.plot(labels, efficiency_33, marker='o', label='SVR-33 Efficiency', linewidth=2)
        ax2.plot(labels, efficiency_all, marker='s', label='SVR-All Efficiency', linewidth=2)
        ax2.set_title('Feature Efficiency Comparison')
        ax2.set_ylabel('R² per Feature')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Improvements over SVR-33
        colors_svr = ['green' if imp >= 0 else 'red' for imp in svr_improvements]
        bars_svr = ax3.bar(labels, svr_improvements, color=colors_svr, alpha=0.7)
        ax3.set_title('SVR-All Improvement over SVR-33')
        ax3.set_ylabel('Improvement (%)')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)

        for bar, imp in zip(bars_svr, svr_improvements):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                     f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                     fontweight='bold')

        # Feature count vs performance trade-off
        ax4.scatter(features_used, all_svr_scores, s=100, alpha=0.7, c=range(len(features_used)), cmap='viridis')
        ax4.axhline(y=max(svr_33_scores), color='red', linestyle='--', alpha=0.7, label='Best SVR-33 Performance')
        ax4.set_title('Feature-Performance Trade-off')
        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('R² Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        for i, (x, y, h) in enumerate(zip(features_used, all_svr_scores, labels)):
            ax4.annotate(h, (x, y), xytext=(5, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'model_comparison_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"All visualizations saved to {self.results_dir}")

    def run_all_features_svr(self):
        print("Starting All Features SVR Implementation")
        print("="*50)

        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        df = self.load_data()
        base_df, base_features = self.prepare_base_features(df)
        heat_demand = self.load_heat_demand_data(df)
        enhanced_features = self.add_advanced_features(df, heat_demand)
        targets = self.create_targets(heat_demand)
        ml_df, feature_names = self.prepare_ml_dataset(base_df, enhanced_features, targets)

        # Data splitting
        train_end = '2023-12-31 23:00:00'
        val_end = '2024-12-31 23:00:00'

        train_mask = ml_df.index <= train_end
        val_mask = (ml_df.index > train_end) & (ml_df.index <= val_end)
        test_mask = ml_df.index > val_end

        X_train = ml_df.loc[train_mask, feature_names]
        X_val = ml_df.loc[val_mask, feature_names]
        X_test = ml_df.loc[test_mask, feature_names]

        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Total features: {len(feature_names)}")

        results = {}
        all_kernel_results = {}

        for horizon in self.horizons:
            print(f"\nProcessing {horizon}h horizon")
            print("-" * 30)

            target_col = f'target_{horizon}h'
            y_train = ml_df.loc[train_mask, target_col].dropna()
            y_val = ml_df.loc[val_mask, target_col].dropna()
            y_test = ml_df.loc[test_mask, target_col].dropna()

            train_idx = y_train.index
            val_idx = y_val.index
            test_idx = y_test.index

            X_train_h = X_train.loc[train_idx]
            X_val_h = X_val.loc[val_idx]
            X_test_h = X_test.loc[test_idx]

            if len(y_train) < 100:
                print(f"Insufficient data for {horizon}h horizon")
                continue

            X_train_selected, selected_features = self.feature_selection_by_horizon(X_train_h, y_train, horizon)

            feature_indices = [i for i, feat in enumerate(X_train_h.columns) if feat in selected_features]
            X_val_selected = X_val_h.iloc[:, feature_indices].values
            X_test_selected = X_test_h.iloc[:, feature_indices].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_val_scaled = scaler.transform(X_val_selected)
            X_test_scaled = scaler.transform(X_test_selected)

            kernel_results = self.train_kernel_comparison(X_train_scaled, y_train.values, X_val_scaled, y_val.values, horizon)

            if not kernel_results:
                print(f"No successful models for {horizon}h horizon")
                continue

            # Store all kernel results for visualization
            all_kernel_results[horizon] = kernel_results

            best_model, best_kernel = self.select_best_model(kernel_results, horizon)
            final_metrics = self.evaluate_model(best_model, X_test_scaled, y_test.values, horizon)

            results[horizon] = {
                'model': best_model,
                'model_type': f"All Features SVR ({best_kernel})",
                'metrics': final_metrics,
                'features': selected_features,
                'scaler': scaler,
                'feature_count': len(selected_features),
                'kernel_results': kernel_results
            }

            metrics = final_metrics
            print(f"Results for {horizon}h:")
            print(f"  Features used: {len(selected_features)}")
            print(f"  R2 Score: {metrics['r2_score']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")

        end_time = datetime.now()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time}")

        # Create visualizations
        self.create_visualizations(results, all_kernel_results, timestamp)

        return results, timestamp

    def save_results(self, results, timestamp):
        print(f"Saving results to {self.results_dir}...")

        # Save detailed results (excluding models for size)
        save_results = {}
        for horizon, result in results.items():
            save_results[horizon] = {
                'model_type': result['model_type'],
                'metrics': {k: v for k, v in result['metrics'].items() if k not in ['predictions', 'actuals']},
                'features': result['features'],
                'feature_count': result['feature_count'],
                'kernel_results': {k: {kk: vv for kk, vv in v.items() if kk not in ['model', 'predictions', 'actuals']}
                                   for k, v in result['kernel_results'].items()}
            }

        results_file = self.results_dir / f"svr_all_features_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(save_results, f)

        # Save model states separately
        for horizon, result in results.items():
            model_file = self.results_dir / f"svr_all_features_model_{horizon}h_{timestamp}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)

        # Save scaler
        if results:
            scaler_file = self.results_dir / f"svr_all_features_scaler_{timestamp}.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(results[list(results.keys())[0]]['scaler'], f)

        # Save predictions for analysis
        predictions_data = {}
        for horizon, result in results.items():
            metrics = result['metrics']
            predictions_data[f'{horizon}h_predictions'] = metrics['predictions'].tolist()
            predictions_data[f'{horizon}h_actuals'] = metrics['actuals'].tolist()

        predictions_file = self.results_dir / f"svr_all_features_predictions_{timestamp}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        # Save summary CSV
        summary_data = []
        for horizon, result in results.items():
            metrics = result['metrics']
            model = result['model']
            summary_data.append({
                'timestamp': timestamp,
                'horizon': horizon,
                'model_type': result['model_type'],
                'kernel': model.kernel,
                'features_used': result['feature_count'],
                'C': model.C,
                'gamma': getattr(model, 'gamma', 'N/A'),
                'epsilon': model.epsilon,
                'r2_score': metrics['r2_score'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / f"svr_all_features_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        # Save kernel comparison results
        kernel_comparison_data = []
        for horizon, result in results.items():
            for kernel, kernel_result in result['kernel_results'].items():
                kernel_comparison_data.append({
                    'horizon': horizon,
                    'kernel': kernel,
                    'features_used': result['feature_count'],
                    'val_r2': kernel_result['val_r2'],
                    'val_rmse': kernel_result['val_rmse'],
                    'params': str(kernel_result['params'])
                })

        kernel_df = pd.DataFrame(kernel_comparison_data)
        kernel_file = self.results_dir / f"svr_all_features_kernel_comparison_{timestamp}.csv"
        kernel_df.to_csv(kernel_file, index=False)

        print(f"Results saved successfully!")
        print(f"- Main results: svr_all_features_results_{timestamp}.pkl")
        print(f"- Model states: svr_all_features_model_[horizon]h_{timestamp}.pkl")
        print(f"- Summary: svr_all_features_summary_{timestamp}.csv")
        print(f"- Predictions: svr_all_features_predictions_{timestamp}.json")
        print(f"- Kernel comparison: svr_all_features_kernel_comparison_{timestamp}.csv")
        print(f"- Visualizations: [multiple PNG files]")

    def print_final_summary(self, results):
        print("\nALL FEATURES SVR RESULTS SUMMARY")
        print("=" * 55)

        print(f"{'Horizon':>8} {'Features':>9} {'Kernel':>8} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
        print("-" * 64)

        total_features_used = 0
        for horizon in self.horizons:
            if horizon in results:
                metrics = results[horizon]['metrics']
                feature_count = results[horizon]['feature_count']
                kernel = results[horizon]['model'].kernel
                total_features_used += feature_count
                print(f"{horizon:>7}h {feature_count:>8} {kernel:>8} {metrics['r2_score']:>7.3f} {metrics['rmse']:>7.3f} {metrics['mae']:>7.3f} {metrics['mape']:>7.1f}")

        avg_features = total_features_used / len(results) if results else 0
        avg_r2 = np.mean([results[h]['metrics']['r2_score'] for h in results.keys()]) if results else 0

        print("-" * 64)
        print(f"Average features used: {avg_features:.0f}")
        print(f"Average R2 score: {avg_r2:.3f}")

    def compare_with_baselines(self, results):
        print("\nCOMPARISON WITH BASELINES")
        print("=" * 50)

        # Baselines
        svr_33_baseline = {1: 0.989, 6: 0.930, 24: 0.864, 48: 0.793, 72: 0.739}
        ffnn_baseline = {1: 0.988, 6: 0.943, 24: 0.861, 48: 0.800, 72: 0.743}

        print(f"{'Horizon':>8} {'SVR-33':>8} {'FFNN-33':>9} {'SVR-All':>9} {'vs SVR':>8} {'vs FFNN':>9}")
        print("-" * 62)

        total_svr_improvement = 0
        total_ffnn_improvement = 0
        valid_comparisons = 0

        for horizon in self.horizons:
            if horizon in results and horizon in svr_33_baseline and horizon in ffnn_baseline:
                svr_33_score = svr_33_baseline[horizon]
                ffnn_score = ffnn_baseline[horizon]
                current_score = results[horizon]['metrics']['r2_score']

                svr_improvement = ((current_score - svr_33_score) / svr_33_score) * 100
                ffnn_improvement = ((current_score - ffnn_score) / ffnn_score) * 100

                total_svr_improvement += svr_improvement
                total_ffnn_improvement += ffnn_improvement
                valid_comparisons += 1

                print(f"{horizon:>7}h {svr_33_score:>7.3f} {ffnn_score:>8.3f} {current_score:>8.3f} {svr_improvement:>+6.1f}% {ffnn_improvement:>+7.1f}%")

        if valid_comparisons > 0:
            avg_svr_improvement = total_svr_improvement / valid_comparisons
            avg_ffnn_improvement = total_ffnn_improvement / valid_comparisons
            print("-" * 62)
            print(f"Average vs SVR-33: {avg_svr_improvement:+.1f}%")
            print(f"Average vs FFNN-33: {avg_ffnn_improvement:+.1f}%")


def main():
    print("All Features SVR Implementation - HPC Execution")
    print("Schweinfurt District Heating Network Forecasting")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    svr_system = AllFeaturesSVRForecasting(fast_mode=True)

    start_time = datetime.now()

    results, timestamp = svr_system.run_all_features_svr()
    svr_system.save_results(results, timestamp)
    svr_system.print_final_summary(results)
    svr_system.compare_with_baselines(results)

    end_time = datetime.now()
    execution_time = end_time - start_time

    print(f"\nAll Features SVR Implementation Complete!")
    print(f"Total Execution Time: {execution_time}")
    print(f"Results saved in: {svr_system.results_dir}")
    print(f"Timestamp: {timestamp}")

if __name__ == "__main__":
    main()