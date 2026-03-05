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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class FFNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AllFeaturesFFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.3):
        super(AllFeaturesFFNN, self).__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            current_dropout = dropout_rate * (1.0 + 0.1 * i)
            layers.append(nn.Dropout(current_dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)

class AllFeaturesFFNNForecasting:

    def __init__(self, data_path=None, device=None):
        if data_path is None:
            self.data_path = Path("/workspace/Thesis/01_data/processed_data")
        else:
            self.data_path = Path(data_path)

        self.results_dir = Path("/workspace/Thesis/05_ffnn_implementation/results/allfeatures")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.horizons = [1, 6, 24, 48, 72]

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

        self.architectures = {
            1: {'hidden_sizes': [256, 128], 'dropout': 0.2, 'lr': 0.0005, 'batch_size': 64},
            6: {'hidden_sizes': [512, 256], 'dropout': 0.25, 'lr': 0.0003, 'batch_size': 64},
            24: {'hidden_sizes': [512, 256, 128], 'dropout': 0.3, 'lr': 0.0003, 'batch_size': 32},
            48: {'hidden_sizes': [768, 384, 192], 'dropout': 0.35, 'lr': 0.0002, 'batch_size': 32},
            72: {'hidden_sizes': [1024, 512, 256, 128], 'dropout': 0.4, 'lr': 0.0002, 'batch_size': 16}
        }

        self.exclude_columns = ['datetime', 'timestamp_local', 'timestamp_utc', 'ts']
        self.results = {}

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        print("All Features FFNN Forecasting System Initialized")
        print(f"Target Horizons: {self.horizons}")
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

        lag_hours = [12, 23, 24, 25, 36, 47, 48, 49, 60, 72, 84, 96, 120, 144]
        for lag in lag_hours:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        weekly_lags = [162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 174, 176]
        for lag in weekly_lags:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        for lag in [336, 504]:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        windows = [24, 48, 72, 96, 120, 144, 168, 240, 336]
        for window in windows:
            enhanced_features[f'demand_rolling_mean_{window}h'] = heat_demand.rolling(window).mean()
            enhanced_features[f'demand_rolling_std_{window}h'] = heat_demand.rolling(window).std()

        enhanced_features['demand_rolling_min_72h'] = heat_demand.rolling(72).min()
        enhanced_features['demand_rolling_max_72h'] = heat_demand.rolling(72).max()
        enhanced_features['demand_rolling_median_72h'] = heat_demand.rolling(72).median()

        hour = df['hour'] if 'hour' in df.columns else df.index.hour
        enhanced_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        enhanced_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        dow = df['day_of_week'] if 'day_of_week' in df.columns else df.index.dayofweek
        enhanced_features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        enhanced_features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        month = df['month'] if 'month' in df.columns else df.index.month
        enhanced_features['month_sin'] = np.sin(2 * np.pi * month / 12)
        enhanced_features['month_cos'] = np.cos(2 * np.pi * month / 12)

        week_position = (dow * 24 + hour) / 168.0
        enhanced_features['week_position_sin'] = np.sin(2 * np.pi * week_position)
        enhanced_features['week_position_cos'] = np.cos(2 * np.pi * week_position)

        for harmonic in [2, 3]:
            enhanced_features[f'week_position_sin_{harmonic}'] = np.sin(2 * np.pi * harmonic * week_position)
            enhanced_features[f'week_position_cos_{harmonic}'] = np.cos(2 * np.pi * harmonic * week_position)

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

        enhanced_features['is_business_day'] = ((dow < 5)).astype(int)
        enhanced_features['is_friday'] = (dow == 4).astype(int)
        enhanced_features['is_sunday'] = (dow == 6).astype(int)
        enhanced_features['is_weekend'] = (dow >= 5).astype(int)
        enhanced_features['is_monday'] = (dow == 0).astype(int)
        enhanced_features['is_tuesday'] = (dow == 1).astype(int)
        enhanced_features['is_wednesday'] = (dow == 2).astype(int)
        enhanced_features['is_thursday'] = (dow == 3).astype(int)
        enhanced_features['is_saturday'] = (dow == 5).astype(int)

        enhanced_features['friday_evening'] = ((dow == 4) & (hour >= 16)).astype(int)
        enhanced_features['sunday_evening'] = ((dow == 6) & (hour >= 18)).astype(int)
        enhanced_features['monday_morning'] = ((dow == 0) & (hour <= 11)).astype(int)

        day_of_year = df.index.dayofyear if hasattr(df.index, 'dayofyear') else ((month - 1) * 30 + 15)
        enhanced_features['season_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        enhanced_features['season_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        if hasattr(df.index, 'isocalendar'):
            week_of_year = df.index.isocalendar().week
        else:
            week_of_year = ((day_of_year - 1) // 7 + 1)
        enhanced_features['week_of_year_sin'] = np.sin(2 * np.pi * week_of_year / 52)
        enhanced_features['week_of_year_cos'] = np.cos(2 * np.pi * week_of_year / 52)

        if 'hdd_15_5' in df.columns:
            enhanced_features['hdd_lag_48h'] = df['hdd_15_5'].shift(48)
            enhanced_features['hdd_lag_72h'] = df['hdd_15_5'].shift(72)
            enhanced_features['hdd_lag_168h'] = df['hdd_15_5'].shift(168)
            enhanced_features['hdd_trend_72h'] = df['hdd_15_5'].rolling(72).apply(safe_trend, raw=False)
            enhanced_features['hdd_trend_168h'] = df['hdd_15_5'].rolling(168).apply(safe_trend, raw=False)

        major_zones = ['B1_B2', 'V1', 'W1', 'F1_Sud']
        for zone in major_zones:
            supply_col = f'{zone}_expected_supply_temp'
            if supply_col in df.columns:
                enhanced_features[f'{zone}_lag_72h'] = df[supply_col].shift(72)
                enhanced_features[f'{zone}_lag_168h'] = df[supply_col].shift(168)
                enhanced_features[f'{zone}_trend_72h'] = df[supply_col].rolling(72).apply(safe_trend, raw=False)

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

        if horizon <= 6:
            k_features = min(120, len(var_features))
        elif horizon <= 24:
            k_features = min(180, len(var_features))
        elif horizon <= 48:
            k_features = min(220, len(var_features))
        else:
            k_features = min(280, len(var_features))

        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X_var, y)
        selected_indices = selector.get_support()
        selected_features = var_features[selected_indices].tolist()

        print(f"{horizon}h feature selection: {len(X_numeric.columns)} -> {len(var_features)} -> {len(selected_features)}")
        return X_selected, selected_features

    def create_data_loaders(self, X_train, y_train, X_val, y_val, batch_size):
        train_dataset = FFNNDataset(X_train, y_train)
        val_dataset = FFNNDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader

    def train_model(self, model, train_loader, val_loader, config, horizon):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

        epochs = 120 if horizon <= 6 else (160 if horizon <= 24 else 200)
        best_val_loss = float('inf')
        patience = 25
        patience_counter = 0

        train_losses = []
        val_losses = []

        print(f"  Training for {epochs} epochs with early stopping")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 40 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        model.load_state_dict(best_model_state)

        return model, train_losses, val_losses

    def evaluate_model(self, model, X_test, y_test, horizon):
        model.eval()

        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred = model(X_test_tensor).cpu().numpy().squeeze()

        metrics = {
            'horizon': horizon,
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'predictions': y_pred,
            'actuals': y_test
        }

        return metrics

    def create_visualizations(self, results, timestamp):
        """Create comprehensive visualizations for All Features FFNN results"""
        print("Creating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('All Features FFNN Comprehensive Performance Analysis', fontsize=16, fontweight='bold')

        horizons = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        mape_scores = []
        features_used = []

        for horizon in self.horizons:
            if horizon in results:
                horizons.append(f"{horizon}h")
                metrics = results[horizon]['metrics']
                r2_scores.append(metrics['r2_score'])
                rmse_scores.append(metrics['rmse'])
                mae_scores.append(metrics['mae'])
                mape_scores.append(metrics['mape'])
                features_used.append(results[horizon]['feature_count'])

        axes[0,0].bar(horizons, r2_scores, color='skyblue', alpha=0.8)
        axes[0,0].set_title('R² Score by Horizon')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_ylim(0, 1)
        for i, v in enumerate(r2_scores):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

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

        axes[1,0].bar(horizons, features_used, color='mediumpurple', alpha=0.8)
        axes[1,0].set_title('Features Used by Horizon')
        axes[1,0].set_ylabel('Number of Features')
        for i, v in enumerate(features_used):
            axes[1,0].text(i, v + max(features_used)*0.01, f'{v}', ha='center', fontweight='bold')

        axes[1,1].plot(horizons, r2_scores, marker='o', linewidth=2, markersize=8, label='R² Score')
        axes[1,1].set_title('Performance Degradation Trend')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        parameters = []
        for horizon in self.horizons:
            if horizon in results:
                param_count = sum(p.numel() for p in results[horizon]['model'].parameters())
                parameters.append(param_count/1000)

        axes[1,2].bar(horizons, parameters, color='gold', alpha=0.8)
        axes[1,2].set_title('Model Parameters by Horizon')
        axes[1,2].set_ylabel('Parameters (thousands)')
        for i, v in enumerate(parameters):
            axes[1,2].text(i, v + max(parameters)*0.01, f'{v:.0f}K', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'comprehensive_performance_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('All Features FFNN Model Analysis - Training Curves', fontsize=16, fontweight='bold')

        for idx, horizon in enumerate(self.horizons[:6]):
            if horizon in results:
                row = idx // 3
                col = idx % 3

                train_losses = results[horizon]['train_losses']
                val_losses = results[horizon]['val_losses']

                axes[row, col].plot(train_losses, label='Training Loss', alpha=0.8)
                axes[row, col].plot(val_losses, label='Validation Loss', alpha=0.8)
                axes[row, col].set_title(f'{horizon}h Horizon ({results[horizon]["feature_count"]} features)')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Loss')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        for idx in range(len(self.horizons), 6):
            row = idx // 3
            col = idx % 3
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.savefig(self.results_dir / f'model_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('All Features FFNN Time Series Forecasts', fontsize=16, fontweight='bold')

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
                axes[row, col].set_title(f'{horizon}h Horizon (R²={metrics["r2_score"]:.3f}, {results[horizon]["feature_count"]} features)')
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
        fig.suptitle('All Features FFNN Prediction Accuracy Analysis', fontsize=16, fontweight='bold')

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

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('All Features FFNN vs Baselines Comparison', fontsize=16, fontweight='bold')

        svr_baseline = {1: 0.989, 6: 0.930, 24: 0.864, 48: 0.793, 72: 0.739}
        ffnn_baseline = {1: 0.988, 6: 0.943, 24: 0.861, 48: 0.800, 72: 0.743}

        all_ffnn_scores = []
        svr_scores = []
        ffnn_33_scores = []
        labels = []
        svr_improvements = []
        ffnn_improvements = []

        for horizon in self.horizons:
            if horizon in results and horizon in svr_baseline and horizon in ffnn_baseline:
                labels.append(f"{horizon}h")
                all_ffnn_score = results[horizon]['metrics']['r2_score']
                svr_score = svr_baseline[horizon]
                ffnn_33_score = ffnn_baseline[horizon]

                all_ffnn_scores.append(all_ffnn_score)
                svr_scores.append(svr_score)
                ffnn_33_scores.append(ffnn_33_score)

                svr_improvement = ((all_ffnn_score - svr_score) / svr_score) * 100
                ffnn_improvement = ((all_ffnn_score - ffnn_33_score) / ffnn_33_score) * 100

                svr_improvements.append(svr_improvement)
                ffnn_improvements.append(ffnn_improvement)

        x_pos = np.arange(len(labels))
        width = 0.25

        ax1.bar(x_pos - width, svr_scores, width, label='SVR-33', alpha=0.8, color='lightcoral')
        ax1.bar(x_pos, ffnn_33_scores, width, label='FFNN-33', alpha=0.8, color='skyblue')
        ax1.bar(x_pos + width, all_ffnn_scores, width, label='FFNN-All', alpha=0.8, color='lightgreen')
        ax1.set_title('R² Score Comparison')
        ax1.set_ylabel('R² Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        feature_counts = [results[self.horizons[i]]['feature_count'] if i < len(results) else 0 for i in range(len(labels))]
        ax2.bar(labels, feature_counts, color='mediumpurple', alpha=0.7)
        ax2.set_title('Features Used by All-FFNN')
        ax2.set_ylabel('Number of Features')
        ax2.grid(True, alpha=0.3)
        for i, v in enumerate(feature_counts):
            ax2.text(i, v + max(feature_counts)*0.01, f'{v}', ha='center', fontweight='bold')

        colors_svr = ['green' if imp >= 0 else 'red' for imp in svr_improvements]
        bars_svr = ax3.bar(labels, svr_improvements, color=colors_svr, alpha=0.7)
        ax3.set_title('FFNN-All Improvement over SVR-33')
        ax3.set_ylabel('Improvement (%)')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)

        for bar, imp in zip(bars_svr, svr_improvements):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                     f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                     fontweight='bold')

        colors_ffnn = ['green' if imp >= 0 else 'red' for imp in ffnn_improvements]
        bars_ffnn = ax4.bar(labels, ffnn_improvements, color=colors_ffnn, alpha=0.7)
        ax4.set_title('FFNN-All Improvement over FFNN-33')
        ax4.set_ylabel('Improvement (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)

        for bar, imp in zip(bars_ffnn, ffnn_improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                     f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                     fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'model_comparison_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"All visualizations saved to {self.results_dir}")

    def run_all_features_ffnn(self):
        print("Starting All Features FFNN Implementation")
        print("="*50)

        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        df = self.load_data()
        base_df, base_features = self.prepare_base_features(df)
        heat_demand = self.load_heat_demand_data(df)
        enhanced_features = self.add_advanced_features(df, heat_demand)
        targets = self.create_targets(heat_demand)
        ml_df, feature_names = self.prepare_ml_dataset(base_df, enhanced_features, targets)

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

            config = self.architectures[horizon]

            model = AllFeaturesFFNN(
                input_size=X_train_scaled.shape[1],
                hidden_sizes=config['hidden_sizes'],
                dropout_rate=config['dropout']
            ).to(self.device)

            print(f"  Model architecture: {X_train_scaled.shape[1]} -> {' -> '.join(map(str, config['hidden_sizes']))} -> 1")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Features used: {len(selected_features)}")

            train_loader, val_loader = self.create_data_loaders(
                X_train_scaled, y_train.values, X_val_scaled, y_val.values, config['batch_size']
            )

            model, train_losses, val_losses = self.train_model(
                model, train_loader, val_loader, config, horizon
            )

            final_metrics = self.evaluate_model(model, X_test_scaled, y_test.values, horizon)

            results[horizon] = {
                'model': model,
                'model_type': f"All Features FFNN ({' -> '.join(map(str, config['hidden_sizes']))})",
                'metrics': final_metrics,
                'features': selected_features,
                'scaler': scaler,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config,
                'feature_count': len(selected_features)
            }

            metrics = final_metrics
            print(f"Results for {horizon}h:")
            print(f"  R2 Score: {metrics['r2_score']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")

        end_time = datetime.now()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time}")

        self.create_visualizations(results, timestamp)

        return results, timestamp

    def save_results(self, results, timestamp):
        print(f"Saving results to {self.results_dir}...")

        save_results = {}
        for horizon, result in results.items():
            save_results[horizon] = {
                'model_type': result['model_type'],
                'metrics': {k: v for k, v in result['metrics'].items() if k not in ['predictions', 'actuals']},
                'features': result['features'],
                'train_losses': result['train_losses'],
                'val_losses': result['val_losses'],
                'config': result['config'],
                'feature_count': result['feature_count']
            }

        results_file = self.results_dir / f"all_features_ffnn_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(save_results, f)

        for horizon, result in results.items():
            model_file = self.results_dir / f"all_features_ffnn_model_{horizon}h_{timestamp}.pth"
            torch.save(result['model'].state_dict(), model_file)

        if results:
            scaler_file = self.results_dir / f"all_features_ffnn_scaler_{timestamp}.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(results[list(results.keys())[0]]['scaler'], f)

        predictions_data = {}
        for horizon, result in results.items():
            metrics = result['metrics']
            predictions_data[f'{horizon}h_predictions'] = metrics['predictions'].tolist()
            predictions_data[f'{horizon}h_actuals'] = metrics['actuals'].tolist()

        predictions_file = self.results_dir / f"all_features_ffnn_predictions_{timestamp}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        summary_data = []
        for horizon, result in results.items():
            metrics = result['metrics']
            summary_data.append({
                'timestamp': timestamp,
                'horizon': horizon,
                'model_type': result['model_type'],
                'features_used': result['feature_count'],
                'parameters': sum(p.numel() for p in result['model'].parameters()),
                'r2_score': metrics['r2_score'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / f"all_features_ffnn_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        if results:
            feature_importance_data = []
            first_horizon = list(results.keys())[0]
            features = results[first_horizon]['features']

            for i, feature in enumerate(features):
                feature_importance_data.append({
                    'feature': feature,
                    'selection_rank': i + 1,
                    'horizon': first_horizon
                })

            feature_df = pd.DataFrame(feature_importance_data)
            feature_file = self.results_dir / f"all_features_ffnn_feature_ranking_{timestamp}.csv"
            feature_df.to_csv(feature_file, index=False)

        print(f"Results saved successfully!")
        print(f"- Main results: all_features_ffnn_results_{timestamp}.pkl")
        print(f"- Model states: all_features_ffnn_model_[horizon]h_{timestamp}.pth")
        print(f"- Summary: all_features_ffnn_summary_{timestamp}.csv")
        print(f"- Predictions: all_features_ffnn_predictions_{timestamp}.json")
        print(f"- Feature ranking: all_features_ffnn_feature_ranking_{timestamp}.csv")
        print(f"- Visualizations: [multiple PNG files]")

    def print_final_summary(self, results):
        print("\nALL FEATURES FFNN RESULTS SUMMARY")
        print("=" * 55)

        print(f"{'Horizon':>8} {'Features':>9} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
        print("-" * 56)

        total_features_used = 0
        for horizon in self.horizons:
            if horizon in results:
                metrics = results[horizon]['metrics']
                feature_count = results[horizon]['feature_count']
                total_features_used += feature_count
                print(f"{horizon:>7}h {feature_count:>8} {metrics['r2_score']:>7.3f} {metrics['rmse']:>7.3f} {metrics['mae']:>7.3f} {metrics['mape']:>7.1f}")

        avg_features = total_features_used / len(results) if results else 0
        avg_r2 = np.mean([results[h]['metrics']['r2_score'] for h in results.keys()]) if results else 0

        print("-" * 56)
        print(f"Average features used: {avg_features:.0f}")
        print(f"Average R2 score: {avg_r2:.3f}")

    def compare_with_baselines(self, results):
        print("\nCOMPARISON WITH BASELINES")
        print("=" * 50)

        svr_baseline = {1: 0.989, 6: 0.930, 24: 0.864, 48: 0.793, 72: 0.739}
        ffnn_baseline = {1: 0.988, 6: 0.943, 24: 0.861, 48: 0.800, 72: 0.743}

        print(f"{'Horizon':>8} {'SVR-33':>8} {'FFNN-33':>9} {'All-FFNN':>9} {'vs SVR':>8} {'vs FFNN':>9}")
        print("-" * 62)

        total_svr_improvement = 0
        total_ffnn_improvement = 0
        valid_comparisons = 0

        for horizon in self.horizons:
            if horizon in results and horizon in svr_baseline and horizon in ffnn_baseline:
                svr_score = svr_baseline[horizon]
                ffnn_score = ffnn_baseline[horizon]
                current_score = results[horizon]['metrics']['r2_score']

                svr_improvement = ((current_score - svr_score) / svr_score) * 100
                ffnn_improvement = ((current_score - ffnn_score) / ffnn_score) * 100

                total_svr_improvement += svr_improvement
                total_ffnn_improvement += ffnn_improvement
                valid_comparisons += 1

                print(f"{horizon:>7}h {svr_score:>7.3f} {ffnn_score:>8.3f} {current_score:>8.3f} {svr_improvement:>+6.1f}% {ffnn_improvement:>+7.1f}%")

        if valid_comparisons > 0:
            avg_svr_improvement = total_svr_improvement / valid_comparisons
            avg_ffnn_improvement = total_ffnn_improvement / valid_comparisons
            print("-" * 62)
            print(f"Average vs SVR: {avg_svr_improvement:+.1f}%")
            print(f"Average vs FFNN-33: {avg_ffnn_improvement:+.1f}%")


def main():
    print("All Features FFNN Implementation - HPC Execution")
    print("Schweinfurt District Heating Network Forecasting")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ffnn_system = AllFeaturesFFNNForecasting()

    start_time = datetime.now()

    results, timestamp = ffnn_system.run_all_features_ffnn()
    ffnn_system.save_results(results, timestamp)
    ffnn_system.print_final_summary(results)
    ffnn_system.compare_with_baselines(results)

    end_time = datetime.now()
    execution_time = end_time - start_time

    print(f"\nAll Features FFNN Implementation Complete!")
    print(f"Total Execution Time: {execution_time}")
    print(f"Results saved in: {ffnn_system.results_dir}")
    print(f"Timestamp: {timestamp}")

if __name__ == "__main__":
    main()