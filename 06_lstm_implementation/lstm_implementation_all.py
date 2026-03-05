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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

class AllFeaturesLSTMDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class AllFeaturesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):
        super(AllFeaturesLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        x_out = self.dropout(last_output)
        x_out = torch.relu(self.layer_norm1(self.fc1(x_out)))
        x_out = self.dropout(x_out)
        x_out = torch.relu(self.layer_norm2(self.fc2(x_out)))
        x_out = self.dropout(x_out)
        output = self.fc3(x_out)

        return output

class AllFeaturesLSTMForecasting:

    def __init__(self, data_path=None, device=None, results_path=None):
        if data_path is None:
            self.data_path = Path("/workspace/Thesis/01_data/processed_data")
        else:
            self.data_path = Path(data_path)

        if results_path is None:
            self.results_path = Path("/home/student22/Thesis/06_lstm_implementation/results/allfeatures")
        else:
            self.results_path = Path(results_path)

        self.results_path.mkdir(parents=True, exist_ok=True)

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.horizons = [1, 6, 24, 48, 72]

        self.sequence_lengths = {
            1: 48, 6: 72, 24: 96, 48: 120, 72: 144
        }

        self.architectures = {
            1: {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.0005, 'batch_size': 16},
            6: {'hidden_size': 192, 'num_layers': 2, 'dropout': 0.25, 'lr': 0.0003, 'batch_size': 12},
            24: {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.0002, 'batch_size': 8},
            48: {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.35, 'lr': 0.0002, 'batch_size': 6},
            72: {'hidden_size': 320, 'num_layers': 2, 'dropout': 0.4, 'lr': 0.0001, 'batch_size': 4}
        }

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

        print("All Features LSTM Forecasting System Initialized")
        print(f"Results will be saved to: {self.results_path}")
        print(f"Target Horizons: {self.horizons}")

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

        return df[available_features].copy()

    def engineer_temporal_features(self, df, feature_df):
        print("Engineering temporal features...")

        feature_df = feature_df.copy()

        if 'hour' not in feature_df.columns and hasattr(df.index, 'hour'):
            feature_df['hour'] = df.index.hour
            feature_df['day_of_week'] = df.index.dayofweek
            feature_df['month'] = df.index.month
            feature_df['day_of_year'] = df.index.dayofyear

            feature_df['hour_sin'] = np.sin(2 * np.pi * feature_df['hour'] / 24)
            feature_df['hour_cos'] = np.cos(2 * np.pi * feature_df['hour'] / 24)
            feature_df['day_sin'] = np.sin(2 * np.pi * feature_df['day_of_week'] / 7)
            feature_df['day_cos'] = np.cos(2 * np.pi * feature_df['day_of_week'] / 7)
            feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
            feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)

            feature_df['is_weekend'] = (feature_df['day_of_week'] >= 5).astype(int)
            feature_df['is_business_hour'] = ((feature_df['hour'] >= 8) & (feature_df['hour'] <= 17)).astype(int)

        return feature_df

    def engineer_lag_features(self, df, target_col='Last', n_lags=[1, 2, 3, 6, 12, 24, 48, 72]):
        print(f"Engineering lag features for {target_col}...")

        feature_df = pd.DataFrame(index=df.index)

        if target_col in df.columns:
            for lag in n_lags:
                feature_df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

            feature_df[f'{target_col}_diff_1'] = df[target_col].diff(1)
            feature_df[f'{target_col}_diff_24'] = df[target_col].diff(24)

        return feature_df

    def engineer_rolling_features(self, df, target_col='Last', windows=[6, 12, 24, 48]):
        print(f"Engineering rolling features for {target_col}...")

        feature_df = pd.DataFrame(index=df.index)

        if target_col in df.columns:
            for window in windows:
                feature_df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
                feature_df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
                feature_df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
                feature_df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()

        return feature_df

    def create_all_features(self, df):
        print("Creating complete feature set...")

        target_col = 'Last'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        base_features = self.prepare_base_features(df)
        temporal_features = self.engineer_temporal_features(df, base_features)
        lag_features = self.engineer_lag_features(df, target_col)
        rolling_features = self.engineer_rolling_features(df, target_col)

        all_features = pd.concat([
            temporal_features,
            lag_features,
            rolling_features
        ], axis=1)

        all_features = all_features.dropna()

        if target_col in df.columns:
            all_features[target_col] = df[target_col]
            all_features = all_features.dropna(subset=[target_col])

        print(f"Total features created: {all_features.shape[1]}")
        print(f"Total samples after cleaning: {all_features.shape[0]}")

        return all_features

    def select_features(self, X, y, k=100):
        print(f"Selecting top {k} features...")

        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()

        print(f"Selected {len(selected_features)} features")

        return X_selected, selected_features, selector

    def create_sequences(self, data, seq_length, horizon):
        sequences = []
        targets = []

        for i in range(len(data) - seq_length - horizon + 1):
            seq = data[i:(i + seq_length)]
            target = data[i + seq_length + horizon - 1, -1]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def split_data(self, sequences, targets, train_ratio=0.7, val_ratio=0.15):
        n = len(sequences)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = sequences[:train_end]
        X_val = sequences[train_end:val_end]
        X_test = sequences[val_end:]

        y_train = targets[:train_end]
        y_val = targets[train_end:val_end]
        y_test = targets[val_end:]

        test_idx = list(range(val_end, n))

        return X_train, X_val, X_test, y_train, y_val, y_test, test_idx

    def scale_data(self, X_train, X_val, X_test):
        scaler = StandardScaler()

        n_samples, seq_len, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_samples, seq_len, n_features)

        X_val_2d = X_val.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape[0], seq_len, n_features)

        X_test_2d = X_test.reshape(-1, n_features)
        X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape[0], seq_len, n_features)

        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    def create_data_loaders(self, X_train, y_train, X_val, y_val, batch_size):
        train_dataset = AllFeaturesLSTMDataset(X_train, y_train)
        val_dataset = AllFeaturesLSTMDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_model(self, model, train_loader, val_loader, config, horizon):
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 30
        best_model_state = None

        epochs = 150

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_batches = 0

            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model(sequences)
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss - 0.0001:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, train_losses, val_losses

    def evaluate_model(self, model, X_test, y_test, horizon, test_idx):
        model.eval()
        predictions = []

        test_dataset = AllFeaturesLSTMDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                outputs = model(sequences)
                predictions.extend(outputs.squeeze().cpu().numpy())

        predictions = np.array(predictions)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        epsilon = 1e-10
        mape = np.mean(np.abs((y_test - predictions) / (y_test + epsilon))) * 100

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'predictions': predictions,
            'actual': y_test,
            'test_index': test_idx
        }

        return metrics

    def create_comprehensive_performance_analysis(self, results, timestamp):
        sns.set_style("whitegrid")

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        horizons_list = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        mape_scores = []

        for horizon in self.horizons:
            if horizon in results:
                horizons_list.append(horizon)
                metrics = results[horizon]['metrics']
                r2_scores.append(metrics['r2_score'])
                rmse_scores.append(metrics['rmse'])
                mae_scores.append(metrics['mae'])
                mape_scores.append(metrics['mape'])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(horizons_list, r2_scores, marker='o', linewidth=2.5, markersize=10, color='#2E86AB')
        ax1.set_xlabel('Forecast Horizon (hours)', fontsize=12)
        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.set_title('R² Score vs Horizon', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(horizons_list, rmse_scores, marker='s', linewidth=2.5, markersize=10, color='#A23B72')
        ax2.set_xlabel('Forecast Horizon (hours)', fontsize=12)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('RMSE vs Horizon', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(horizons_list, mae_scores, marker='^', linewidth=2.5, markersize=10, color='#F18F01')
        ax3.set_xlabel('Forecast Horizon (hours)', fontsize=12)
        ax3.set_ylabel('MAE', fontsize=12)
        ax3.set_title('MAE vs Horizon', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        for idx, horizon in enumerate(self.horizons[:5]):
            if horizon in results:
                row = (idx // 3) + 1
                col = idx % 3
                ax = fig.add_subplot(gs[row, col])

                metrics = results[horizon]['metrics']
                predictions = metrics['predictions'][:500]
                actual = metrics['actual'][:500]

                ax.plot(actual, label='Actual', alpha=0.7, linewidth=2)
                ax.plot(predictions, label='Predicted', alpha=0.7, linewidth=2)
                ax.set_xlabel('Sample Index', fontsize=11)
                ax.set_ylabel('Heat Demand', fontsize=11)
                ax.set_title(f'{horizon}h Horizon - R²={metrics["r2_score"]:.3f}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)

        plt.savefig(self.results_path / f'comprehensive_performance_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_model_analysis(self, results, timestamp):
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, horizon in enumerate(self.horizons):
            if horizon in results:
                ax = axes[idx]
                train_losses = results[horizon]['train_losses']
                val_losses = results[horizon]['val_losses']

                ax.plot(train_losses, label='Train Loss', alpha=0.8, linewidth=2)
                ax.plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('Loss (MSE)', fontsize=11)
                ax.set_title(f'{horizon}h Horizon Training', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')

        if len(self.horizons) < len(axes):
            for idx in range(len(self.horizons), len(axes)):
                fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(self.results_path / f'model_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_time_series_forecasts(self, results, timestamp):
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, horizon in enumerate(self.horizons):
            if horizon in results:
                ax = axes[idx]
                metrics = results[horizon]['metrics']
                predictions = metrics['predictions']
                actual = metrics['actual']

                sample_size = min(1000, len(predictions))
                x_range = range(sample_size)

                ax.plot(x_range, actual[:sample_size], label='Actual', alpha=0.7, linewidth=1.5)
                ax.plot(x_range, predictions[:sample_size], label='Predicted', alpha=0.7, linewidth=1.5)
                ax.set_xlabel('Sample Index', fontsize=10)
                ax.set_ylabel('Heat Demand', fontsize=10)
                ax.set_title(f'{horizon}h Horizon Forecast\nR²={metrics["r2_score"]:.3f}, RMSE={metrics["rmse"]:.3f}',
                             fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        if len(self.horizons) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig(self.results_path / f'time_series_forecasts_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_general_graphs(self, results, timestamp):
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, horizon in enumerate(self.horizons):
            if horizon in results:
                ax = axes[idx]
                metrics = results[horizon]['metrics']
                predictions = metrics['predictions']
                actual = metrics['actual']

                ax.scatter(actual, predictions, alpha=0.5, s=20)
                min_val = min(actual.min(), predictions.min())
                max_val = max(actual.max(), predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

                ax.set_xlabel('Actual Values', fontsize=10)
                ax.set_ylabel('Predicted Values', fontsize=10)
                ax.set_title(f'{horizon}h Horizon\nR²={metrics["r2_score"]:.3f}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        if len(self.horizons) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig(self.results_path / f'general_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_all_features_lstm_forecasting(self):
        start_time = datetime.now()

        df = self.load_data()
        all_features_df = self.create_all_features(df)

        target_col = 'Last'
        X = all_features_df.drop(columns=[target_col])
        y = all_features_df[target_col].values

        results = {}

        for horizon in self.horizons:
            print(f"\nTraining for {horizon}h horizon...")

            config = self.architectures[horizon]
            seq_length = self.sequence_lengths[horizon]

            k_features = min(100, X.shape[1])
            X_selected, selected_features, selector = self.select_features(X, y, k=k_features)

            data_with_target = np.column_stack([X_selected, y])
            sequences, targets = self.create_sequences(data_with_target, seq_length, horizon)

            X_seq_train, X_seq_val, X_seq_test, y_seq_train, y_seq_val, y_seq_test, test_idx = self.split_data(
                sequences, targets
            )

            X_seq_train, X_seq_val, X_seq_test, scaler = self.scale_data(
                X_seq_train, X_seq_val, X_seq_test
            )

            input_size = X_seq_train.shape[2]
            model = AllFeaturesLSTM(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )

            print(f"Model architecture:")
            print(f"  Hidden size: {config['hidden_size']}")
            print(f"  Layers: {config['num_layers']}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

            train_loader, val_loader = self.create_data_loaders(
                X_seq_train, y_seq_train, X_seq_val, y_seq_val, config['batch_size']
            )

            model, train_losses, val_losses = self.train_model(
                model, train_loader, val_loader, config, horizon
            )

            final_metrics = self.evaluate_model(model, X_seq_test, y_seq_test, horizon, test_idx)

            results[horizon] = {
                'model': model,
                'model_type': f"All Features LSTM (h={config['hidden_size']}, l={config['num_layers']}, seq={self.sequence_lengths[horizon]})",
                'metrics': final_metrics,
                'features': selected_features,
                'scaler': scaler,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config,
                'sequence_length': self.sequence_lengths[horizon],
                'feature_count': len(selected_features)
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

        return results

    def save_results(self, results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.create_comprehensive_performance_analysis(results, timestamp)
        self.create_model_analysis(results, timestamp)
        self.create_time_series_forecasts(results, timestamp)
        self.create_general_graphs(results, timestamp)

        save_results = {}
        for horizon, result in results.items():
            save_results[horizon] = {
                'model_type': result['model_type'],
                'metrics': {k: v for k, v in result['metrics'].items()
                            if k not in ['predictions', 'actual', 'test_index']},
                'features': result['features'],
                'train_losses': result['train_losses'],
                'val_losses': result['val_losses'],
                'config': result['config'],
                'sequence_length': result['sequence_length'],
                'feature_count': result['feature_count']
            }

        results_file = self.results_path / f"all_features_lstm_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(save_results, f)

        for horizon, result in results.items():
            model_file = self.results_path / f"all_features_lstm_model_{horizon}h_{timestamp}.pth"
            torch.save(result['model'].state_dict(), model_file)

        summary_data = []
        for horizon, result in results.items():
            metrics = result['metrics']
            summary_data.append({
                'horizon': horizon,
                'model_type': result['model_type'],
                'sequence_length': result['sequence_length'],
                'features_used': result['feature_count'],
                'parameters': sum(p.numel() for p in result['model'].parameters()),
                'r2_score': metrics['r2_score'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_path / f"all_features_lstm_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        print(f"Results saved to: {self.results_path}")
        print(f"Generated visualizations:")
        print(f"  - comprehensive_performance_analysis_{timestamp}.png")
        print(f"  - model_analysis_{timestamp}.png")
        print(f"  - time_series_forecasts_{timestamp}.png")
        print(f"  - general_analysis_{timestamp}.png")

    def print_final_summary(self, results):
        print("\nFINAL ALL FEATURES LSTM RESULTS SUMMARY")
        print("=" * 60)

        print(f"{'Horizon':>8} {'Features':>9} {'R2':>8} {'RMSE':>8} {'MAE':>8}")
        print("-" * 50)

        total_features_used = 0
        for horizon in self.horizons:
            if horizon in results:
                metrics = results[horizon]['metrics']
                feature_count = results[horizon]['feature_count']
                total_features_used += feature_count
                print(f"{horizon:>7}h {feature_count:>8} {metrics['r2_score']:>7.3f} {metrics['rmse']:>7.3f} {metrics['mae']:>7.3f}")

        avg_features = total_features_used / len(results)
        avg_r2 = np.mean([results[h]['metrics']['r2_score'] for h in results.keys()])

        print("-" * 50)
        print(f"Average features used: {avg_features:.0f}")
        print(f"Average R2 score: {avg_r2:.3f}")

    def compare_with_all_models(self, results):
        print("\nCOMPARISON WITH ALL PREVIOUS MODELS")
        print("=" * 90)

        svr_33 = {1: 0.989, 6: 0.930, 24: 0.864, 48: 0.793, 72: 0.739}
        ffnn_33 = {1: 0.988, 6: 0.943, 24: 0.861, 48: 0.800, 72: 0.743}
        lstm_33 = {1: 0.991, 6: 0.943, 24: 0.813, 48: 0.727, 72: 0.687}
        enhanced_lstm = {1: 0.994, 6: 0.937, 24: 0.842, 48: 0.751, 72: 0.664}

        print(f"{'Horizon':>8} {'SVR-33':>8} {'FFNN-33':>9} {'LSTM-33':>9} {'Enh-LSTM':>10} {'All-LSTM':>10} {'Best Gain':>10}")
        print("-" * 85)

        total_improvements = []

        for horizon in self.horizons:
            if horizon in results:
                svr_score = svr_33[horizon]
                ffnn_score = ffnn_33[horizon]
                lstm_score = lstm_33[horizon]
                enhanced_score = enhanced_lstm[horizon]
                all_features_score = results[horizon]['metrics']['r2_score']

                best_previous = max(svr_score, ffnn_score, lstm_score, enhanced_score)
                improvement = ((all_features_score - best_previous) / best_previous) * 100
                total_improvements.append(improvement)

                print(f"{horizon:>7}h {svr_score:>7.3f} {ffnn_score:>8.3f} {lstm_score:>8.3f} {enhanced_score:>9.3f} {all_features_score:>9.3f} {improvement:>+8.1f}%")

        if total_improvements:
            avg_improvement = np.mean(total_improvements)
            print("-" * 85)
            print(f"Average improvement over best previous model: {avg_improvement:+.1f}%")

            long_horizon_improvements = [total_improvements[i] for i in range(len(self.horizons)) if self.horizons[i] >= 24]
            if long_horizon_improvements:
                long_avg = np.mean(long_horizon_improvements)
                print(f"Average improvement for long horizons (24h+): {long_avg:+.1f}%")


def main():
    print("All Features LSTM Implementation - HPC Execution")
    print("Schweinfurt District Heating Network Forecasting")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nFeatures:")
    print("- Complete feature set from MLR implementation")
    print("- Advanced feature engineering with lag and rolling features")
    print("- Bidirectional LSTM with feature selection")
    print("- Memory-optimized for GPU constraints")
    print("- Comprehensive visualization and analysis")

    lstm_system = AllFeaturesLSTMForecasting()

    start_time = datetime.now()

    results = lstm_system.run_all_features_lstm_forecasting()
    lstm_system.save_results(results)
    lstm_system.print_final_summary(results)
    lstm_system.compare_with_all_models(results)

    end_time = datetime.now()
    execution_time = end_time - start_time

    print(f"\nAll Features LSTM Implementation Complete!")
    print(f"Total Execution Time: {execution_time}")
    print(f"Results saved in: {lstm_system.results_path}")

if __name__ == "__main__":
    main()