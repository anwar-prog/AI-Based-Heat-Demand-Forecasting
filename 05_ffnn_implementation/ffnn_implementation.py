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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
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

class DistrictHeatingFFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.2):
        super(DistrictHeatingFFNN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)

class FFNNForecasting:

    def __init__(self, data_path=None, device=None):
        if data_path is None:
            self.data_path = Path("/workspace/Thesis/01_data/processed_data")
        else:
            self.data_path = Path(data_path)

        # Fixed results path for HPC
        self.results_dir = Path("/workspace/Thesis/05_ffnn_implementation/results/33features")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.horizons = [1, 6, 24, 48, 72]

        # Network architectures optimized for different horizons
        self.architectures = {
            1: {'hidden_sizes': [128, 64], 'dropout': 0.1, 'lr': 0.001, 'batch_size': 128},
            6: {'hidden_sizes': [256, 128], 'dropout': 0.15, 'lr': 0.0005, 'batch_size': 128},
            24: {'hidden_sizes': [256, 128, 64], 'dropout': 0.2, 'lr': 0.0005, 'batch_size': 64},
            48: {'hidden_sizes': [512, 256, 128], 'dropout': 0.25, 'lr': 0.0003, 'batch_size': 64},
            72: {'hidden_sizes': [512, 256, 128, 64], 'dropout': 0.3, 'lr': 0.0003, 'batch_size': 32}
        }

        self.results = {}

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        print("FFNN Forecasting System Initialized")
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

    def prepare_features(self, df):
        print("Preparing feature set...")

        # Filter numeric columns only
        numeric_columns = []
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                continue

        print(f"Found {len(numeric_columns)} numeric columns")
        df_numeric = df[numeric_columns].copy()

        # Select top features for efficiency (similar to 33-feature approach)
        priority_patterns = [
            'V1', 'N1', 'N2', 'V6', 'W1', 'F1_Sud', 'F1_Nord', 'B1_B2', 'V2', 'ZN',
            'temp', 'hdd', 'hour', 'pressure', 'humidity', 'solar', 'wind'
        ]

        selected_features = []

        # Add high priority features
        for pattern in priority_patterns:
            matching_cols = [col for col in numeric_columns if pattern.lower() in col.lower()]
            selected_features.extend(matching_cols[:2])  # Up to 2 per pattern
            if len(selected_features) >= 30:
                break

        # Fill remaining slots
        remaining_cols = [col for col in numeric_columns if col not in selected_features]
        selected_features.extend(remaining_cols[:max(0, 33-len(selected_features))])

        features = selected_features[:33]  # Target 33 features

        # Add basic temporal features if not present
        if 'hour' not in df_numeric.columns:
            df_numeric['hour'] = df_numeric.index.hour
            if 'hour' not in features:
                features.append('hour')

        # Add cyclical encoding
        if len(features) < 33:
            if 'hour_sin' not in df_numeric.columns:
                df_numeric['hour_sin'] = np.sin(2 * np.pi * df_numeric['hour'] / 24)
                df_numeric['hour_cos'] = np.cos(2 * np.pi * df_numeric['hour'] / 24)
                features.extend(['hour_sin', 'hour_cos'])

        # Ensure all features exist
        valid_features = [f for f in features if f in df_numeric.columns and not df_numeric[f].isna().all()]

        print(f"Final feature set: {len(valid_features)} features")

        return df_numeric[valid_features], valid_features

    def create_synthetic_heat_demand(self, df):
        print("Creating synthetic heat demand target...")

        # Find temperature or HDD column
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
            # Create synthetic temperature
            hours = np.arange(len(df))
            synthetic_temp = 10 + 8 * np.sin(2 * np.pi * hours / (24 * 365)) + 5 * np.sin(2 * np.pi * hours / 24)
            base_demand = np.maximum(0, 15.5 - synthetic_temp) * 3.5
            print("Created synthetic temperature for base demand")

        # Time factor
        hour = df.index.hour if hasattr(df.index, 'hour') else np.arange(len(df)) % 24
        time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24)

        # Seasonal factor
        if hasattr(df.index, 'dayofyear'):
            day_of_year = df.index.dayofyear
        else:
            day_of_year = np.arange(len(df)) % 365

        seasonal_factor = 1.0 + 0.3 * np.cos(2 * np.pi * day_of_year / 365)

        # Combine factors
        heat_demand = base_demand * time_factor * seasonal_factor
        heat_demand = np.maximum(heat_demand, 0.1)

        print(f"Synthetic heat demand range: {heat_demand.min():.2f} to {heat_demand.max():.2f} MWh")
        return heat_demand

    def create_targets(self, df):
        targets = pd.DataFrame(index=df.index)

        # Check for existing heat demand
        target_cols = ['heat_demand', 'demand', 'load', 'consumption']
        heat_demand = None

        for col in target_cols:
            if col in df.columns:
                heat_demand = df[col]
                print(f"Using existing {col} column")
                break

        if heat_demand is None:
            heat_demand = self.create_synthetic_heat_demand(df)

        # Create targets for all horizons
        for horizon in self.horizons:
            targets[f'target_{horizon}h'] = heat_demand.shift(-horizon)

        print(f"Created targets for horizons: {self.horizons}")
        return targets

    def prepare_data_splits(self, X_df, y_df):
        # Remove NaN rows
        valid_idx = ~(X_df.isna().any(axis=1) | y_df.isna().any(axis=1))
        X_df = X_df[valid_idx]
        y_df = y_df[valid_idx]

        print(f"Final dataset shape: {X_df.shape}")

        # Data splitting - 70% train, 15% val, 15% test
        n_train = int(len(X_df) * 0.7)
        n_val = int(len(X_df) * 0.15)

        X_train = X_df.iloc[:n_train].values
        X_val = X_df.iloc[n_train:n_train+n_val].values
        X_test = X_df.iloc[n_train+n_val:].values

        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_df, n_train, n_val

    def create_data_loaders(self, X_train, y_train, X_val, y_val, batch_size):
        train_dataset = FFNNDataset(X_train, y_train)
        val_dataset = FFNNDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_model(self, model, train_loader, val_loader, config, horizon):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # Training parameters
        epochs = 150 if horizon <= 6 else (200 if horizon <= 24 else 250)
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        train_losses = []
        val_losses = []

        print(f"  Training for {epochs} epochs with early stopping")

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            # Validation phase
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

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Load best model
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
        """Create comprehensive visualizations for FFNN results"""
        print("Creating visualizations...")

        # 1. Comprehensive Performance Analysis
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('FFNN Comprehensive Performance Analysis', fontsize=16, fontweight='bold')

        horizons = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        mape_scores = []

        for horizon in self.horizons:
            if horizon in results:
                horizons.append(f"{horizon}h")
                metrics = results[horizon]['metrics']
                r2_scores.append(metrics['r2_score'])
                rmse_scores.append(metrics['rmse'])
                mae_scores.append(metrics['mae'])
                mape_scores.append(metrics['mape'])

        # R2 Score comparison
        axes[0,0].bar(horizons, r2_scores, color='skyblue', alpha=0.8)
        axes[0,0].set_title('R² Score by Horizon')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_ylim(0, 1)
        for i, v in enumerate(r2_scores):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

        # RMSE comparison
        axes[0,1].bar(horizons, rmse_scores, color='lightcoral', alpha=0.8)
        axes[0,1].set_title('RMSE by Horizon')
        axes[0,1].set_ylabel('RMSE')
        for i, v in enumerate(rmse_scores):
            axes[0,1].text(i, v + max(rmse_scores)*0.01, f'{v:.2f}', ha='center', fontweight='bold')

        # MAE comparison
        axes[0,2].bar(horizons, mae_scores, color='lightgreen', alpha=0.8)
        axes[0,2].set_title('MAE by Horizon')
        axes[0,2].set_ylabel('MAE')
        for i, v in enumerate(mae_scores):
            axes[0,2].text(i, v + max(mae_scores)*0.01, f'{v:.2f}', ha='center', fontweight='bold')

        # MAPE comparison
        axes[1,0].bar(horizons, mape_scores, color='gold', alpha=0.8)
        axes[1,0].set_title('MAPE by Horizon')
        axes[1,0].set_ylabel('MAPE (%)')
        for i, v in enumerate(mape_scores):
            axes[1,0].text(i, v + max(mape_scores)*0.01, f'{v:.1f}', ha='center', fontweight='bold')

        # Performance degradation trend
        axes[1,1].plot(horizons, r2_scores, marker='o', linewidth=2, markersize=8, label='R² Score')
        axes[1,1].set_title('Performance Degradation Trend')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # Model complexity overview
        parameters = []
        for horizon in self.horizons:
            if horizon in results:
                param_count = sum(p.numel() for p in results[horizon]['model'].parameters())
                parameters.append(param_count/1000)  # In thousands

        axes[1,2].bar(horizons, parameters, color='mediumpurple', alpha=0.8)
        axes[1,2].set_title('Model Parameters by Horizon')
        axes[1,2].set_ylabel('Parameters (thousands)')
        for i, v in enumerate(parameters):
            axes[1,2].text(i, v + max(parameters)*0.01, f'{v:.1f}K', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'comprehensive_performance_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Model Analysis - Training Curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('FFNN Model Analysis - Training Curves', fontsize=16, fontweight='bold')

        for idx, horizon in enumerate(self.horizons[:6]):  # Max 6 subplots
            if horizon in results:
                row = idx // 3
                col = idx % 3

                train_losses = results[horizon]['train_losses']
                val_losses = results[horizon]['val_losses']

                axes[row, col].plot(train_losses, label='Training Loss', alpha=0.8)
                axes[row, col].plot(val_losses, label='Validation Loss', alpha=0.8)
                axes[row, col].set_title(f'{horizon}h Horizon Training')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Loss')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(self.horizons), 6):
            row = idx // 3
            col = idx % 3
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.savefig(self.results_dir / f'model_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Time Series Forecasts
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('FFNN Time Series Forecasts', fontsize=16, fontweight='bold')

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
                axes[row, col].set_title(f'{horizon}h Horizon (R²={metrics["r2_score"]:.3f})')
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

        # 4. Additional Analysis - Prediction Scatter Plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('FFNN Prediction Accuracy Analysis', fontsize=16, fontweight='bold')

        for idx, horizon in enumerate(self.horizons[:6]):
            if horizon in results:
                row = idx // 3
                col = idx % 3

                metrics = results[horizon]['metrics']
                predictions = metrics['predictions']
                actuals = metrics['actuals']

                # Scatter plot
                axes[row, col].scatter(actuals, predictions, alpha=0.6, s=10)

                # Perfect prediction line
                min_val = min(np.min(actuals), np.min(predictions))
                max_val = max(np.max(actuals), np.max(predictions))
                axes[row, col].plot([min_val, max_val], [min_val, max_val],
                                    'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')

                axes[row, col].set_title(f'{horizon}h Horizon\nR²={metrics["r2_score"]:.3f}, RMSE={metrics["rmse"]:.2f}')
                axes[row, col].set_xlabel('Actual Heat Demand (MWh)')
                axes[row, col].set_ylabel('Predicted Heat Demand (MWh)')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(self.horizons), 6):
            if idx < 6:
                row = idx // 3
                col = idx % 3
                fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.savefig(self.results_dir / f'prediction_accuracy_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 5. SVR Comparison Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('FFNN vs SVR Performance Comparison', fontsize=16, fontweight='bold')

        # SVR baseline results (from your earlier results)
        svr_baseline = {1: 0.989, 6: 0.930, 24: 0.864, 48: 0.793, 72: 0.739}

        ffnn_scores = []
        svr_scores = []
        improvements = []
        labels = []

        for horizon in self.horizons:
            if horizon in results and horizon in svr_baseline:
                labels.append(f"{horizon}h")
                ffnn_score = results[horizon]['metrics']['r2_score']
                svr_score = svr_baseline[horizon]
                ffnn_scores.append(ffnn_score)
                svr_scores.append(svr_score)
                improvement = ((ffnn_score - svr_score) / svr_score) * 100
                improvements.append(improvement)

        # Side-by-side comparison
        x_pos = np.arange(len(labels))
        width = 0.35

        ax1.bar(x_pos - width/2, svr_scores, width, label='SVR', alpha=0.8, color='lightcoral')
        ax1.bar(x_pos + width/2, ffnn_scores, width, label='FFNN', alpha=0.8, color='skyblue')
        ax1.set_title('R² Score Comparison')
        ax1.set_ylabel('R² Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Improvement percentages
        colors = ['green' if imp >= 0 else 'red' for imp in improvements]
        bars = ax2.bar(labels, improvements, color=colors, alpha=0.7)
        ax2.set_title('FFNN Improvement over SVR')
        ax2.set_ylabel('Improvement (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # Add improvement values on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                     f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                     fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'ffnn_vs_svr_comparison_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"All visualizations saved to {self.results_dir}")

    def run_ffnn_forecasting(self):
        print("Starting FFNN Forecasting Implementation")
        print("="*50)

        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        # Load and prepare data
        df = self.load_data()
        X_df, feature_names = self.prepare_features(df)
        y_df = self.create_targets(df)

        X_train, X_val, X_test, y_df, n_train, n_val = self.prepare_data_splits(X_df, y_df)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        # Process each horizon
        for horizon in self.horizons:
            print(f"\nProcessing {horizon}h horizon")
            print("-" * 30)

            # Prepare target data
            y_train = y_df.iloc[:n_train][f'target_{horizon}h'].dropna().values
            y_val = y_df.iloc[n_train:n_train+n_val][f'target_{horizon}h'].dropna().values
            y_test = y_df.iloc[n_train+n_val:][f'target_{horizon}h'].dropna().values

            # Align arrays
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

            # Get architecture config
            config = self.architectures[horizon]

            # Create model
            model = DistrictHeatingFFNN(
                input_size=X_train_h.shape[1],
                hidden_sizes=config['hidden_sizes'],
                dropout_rate=config['dropout']
            ).to(self.device)

            print(f"  Model architecture: {X_train_h.shape[1]} -> {' -> '.join(map(str, config['hidden_sizes']))} -> 1")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(
                X_train_h, y_train_h, X_val_h, y_val_h, config['batch_size']
            )

            # Train model
            model, train_losses, val_losses = self.train_model(
                model, train_loader, val_loader, config, horizon
            )

            # Evaluate on test set
            final_metrics = self.evaluate_model(model, X_test_h, y_test_h, horizon)

            # Store results
            results[horizon] = {
                'model': model,
                'model_type': f"FFNN ({' -> '.join(map(str, config['hidden_sizes']))})",
                'metrics': final_metrics,
                'features': feature_names,
                'scaler': scaler,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config
            }

            # Print results
            metrics = final_metrics
            print(f"Results for {horizon}h:")
            print(f"  R2 Score: {metrics['r2_score']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")

        end_time = datetime.now()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time}")

        # Create visualizations
        self.create_visualizations(results, timestamp)

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
                'train_losses': result['train_losses'],
                'val_losses': result['val_losses'],
                'config': result['config']
            }

        results_file = self.results_dir / f"ffnn_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(save_results, f)

        # Save model states separately
        for horizon, result in results.items():
            model_file = self.results_dir / f"ffnn_model_{horizon}h_{timestamp}.pth"
            torch.save(result['model'].state_dict(), model_file)

        # Save scaler
        scaler_file = self.results_dir / f"ffnn_scaler_{timestamp}.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(results[list(results.keys())[0]]['scaler'], f)

        # Save predictions for analysis
        predictions_data = {}
        for horizon, result in results.items():
            metrics = result['metrics']
            predictions_data[f'{horizon}h_predictions'] = metrics['predictions'].tolist()
            predictions_data[f'{horizon}h_actuals'] = metrics['actuals'].tolist()

        predictions_file = self.results_dir / f"ffnn_predictions_{timestamp}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        # Save summary CSV
        summary_data = []
        for horizon, result in results.items():
            metrics = result['metrics']
            summary_data.append({
                'timestamp': timestamp,
                'horizon': horizon,
                'model_type': result['model_type'],
                'features_count': len(result['features']),
                'parameters': sum(p.numel() for p in result['model'].parameters()),
                'r2_score': metrics['r2_score'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / f"ffnn_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        # Save feature importance (based on model weights for first layer)
        feature_importance_data = []
        first_horizon = list(results.keys())[0]
        first_model = results[first_horizon]['model']
        features = results[first_horizon]['features']

        # Get first layer weights
        first_layer_weights = first_model.network[0].weight.data.cpu().numpy()
        importance_scores = np.mean(np.abs(first_layer_weights), axis=0)

        for i, feature in enumerate(features):
            feature_importance_data.append({
                'feature': feature,
                'importance_score': importance_scores[i],
                'rank': i + 1
            })

        feature_df = pd.DataFrame(feature_importance_data).sort_values('importance_score', ascending=False)
        feature_df['rank'] = range(1, len(feature_df) + 1)
        feature_file = self.results_dir / f"ffnn_feature_importance_{timestamp}.csv"
        feature_df.to_csv(feature_file, index=False)

        print(f"Results saved successfully!")
        print(f"- Main results: ffnn_results_{timestamp}.pkl")
        print(f"- Model states: ffnn_model_[horizon]h_{timestamp}.pth")
        print(f"- Summary: ffnn_summary_{timestamp}.csv")
        print(f"- Predictions: ffnn_predictions_{timestamp}.json")
        print(f"- Feature importance: ffnn_feature_importance_{timestamp}.csv")
        print(f"- Visualizations: [multiple PNG files]")

    def print_final_summary(self, results):
        print("\nFFNN RESULTS SUMMARY")
        print("=" * 50)

        print(f"{'Horizon':>8} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
        print("-" * 48)

        for horizon in self.horizons:
            if horizon in results:
                metrics = results[horizon]['metrics']
                print(f"{horizon:>7}h {metrics['r2_score']:>7.3f} {metrics['rmse']:>7.3f} {metrics['mae']:>7.3f} {metrics['mape']:>7.1f}")

    def compare_with_svr_baseline(self, results):
        print("\nCOMPARISON WITH SVR BASELINE")
        print("=" * 50)

        # SVR baseline results
        svr_baseline = {1: 0.989, 6: 0.930, 24: 0.864, 48: 0.793, 72: 0.739}

        print(f"{'Horizon':>8} {'SVR':>8} {'FFNN':>8} {'Improvement':>12}")
        print("-" * 40)

        total_improvement = 0
        valid_comparisons = 0

        for horizon in self.horizons:
            if horizon in results and horizon in svr_baseline:
                svr_score = svr_baseline[horizon]
                ffnn_score = results[horizon]['metrics']['r2_score']
                improvement = ((ffnn_score - svr_score) / svr_score) * 100
                total_improvement += improvement
                valid_comparisons += 1

                print(f"{horizon:>7}h {svr_score:>7.3f} {ffnn_score:>7.3f} {improvement:>+10.1f}%")

        if valid_comparisons > 0:
            avg_improvement = total_improvement / valid_comparisons
            print("-" * 40)
            print(f"Average improvement: {avg_improvement:+.1f}%")


def main():
    print("FFNN Implementation - HPC Execution (33 Features)")
    print("Schweinfurt District Heating Network Forecasting")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ffnn_system = FFNNForecasting()

    start_time = datetime.now()

    results, timestamp = ffnn_system.run_ffnn_forecasting()
    ffnn_system.save_results(results, timestamp)
    ffnn_system.print_final_summary(results)
    ffnn_system.compare_with_svr_baseline(results)

    end_time = datetime.now()
    execution_time = end_time - start_time

    print(f"\nFFNN Implementation Complete!")
    print(f"Total Execution Time: {execution_time}")
    print(f"Results saved in: {ffnn_system.results_dir}")
    print(f"Timestamp: {timestamp}")

if __name__ == "__main__":
    main()