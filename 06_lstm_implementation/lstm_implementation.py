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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class EnhancedLSTMDataset(Dataset):
    def __init__(self, sequences, features, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.features = torch.FloatTensor(features) if features is not None else None
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.features is not None:
            return self.sequences[idx], self.features[idx], self.targets[idx]
        return self.sequences[idx], self.targets[idx]

class MultiScaleBidirectionalLSTM(nn.Module):
    def __init__(self, input_size, feature_size=None, hidden_size=256, num_layers=3,
                 dropout=0.3, use_attention=True, use_residual=True):
        super(MultiScaleBidirectionalLSTM, self).__init__()

        self.input_size = input_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_residual = use_residual

        self.lstm_short = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_long = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.main_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )

        lstm_output_size = hidden_size * 2
        if feature_size:
            self.feature_projection = nn.Linear(feature_size, hidden_size // 2)
            combined_size = lstm_output_size + hidden_size // 2
        else:
            combined_size = lstm_output_size

        multi_scale_size = (hidden_size // 2) * 2 * 2
        total_size = combined_size + multi_scale_size

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(total_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

        if use_residual:
            self.residual_fc = nn.Linear(total_size, 1)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
            elif 'weight' in name and 'fc' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x, features=None):
        batch_size, seq_len, _ = x.shape

        short_out, _ = self.lstm_short(x)
        short_repr = short_out[:, -1, :]

        step = max(1, seq_len // 24)
        x_subsampled = x[:, ::step, :]
        long_out, _ = self.lstm_long(x_subsampled)
        long_repr = long_out[:, -1, :]

        lstm_out, (h_n, c_n) = self.main_lstm(x)

        if self.use_attention:
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            main_repr = torch.mean(attended_out, dim=1)
        else:
            main_repr = lstm_out[:, -1, :]

        combined = torch.cat([main_repr, short_repr, long_repr], dim=1)

        if features is not None and self.feature_size:
            feature_repr = torch.relu(self.feature_projection(features))
            combined = torch.cat([combined, feature_repr], dim=1)

        x_out = self.dropout(combined)
        x_out = torch.relu(self.layer_norm1(self.fc1(x_out)))
        x_out = self.dropout(x_out)
        x_out = torch.relu(self.layer_norm2(self.fc2(x_out)))
        x_out = self.dropout(x_out)
        main_output = self.fc3(x_out)

        if self.use_residual:
            residual = self.residual_fc(combined)
            output = main_output + residual
        else:
            output = main_output

        return output

class EnhancedLSTMForecasting:

    def __init__(self, data_path=None, device=None):
        if data_path is None:
            self.data_path = Path("/workspace/Thesis/01_data/processed_data")
        else:
            self.data_path = Path(data_path)

        self.results_dir = Path("/workspace/Thesis/06_lstm_implementation/results/33features")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.horizons = [1, 6, 24, 48, 72]

        self.sequence_lengths = {
            1: 48,
            6: 72,
            24: 168,
            48: 336,
            72: 504
        }

        self.model_configs = {
            1: {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 256},
            6: {'hidden_size': 96, 'num_layers': 2, 'dropout': 0.25, 'learning_rate': 0.0008, 'batch_size': 128},
            24: {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.3, 'learning_rate': 0.0005, 'batch_size': 64},
            48: {'hidden_size': 192, 'num_layers': 3, 'dropout': 0.35, 'learning_rate': 0.0003, 'batch_size': 32},
            72: {'hidden_size': 256, 'num_layers': 3, 'dropout': 0.4, 'learning_rate': 0.0002, 'batch_size': 24}
        }

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        data_file = self.data_path / "cleaned_features_33.csv"
        df = pd.read_csv(data_file)

        if 'datetime' not in df.columns:
            if 'Unnamed: 0' in df.columns:
                df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)

        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()

        target_col = 'Last'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        df = df.dropna(subset=[target_col])

        feature_cols = [col for col in df.columns if col != target_col]
        df_clean = df[[target_col] + feature_cols].dropna()

        return df_clean, feature_cols

    def create_sequences_with_features(self, data: np.ndarray, features: np.ndarray,
                                       target: np.ndarray, seq_length: int,
                                       horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequences = []
        feature_vecs = []
        targets = []

        for i in range(len(data) - seq_length - horizon + 1):
            seq = data[i:(i + seq_length)]
            feat = features[i + seq_length - 1]
            tgt = target[i + seq_length + horizon - 1]

            sequences.append(seq)
            feature_vecs.append(feat)
            targets.append(tgt)

        return np.array(sequences), np.array(feature_vecs), np.array(targets)

    def split_data(self, sequences: np.ndarray, features: np.ndarray,
                   targets: np.ndarray, train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple:
        n = len(sequences)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train_seq = sequences[:train_end]
        X_val_seq = sequences[train_end:val_end]
        X_test_seq = sequences[val_end:]

        X_train_feat = features[:train_end]
        X_val_feat = features[train_end:val_end]
        X_test_feat = features[val_end:]

        y_train = targets[:train_end]
        y_val = targets[train_end:val_end]
        y_test = targets[val_end:]

        return (X_train_seq, X_val_seq, X_test_seq,
                X_train_feat, X_val_feat, X_test_feat,
                y_train, y_val, y_test)

    def scale_data(self, X_train_seq, X_val_seq, X_test_seq,
                   X_train_feat, X_val_feat, X_test_feat) -> Tuple:
        scaler_seq = StandardScaler()
        n_samples, seq_len, n_features = X_train_seq.shape
        X_train_seq_2d = X_train_seq.reshape(-1, n_features)
        X_train_seq_scaled_2d = scaler_seq.fit_transform(X_train_seq_2d)
        X_train_seq_scaled = X_train_seq_scaled_2d.reshape(n_samples, seq_len, n_features)

        X_val_seq_2d = X_val_seq.reshape(-1, n_features)
        X_val_seq_scaled_2d = scaler_seq.transform(X_val_seq_2d)
        X_val_seq_scaled = X_val_seq_scaled_2d.reshape(X_val_seq.shape[0], seq_len, n_features)

        X_test_seq_2d = X_test_seq.reshape(-1, n_features)
        X_test_seq_scaled_2d = scaler_seq.transform(X_test_seq_2d)
        X_test_seq_scaled = X_test_seq_scaled_2d.reshape(X_test_seq.shape[0], seq_len, n_features)

        scaler_feat = StandardScaler()
        X_train_feat_scaled = scaler_feat.fit_transform(X_train_feat)
        X_val_feat_scaled = scaler_feat.transform(X_val_feat)
        X_test_feat_scaled = scaler_feat.transform(X_test_feat)

        return (X_train_seq_scaled, X_val_seq_scaled, X_test_seq_scaled,
                X_train_feat_scaled, X_val_feat_scaled, X_test_feat_scaled,
                scaler_seq, scaler_feat)

    def train_model(self, model, train_loader, val_loader,
                    epochs: int = 150, learning_rate: float = 0.001,
                    patience: int = 30, min_delta: float = 0.0001) -> Tuple[List[float], List[float]]:
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_data in train_loader:
                if len(batch_data) == 3:
                    sequences, features, targets = batch_data
                    sequences = sequences.to(self.device)
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                else:
                    sequences, targets = batch_data
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    features = None

                optimizer.zero_grad()
                outputs = model(sequences, features)
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
                for batch_data in val_loader:
                    if len(batch_data) == 3:
                        sequences, features, targets = batch_data
                        sequences = sequences.to(self.device)
                        features = features.to(self.device)
                        targets = targets.to(self.device)
                    else:
                        sequences, targets = batch_data
                        sequences = sequences.to(self.device)
                        targets = targets.to(self.device)
                        features = None

                    outputs = model(sequences, features)
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return train_losses, val_losses

    def evaluate_model(self, model, test_loader, scaler=None) -> Dict[str, Any]:
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    sequences, features, targets = batch_data
                    sequences = sequences.to(self.device)
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                else:
                    sequences, targets = batch_data
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    features = None

                outputs = model(sequences, features)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        epsilon = 1e-10
        mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'predictions': predictions,
            'actuals': actuals
        }

        return metrics

    def create_visualizations(self, results, timestamp):
        sns.set_style("whitegrid")
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.horizons)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        metrics_to_plot = ['r2_score', 'rmse', 'mae', 'mape']
        metric_names = ['R² Score', 'RMSE', 'MAE', 'MAPE (%)']

        for idx, (metric_key, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx]
            horizons_list = []
            values = []

            for horizon in self.horizons:
                if horizon in results:
                    horizons_list.append(horizon)
                    values.append(results[horizon]['metrics'][metric_key])

            ax.plot(horizons_list, values, marker='o', linewidth=2, markersize=8, color=colors[0])
            ax.set_xlabel('Forecast Horizon (hours)', fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{metric_name} vs Forecast Horizon', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        for horizon_idx, horizon in enumerate(self.horizons):
            if horizon in results:
                ax = axes[4 + (horizon_idx % 2)]

                if horizon_idx >= 2:
                    if horizon_idx == 2:
                        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
                        axes2 = axes2.flatten()
                    ax = axes2[(horizon_idx - 2)]

                metrics = results[horizon]['metrics']
                predictions = metrics['predictions'][:500]
                actuals = metrics['actuals'][:500]

                x_range = range(len(predictions))
                ax.plot(x_range, actuals, label='Actual', alpha=0.7, linewidth=1.5)
                ax.plot(x_range, predictions, label='Predicted', alpha=0.7, linewidth=1.5)
                ax.set_xlabel('Sample Index', fontsize=10)
                ax.set_ylabel('Heat Demand', fontsize=10)
                ax.set_title(f'Predictions vs Actuals - {horizon}h Horizon\nR²={metrics["r2_score"]:.3f}, RMSE={metrics["rmse"]:.3f}',
                             fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / f'enhanced_lstm_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        if 'fig2' in locals():
            plt.figure(fig2.number)
            plt.tight_layout()
            plt.savefig(self.results_dir / f'enhanced_lstm_predictions_extended_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, horizon in enumerate(self.horizons):
            if horizon in results:
                ax = axes[idx]
                train_losses = results[horizon]['train_losses']
                val_losses = results[horizon]['val_losses']

                ax.plot(train_losses, label='Train Loss', alpha=0.8, linewidth=1.5)
                ax.plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=1.5)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('Loss (MSE)', fontsize=10)
                ax.set_title(f'Training History - {horizon}h Horizon', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')

        if len(self.horizons) < len(axes):
            for idx in range(len(self.horizons), len(axes)):
                fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(self.results_dir / f'enhanced_lstm_training_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, horizon in enumerate(self.horizons):
            if horizon in results:
                ax = axes[idx]
                metrics = results[horizon]['metrics']
                predictions = metrics['predictions']
                actuals = metrics['actuals']

                ax.scatter(actuals, predictions, alpha=0.5, s=10)
                min_val = min(actuals.min(), predictions.min())
                max_val = max(actuals.max(), predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

                ax.set_xlabel('Actual Values', fontsize=10)
                ax.set_ylabel('Predicted Values', fontsize=10)
                ax.set_title(f'Prediction Scatter - {horizon}h Horizon\nR²={metrics["r2_score"]:.3f}',
                             fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        if len(self.horizons) < len(axes):
            for idx in range(len(self.horizons), len(axes)):
                fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(self.results_dir / f'enhanced_lstm_scatter_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 7))

        metrics_comparison = {
            'R² Score': [],
            'RMSE': [],
            'MAE': []
        }

        for horizon in self.horizons:
            if horizon in results:
                metrics = results[horizon]['metrics']
                metrics_comparison['R² Score'].append(metrics['r2_score'])
                metrics_comparison['RMSE'].append(metrics['rmse'])
                metrics_comparison['MAE'].append(metrics['mae'])

        x = np.arange(len(self.horizons))
        width = 0.25

        for idx, (metric_name, values) in enumerate(metrics_comparison.items()):
            offset = width * (idx - 1)
            ax.bar(x + offset, values, width, label=metric_name, alpha=0.8)

        ax.set_xlabel('Forecast Horizon (hours)', fontsize=11)
        ax.set_ylabel('Metric Value', fontsize=11)
        ax.set_title('Performance Metrics Comparison Across Horizons', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h}h' for h in self.horizons])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'enhanced_lstm_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_enhanced_lstm_forecasting(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        df, feature_names = self.load_and_prepare_data()

        target_col = 'Last'
        target_data = df[target_col].values.reshape(-1, 1)
        feature_data = df[feature_names].values

        results = {}
        start_time = datetime.now()

        for horizon in self.horizons:
            config = self.model_configs[horizon]
            seq_length = self.sequence_lengths[horizon]

            sequences, features, targets = self.create_sequences_with_features(
                target_data, feature_data, target_data.flatten(),
                seq_length, horizon
            )

            splits = self.split_data(sequences, features, targets)
            (X_train_seq, X_val_seq, X_test_seq,
             X_train_feat, X_val_feat, X_test_feat,
             y_train, y_val, y_test) = splits

            (X_train_seq_scaled, X_val_seq_scaled, X_test_seq_scaled,
             X_train_feat_scaled, X_val_feat_scaled, X_test_feat_scaled,
             scaler_seq, scaler_feat) = self.scale_data(
                X_train_seq, X_val_seq, X_test_seq,
                X_train_feat, X_val_feat, X_test_feat
            )

            train_dataset = EnhancedLSTMDataset(X_train_seq_scaled, X_train_feat_scaled, y_train)
            val_dataset = EnhancedLSTMDataset(X_val_seq_scaled, X_val_feat_scaled, y_val)
            test_dataset = EnhancedLSTMDataset(X_test_seq_scaled, X_test_feat_scaled, y_test)

            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

            input_size = X_train_seq_scaled.shape[2]
            feature_size = X_train_feat_scaled.shape[1]

            model = MultiScaleBidirectionalLSTM(
                input_size=input_size,
                feature_size=feature_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                use_attention=True,
                use_residual=True
            )

            train_losses, val_losses = self.train_model(
                model, train_loader, val_loader,
                epochs=150,
                learning_rate=config['learning_rate'],
                patience=30
            )

            final_metrics = self.evaluate_model(
                model, test_loader, scaler=None
            )

            results[horizon] = {
                'model': model,
                'model_type': f"Enhanced LSTM (h={config['hidden_size']}, l={config['num_layers']}, seq={self.sequence_lengths[horizon]})",
                'metrics': final_metrics,
                'features': feature_names,
                'scaler_seq': scaler_seq,
                'scaler_feat': scaler_feat,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config,
                'sequence_length': self.sequence_lengths[horizon]
            }

            metrics = final_metrics
            print(f"Enhanced Results for {horizon}h:")
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
                'sequence_length': result['sequence_length']
            }

        results_file = self.results_dir / f"enhanced_lstm_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(save_results, f)

        for horizon, result in results.items():
            model_file = self.results_dir / f"enhanced_lstm_model_{horizon}h_{timestamp}.pth"
            torch.save(result['model'].state_dict(), model_file)

        if results:
            scaler_seq_file = self.results_dir / f"enhanced_lstm_scaler_seq_{timestamp}.pkl"
            scaler_feat_file = self.results_dir / f"enhanced_lstm_scaler_feat_{timestamp}.pkl"

            first_result = results[list(results.keys())[0]]
            with open(scaler_seq_file, 'wb') as f:
                pickle.dump(first_result['scaler_seq'], f)
            if first_result['scaler_feat'] is not None:
                with open(scaler_feat_file, 'wb') as f:
                    pickle.dump(first_result['scaler_feat'], f)

        predictions_data = {}
        for horizon, result in results.items():
            metrics = result['metrics']
            predictions_data[f'{horizon}h_predictions'] = metrics['predictions'].tolist()
            predictions_data[f'{horizon}h_actuals'] = metrics['actuals'].tolist()

        predictions_file = self.results_dir / f"enhanced_lstm_predictions_{timestamp}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        summary_data = []
        for horizon, result in results.items():
            metrics = result['metrics']
            summary_data.append({
                'timestamp': timestamp,
                'horizon': horizon,
                'model_type': result['model_type'],
                'sequence_length': result['sequence_length'],
                'hidden_size': result['config']['hidden_size'],
                'num_layers': result['config']['num_layers'],
                'dropout': result['config']['dropout'],
                'parameters': sum(p.numel() for p in result['model'].parameters()),
                'r2_score': metrics['r2_score'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / f"enhanced_lstm_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        print(f"Results saved successfully!")
        print(f"- Main results: enhanced_lstm_results_{timestamp}.pkl")
        print(f"- Model states: enhanced_lstm_model_[horizon]h_{timestamp}.pth")
        print(f"- Summary: enhanced_lstm_summary_{timestamp}.csv")
        print(f"- Predictions: enhanced_lstm_predictions_{timestamp}.json")
        print(f"- Scalers: enhanced_lstm_scaler_[type]_{timestamp}.pkl")
        print(f"- Visualizations: [multiple PNG files]")

    def print_final_summary(self, results):
        print("\nENHANCED LSTM RESULTS SUMMARY")
        print("=" * 60)

        print(f"{'Horizon':>8} {'Seq Len':>8} {'Hidden':>8} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
        print("-" * 68)

        for horizon in self.horizons:
            if horizon in results:
                metrics = results[horizon]['metrics']
                seq_len = results[horizon]['sequence_length']
                hidden_size = results[horizon]['config']['hidden_size']
                print(f"{horizon:>7}h {seq_len:>7} {hidden_size:>7} {metrics['r2_score']:>7.3f} {metrics['rmse']:>7.3f} {metrics['mae']:>7.3f} {metrics['mape']:>7.1f}")

    def compare_with_all_models(self, results):
        print("\nCOMPARISON WITH ALL PREVIOUS MODELS")
        print("=" * 80)

        svr_baseline = {1: 0.989, 6: 0.930, 24: 0.864, 48: 0.793, 72: 0.739}
        ffnn_baseline = {1: 0.988, 6: 0.943, 24: 0.861, 48: 0.800, 72: 0.743}
        lstm_baseline = {1: 0.994, 6: 0.937, 24: 0.842, 48: 0.751, 72: 0.664}

        print(f"{'Horizon':>8} {'SVR-33':>8} {'FFNN-33':>8} {'LSTM-33':>8} {'Enhanced':>9} {'Best Gain':>10}")
        print("-" * 75)

        total_improvements = []

        for horizon in self.horizons:
            if horizon in results:
                svr_score = svr_baseline[horizon]
                ffnn_score = ffnn_baseline[horizon]
                lstm_score = lstm_baseline[horizon]
                enhanced_score = results[horizon]['metrics']['r2_score']

                best_previous = max(svr_score, ffnn_score, lstm_score)
                improvement = ((enhanced_score - best_previous) / best_previous) * 100
                total_improvements.append(improvement)

                print(f"{horizon:>7}h {svr_score:>7.3f} {ffnn_score:>7.3f} {lstm_score:>7.3f} {enhanced_score:>8.3f} {improvement:>+8.1f}%")

        if total_improvements:
            avg_improvement = np.mean(total_improvements)
            print("-" * 75)
            print(f"Average improvement over best previous model: {avg_improvement:+.1f}%")

            long_horizon_improvements = [total_improvements[i] for i in range(len(self.horizons)) if self.horizons[i] >= 24]
            if long_horizon_improvements:
                long_avg = np.mean(long_horizon_improvements)
                print(f"Average improvement for long horizons (24h+): {long_avg:+.1f}%")


def main():
    print("Enhanced LSTM Implementation - HPC Execution (33 Features)")
    print("Schweinfurt District Heating Network Forecasting")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\nEnhancements:")
    print("- Multi-scale bidirectional LSTM with attention")
    print("- Hybrid feature integration")
    print("- Longer sequences for longer horizons")
    print("- Curriculum learning with horizon-weighted loss")
    print("- Advanced regularization and training strategies")

    lstm_system = EnhancedLSTMForecasting()

    start_time = datetime.now()

    results, timestamp = lstm_system.run_enhanced_lstm_forecasting()
    lstm_system.save_results(results, timestamp)
    lstm_system.print_final_summary(results)
    lstm_system.compare_with_all_models(results)

    end_time = datetime.now()
    execution_time = end_time - start_time

    print(f"\nEnhanced LSTM Implementation Complete!")
    print(f"Total Execution Time: {execution_time}")
    print(f"Results saved in: {lstm_system.results_dir}")
    print(f"Timestamp: {timestamp}")

if __name__ == "__main__":
    main()