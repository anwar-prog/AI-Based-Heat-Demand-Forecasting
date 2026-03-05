import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.ensemble import VotingRegressor
import warnings
import os
warnings.filterwarnings('ignore')

class AdvancedMLRForecasting:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.scalers = {}

        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_dir, '01_data', 'processed_data')

        self.results_dir = os.path.join(self.base_dir, '03_baseline_models', 'results', '33features')
        self.plots_dir_main = self.results_dir
        self.plots_dir_results = self.results_dir
        self.plot_dirs = [self.results_dir]

        self.base_features = [
            'V1_expected_delta_T', 'N1_expected_supply_temp', 'N2_expected_delta_T',
            'V1_expected_supply_temp', 'V6_expected_supply_temp', 'W1_expected_supply_temp',
            'V6_expected_delta_T', 'W1_expected_delta_T', 'F1_Sud_expected_supply_temp',
            'F1_Nord_expected_supply_temp', 'B1_B2_expected_supply_temp', 'F1_Sud_expected_delta_T',
            'B1_B2_expected_delta_T', 'dewpt', 'hdd_15_5', 'ZN_expected_delta_T',
            'N2_expected_supply_temp', 'N1_expected_delta_T', 'F1_Nord_expected_delta_T',
            'temp', 'app_temp', 'V2_expected_supply_temp', 'V2_expected_delta_T',
            'ZN_expected_supply_temp'
        ]

        self.horizons = [1, 6, 24, 48, 72]
        self.zones = ['V1', 'N1', 'N2', 'V6', 'W1', 'F1_Sud', 'F1_Nord', 'B1_B2', 'V2', 'ZN']

    def save_plot(self, filename):
        for plot_dir in self.plot_dirs:
            os.makedirs(plot_dir, exist_ok=True)
            full_path = os.path.join(plot_dir, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to both: {self.plots_dir_results} and {self.plots_dir_main}")

    def load_and_prepare_data(self, filename='merged_dataset.csv'):
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at: {filepath}")

        df = pd.read_csv(filepath)

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Data loaded from: {filepath}")

        missing_values = df[self.base_features].isnull().sum()
        if missing_values.sum() > 0:
            df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def create_enhanced_heat_demand(self, df):
        zone_weights = {
            'V1': 0.17, 'N1': 0.14, 'N2': 0.07, 'V6': 0.05, 'W1': 0.15,
            'F1_Sud': 0.12, 'F1_Nord': 0.10, 'B1_B2': 0.16, 'V2': 0.04, 'ZN': 0.00
        }

        base_demand = df['hdd_15_5'] * 3.5

        temp_contribution = 0
        for zone, weight in zone_weights.items():
            if f'{zone}_expected_supply_temp' in df.columns:
                temp_norm = (df[f'{zone}_expected_supply_temp'] - 65) / 45
                temp_contribution += temp_norm * weight * 18

        hour = df.index.hour
        time_factor = 1.0 + 0.45 * (
                np.exp(-((hour - 7) ** 2) / 14) +
                np.exp(-((hour - 19) ** 2) / 18) +
                0.4 * np.exp(-((hour - 12) ** 2) / 25) +
                0.2 * np.exp(-((hour - 22) ** 2) / 15)
        )

        dow = df.index.dayofweek
        enhanced_dow_factor = np.array([1.25, 1.22, 1.20, 1.18, 1.15, 0.75, 0.70])[dow]

        month = df.index.month
        seasonal_factor = 1.0 + 0.35 * np.cos(2 * np.pi * (month - 1) / 12) + 0.1 * np.cos(4 * np.pi * (month - 1) / 12)

        week_of_month = (df.index.day - 1) // 7 + 1
        monthly_pattern = 1.0 + 0.08 * np.sin(2 * np.pi * week_of_month / 4)

        day_of_year = df.index.dayofyear
        yearly_pattern = 1.0 + 0.05 * np.sin(2 * np.pi * day_of_year / 365.25)

        heat_demand = (base_demand + temp_contribution) * time_factor * enhanced_dow_factor * seasonal_factor * monthly_pattern * yearly_pattern

        np.random.seed(42)
        noise = np.random.normal(0, heat_demand.std() * 0.02, len(heat_demand))
        heat_demand += noise
        heat_demand = np.maximum(heat_demand, 0.1)

        print(f"Heat demand created: {heat_demand.min():.2f} to {heat_demand.max():.2f} MWh")
        return heat_demand

    def add_comprehensive_features(self, df, heat_demand):
        enhanced_features = {}

        enhanced_features['demand_lag_24h'] = heat_demand.shift(24)
        enhanced_features['demand_lag_12h'] = heat_demand.shift(12)
        enhanced_features['demand_rolling_mean_24h'] = heat_demand.rolling(24).mean()

        enhanced_features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        enhanced_features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        enhanced_features['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        enhanced_features['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        enhanced_features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        enhanced_features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        enhanced_features['temp_change_6h'] = df['temp'].diff(6)
        enhanced_features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        enhanced_features['is_business_day'] = ((df.index.dayofweek < 5)).astype(int)
        enhanced_features['demand_lag_23h'] = heat_demand.shift(23)
        enhanced_features['demand_lag_25h'] = heat_demand.shift(25)
        enhanced_features['daily_volatility'] = heat_demand.rolling(24).std() / (heat_demand.rolling(24).mean() + 1e-6)

        enhanced_features['demand_lag_48h'] = heat_demand.shift(48)
        enhanced_features['temp_momentum_48h'] = (df['temp'].rolling(48).mean() - df['temp'].rolling(96).mean())
        enhanced_features['is_friday'] = (df.index.dayofweek == 4).astype(int)
        enhanced_features['is_sunday'] = (df.index.dayofweek == 6).astype(int)

        weekly_lags = [164, 165, 166, 167, 168, 169, 170, 171, 172]
        for lag in weekly_lags:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        week_position = (df.index.dayofweek * 24 + df.index.hour) / 168.0
        for harmonic in [1, 2, 3, 4]:
            enhanced_features[f'week_sin_{harmonic}'] = np.sin(2 * np.pi * harmonic * week_position)
            enhanced_features[f'week_cos_{harmonic}'] = np.cos(2 * np.pi * harmonic * week_position)

        for window in [48, 60, 72, 84, 96]:
            enhanced_features[f'temp_trend_{window}h'] = df['temp'].rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0, raw=False
            )
            enhanced_features[f'temp_curvature_{window}h'] = df['temp'].rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 2)[0] if len(x) == window else 0, raw=False
            )

        day_of_year = df.index.dayofyear
        for harmonic in [1, 2, 3]:
            enhanced_features[f'season_sin_{harmonic}'] = np.sin(2 * np.pi * harmonic * day_of_year / 365.25)
            enhanced_features[f'season_cos_{harmonic}'] = np.cos(2 * np.pi * harmonic * day_of_year / 365.25)

        for weeks in [1, 2, 3, 4]:
            hours = weeks * 168
            enhanced_features[f'demand_rolling_mean_{hours}h'] = heat_demand.rolling(hours).mean()
            enhanced_features[f'demand_rolling_std_{hours}h'] = heat_demand.rolling(hours).std()
            enhanced_features[f'demand_trend_{hours}h'] = heat_demand.rolling(hours).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == hours else 0, raw=False
            )

        enhanced_features['day_of_month'] = df.index.day
        enhanced_features['week_of_year'] = df.index.isocalendar().week
        enhanced_features['days_to_month_end'] = (df.index + pd.offsets.MonthEnd(0) - df.index).days
        enhanced_features['is_month_start'] = (df.index.day <= 3).astype(int)
        enhanced_features['is_month_middle'] = ((df.index.day >= 13) & (df.index.day <= 17)).astype(int)
        enhanced_features['is_month_end'] = (df.index.day >= 26).astype(int)

        enhanced_features['hdd_trend_72h'] = df['hdd_15_5'].rolling(72).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 72 else 0, raw=False
        )
        enhanced_features['hdd_acceleration_72h'] = df['hdd_15_5'].diff(72).diff(72)
        enhanced_features['dewpt_trend_72h'] = df['dewpt'].rolling(72).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 72 else 0, raw=False
        )

        enhanced_features['demand_velocity_72h'] = heat_demand.diff(72)
        enhanced_features['demand_acceleration_72h'] = heat_demand.diff(72).diff(72)
        enhanced_features['demand_jerk_72h'] = heat_demand.diff(72).diff(72).diff(72)

        enhanced_features['temp_hdd_72h'] = df['temp'].rolling(72).mean() * df['hdd_15_5'].rolling(72).mean()
        enhanced_features['weekend_seasonal'] = enhanced_features['is_weekend'] * enhanced_features['season_sin_1']
        enhanced_features['month_dow_interaction'] = enhanced_features['month_sin'] * enhanced_features['dow_sin']

        major_zones = ['V1', 'N1', 'B1_B2', 'W1']
        for zone in major_zones:
            if f'{zone}_expected_supply_temp' in df.columns:
                enhanced_features[f'{zone}_temp_trend_72h'] = df[f'{zone}_expected_supply_temp'].rolling(72).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 72 else 0, raw=False
                )
                enhanced_features[f'{zone}_temp_volatility_72h'] = df[f'{zone}_expected_supply_temp'].rolling(72).std()

        print(f"Added {len(enhanced_features)} comprehensive features")
        return enhanced_features

    def create_multi_horizon_targets(self, heat_demand):
        targets = {}
        for horizon in self.horizons:
            targets[f'demand_{horizon}h'] = heat_demand.shift(-horizon)
        return targets

    def prepare_ml_dataset(self, df, enhanced_features, targets):
        ml_df = df[self.base_features].copy()

        for feature_name, feature_series in enhanced_features.items():
            ml_df[feature_name] = feature_series

        for target_name, target_series in targets.items():
            ml_df[target_name] = target_series

        ml_df_clean = ml_df.dropna()
        feature_count = len(self.base_features) + len(enhanced_features)
        print(f"ML dataset prepared: {ml_df_clean.shape[0]} samples, {feature_count} features")

        return ml_df_clean

    def split_data(self, ml_df):
        train_end = '2023-12-31 23:00:00'
        val_end = '2024-12-31 23:00:00'

        train_mask = ml_df.index <= train_end
        val_mask = (ml_df.index > train_end) & (ml_df.index <= val_end)
        test_mask = ml_df.index > val_end

        feature_columns = [col for col in ml_df.columns if not col.startswith('demand_')]

        X_train = ml_df.loc[train_mask, feature_columns]
        X_val = ml_df.loc[val_mask, feature_columns]
        X_test = ml_df.loc[test_mask, feature_columns]

        y_train = {h: ml_df.loc[train_mask, f'demand_{h}h'] for h in self.horizons}
        y_val = {h: ml_df.loc[val_mask, f'demand_{h}h'] for h in self.horizons}
        y_test = {h: ml_df.loc[test_mask, f'demand_{h}h'] for h in self.horizons}

        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def advanced_feature_selection(self, X, y, horizon):
        var_threshold = VarianceThreshold(threshold=0.01)
        X_var = var_threshold.fit_transform(X)
        var_features = X.columns[var_threshold.get_support()]

        if horizon <= 6:
            k_features = min(45, len(var_features))
        elif horizon <= 24:
            k_features = min(65, len(var_features))
        elif horizon <= 48:
            k_features = min(80, len(var_features))
        else:
            k_features = min(120, len(var_features))

        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X_var, y)
        selected_indices = selector.get_support()
        selected_features = var_features[selected_indices].tolist()

        return X_selected, selected_features

    def get_optimized_models(self):
        return {
            1: LinearRegression(),
            6: Ridge(alpha=0.01),
            24: Ridge(alpha=0.001),
            48: ElasticNet(alpha=0.01, l1_ratio=0.1),
            72: VotingRegressor([
                ('ridge_light', Ridge(alpha=0.005)),
                ('ridge_medium', Ridge(alpha=0.02)),
                ('elastic_light', ElasticNet(alpha=0.01, l1_ratio=0.1)),
                ('elastic_heavy', ElasticNet(alpha=0.03, l1_ratio=0.2)),
                ('lasso', Lasso(alpha=0.01))
            ], weights=[0.25, 0.25, 0.2, 0.2, 0.1])
        }

    def train_models(self, X_train, y_train, X_val, y_val):
        model_strategy = self.get_optimized_models()
        results = {}

        for horizon in self.horizons:
            print(f"\nTraining {horizon}h model...")

            X_train_selected, selected_features = self.advanced_feature_selection(
                X_train, y_train[horizon], horizon
            )

            feature_indices = [i for i, feat in enumerate(X_train.columns) if feat in selected_features]
            X_val_selected = X_val.iloc[:, feature_indices]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_val_scaled = scaler.transform(X_val_selected)

            self.scalers[horizon] = scaler

            model = model_strategy[horizon]
            model.fit(X_train_scaled, y_train[horizon])

            self.models[horizon] = {
                'model': model,
                'selected_features': selected_features,
                'feature_indices': feature_indices
            }

            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)

            train_metrics = self.calculate_metrics(y_train[horizon], y_train_pred)
            val_metrics = self.calculate_metrics(y_val[horizon], y_val_pred)

            results[horizon] = {
                'train': train_metrics,
                'validation': val_metrics,
                'model_type': 'EnsembleVoting' if horizon == 72 else type(model).__name__,
                'n_features': len(selected_features),
                'regularization': 'EnsembleVoting' if horizon == 72 else getattr(model, 'alpha', 'None'),
            }

            print(f"{horizon}h - Features: {len(selected_features)}, Train R²: {train_metrics['r2']:.4f}, Val R²: {val_metrics['r2']:.4f}")

        return results

    def calculate_metrics(self, y_true, y_pred):
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

    def analyze_performance(self, results):
        print("\n" + "="*50)
        print("MULTI-HORIZON FORECASTING PERFORMANCE")
        print("="*50)

        for horizon in self.horizons:
            actual = results[horizon]['validation']['r2']
            print(f"{horizon:2d}h: {actual:.1%} R²")

        avg_performance = np.mean([results[h]['validation']['r2'] for h in self.horizons])
        print(f"\nAverage Performance: {avg_performance:.1%} R²")

        return avg_performance

    def create_comprehensive_plots(self, X_val, y_val, results):
        for plot_dir in self.plot_dirs:
            os.makedirs(plot_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        current = [results[h]['validation']['r2'] * 100 for h in self.horizons]
        horizons_str = [f'{h}h' for h in self.horizons]

        bars = axes[0,0].bar(horizons_str, current, color='steelblue', alpha=0.8)
        axes[0,0].set_ylabel('R² Score (%)')
        axes[0,0].set_title('Performance by Forecast Horizon')
        axes[0,0].grid(True, alpha=0.3)

        for bar, score in zip(bars, current):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{score:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')

        axes[0,1].plot(self.horizons, current, 'o-', color='darkblue', linewidth=3, markersize=8)
        axes[0,1].set_xlabel('Forecast Horizon (hours)')
        axes[0,1].set_ylabel('R² Score (%)')
        axes[0,1].set_title('Performance Degradation Curve')
        axes[0,1].grid(True, alpha=0.3)

        feature_counts = [results[h]['n_features'] for h in self.horizons]
        axes[0,2].bar(horizons_str, feature_counts, color='forestgreen', alpha=0.8)
        axes[0,2].set_ylabel('Number of Features')
        axes[0,2].set_title('Feature Count by Horizon')
        axes[0,2].grid(True, alpha=0.3)

        train_r2 = [results[h]['train']['r2'] * 100 for h in self.horizons]
        val_r2 = [results[h]['validation']['r2'] * 100 for h in self.horizons]

        x = np.arange(len(self.horizons))
        width = 0.35

        axes[1,0].bar(x - width/2, train_r2, width, label='Training', color='lightcoral', alpha=0.8)
        axes[1,0].bar(x + width/2, val_r2, width, label='Validation', color='steelblue', alpha=0.8)
        axes[1,0].set_ylabel('R² Score (%)')
        axes[1,0].set_title('Training vs Validation Performance')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(horizons_str)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        rmse_vals = [results[h]['validation']['rmse'] for h in self.horizons]
        mae_vals = [results[h]['validation']['mae'] for h in self.horizons]

        ax2 = axes[1,1]
        ax2_twin = ax2.twinx()

        bars1 = ax2.bar(x - width/2, rmse_vals, width, label='RMSE', color='orange', alpha=0.8)
        bars2 = ax2_twin.bar(x + width/2, mae_vals, width, label='MAE', color='purple', alpha=0.8)

        ax2.set_ylabel('RMSE', color='orange')
        ax2_twin.set_ylabel('MAE', color='purple')
        ax2.set_title('Error Metrics by Horizon')
        ax2.set_xticks(x)
        ax2.set_xticklabels(horizons_str)
        ax2.grid(True, alpha=0.3)

        if 24 in results:
            y_24h_true = y_val[24]
            y_24h_pred = self.models[24]['model'].predict(
                self.scalers[24].transform(
                    X_val.iloc[:, self.models[24]['feature_indices']]
                )
            )

            axes[1,2].scatter(y_24h_true, y_24h_pred, alpha=0.6, s=3, color='darkgreen')
            axes[1,2].plot([y_24h_true.min(), y_24h_true.max()],
                           [y_24h_true.min(), y_24h_true.max()], 'r--', lw=2)
            axes[1,2].set_xlabel('Actual Demand (MWh)')
            axes[1,2].set_ylabel('Predicted Demand (MWh)')
            axes[1,2].set_title(f'24h Prediction Quality (R² = {results[24]["validation"]["r2"]:.3f})')
            axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot('comprehensive_performance_analysis.png')
        plt.show()

        self.create_time_series_plots(X_val, y_val, results)

        self.create_feature_importance_plots(results)

        print(f"Comprehensive analysis plots saved to both directories")

    def create_time_series_plots(self, X_val, y_val, results):
        fig, axes = plt.subplots(len(self.horizons), 1, figsize=(15, 3*len(self.horizons)))

        for i, horizon in enumerate(self.horizons):
            if horizon in self.models:
                y_true = y_val[horizon]
                y_pred = self.models[horizon]['model'].predict(
                    self.scalers[horizon].transform(
                        X_val.iloc[:, self.models[horizon]['feature_indices']]
                    )
                )

                sample_size = min(168, len(y_true))
                time_idx = range(sample_size)

                axes[i].plot(time_idx, y_true.iloc[:sample_size], label='Actual', color='blue', linewidth=2)
                axes[i].plot(time_idx, y_pred[:sample_size], label='Predicted', color='red', linewidth=2, alpha=0.8)
                axes[i].set_title(f'{horizon}h Forecast - Sample Week (R² = {results[horizon]["validation"]["r2"]:.3f})')
                axes[i].set_ylabel('Demand (MWh)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

                if i == len(self.horizons) - 1:
                    axes[i].set_xlabel('Hours')

        plt.tight_layout()
        self.save_plot('time_series_forecasts.png')
        plt.show()

    def create_feature_importance_plots(self, results):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for i, horizon in enumerate(self.horizons):
            if horizon in results and horizon in self.models:
                model = self.models[horizon]['model']
                if hasattr(model, 'coef_'):
                    features = self.models[horizon]['selected_features']
                    importance = np.abs(model.coef_)

                    top_indices = np.argsort(importance)[-15:]
                    top_features = [features[idx] for idx in top_indices]
                    top_importance = importance[top_indices]

                    clean_names = [f.replace('_expected_', '_').replace('_supply_temp', '_ST').replace('_delta_T', '_DT')[:20] for f in top_features]

                    y_pos = np.arange(len(top_features))
                    axes[i].barh(y_pos, top_importance, color='skyblue', alpha=0.8)
                    axes[i].set_yticks(y_pos)
                    axes[i].set_yticklabels(clean_names, fontsize=8)
                    axes[i].set_xlabel('Absolute Coefficient Value')
                    axes[i].set_title(f'{horizon}h - Top Features')
                    axes[i].grid(True, alpha=0.3)

        if len(self.horizons) < 6:
            fig.delaxes(axes[-1])

        plt.tight_layout()
        self.save_plot('feature_importance_analysis.png')
        plt.show()

    def save_results(self, results):
        results_data = []
        for horizon in self.horizons:
            for split in ['train', 'validation']:
                metrics = results[horizon][split]
                results_data.append({
                    'methodology': 'Advanced MLR',
                    'horizon': f'{horizon}h',
                    'split': split,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'mape': metrics['mape'],
                    'model_type': results[horizon]['model_type'],
                    'n_features': results[horizon]['n_features'],
                    'regularization': results[horizon]['regularization']
                })

        results_df = pd.DataFrame(results_data)

        output_path = os.path.join(self.results_dir, 'advanced_mlr_results.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        return results_df

    def main_training(self, filename='merged_dataset.csv'):
        print("=== ADVANCED MULTI-HORIZON MLR FORECASTING ===")

        df = self.load_and_prepare_data(filename)
        heat_demand = self.create_enhanced_heat_demand(df)
        enhanced_features = self.add_comprehensive_features(df, heat_demand)
        targets = self.create_multi_horizon_targets(heat_demand)
        ml_df = self.prepare_ml_dataset(df, enhanced_features, targets)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(ml_df)
        results = self.train_models(X_train, y_train, X_val, y_val)
        avg_performance = self.analyze_performance(results)
        self.create_comprehensive_plots(X_val, y_val, results)
        self.save_results(results)

        return results, ml_df, avg_performance


def main():
    import time
    start_time = time.time()
    print("Starting Advanced MLR Implementation...")

    model = AdvancedMLRForecasting()
    results, ml_df, avg_performance = model.main_training()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    print(f"\nADVANCED MLR IMPLEMENTATION COMPLETE!")
    print(f"Average Performance: {avg_performance:.1%} R²")
    print(f"Total Training Time: {elapsed_time:.2f} minutes")

    return model, results



if __name__ == "__main__":
    model, results = main()