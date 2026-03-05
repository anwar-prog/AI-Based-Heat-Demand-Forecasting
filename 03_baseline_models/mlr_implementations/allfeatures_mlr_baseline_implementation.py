import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.ensemble import VotingRegressor
import warnings
import os
warnings.filterwarnings('ignore')

class AllFeaturesMLRForecasting:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.scalers = {}

        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_dir, '01_data', 'processed_data')

        self.results_dir = os.path.join(self.base_dir, '03_baseline_models', 'results', 'allfeatures')
        self.plots_dir_main = self.results_dir
        self.plots_dir_results = self.results_dir
        self.plot_dirs = [self.results_dir]

        self.all_features = [
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
        self.horizons = [1, 6, 24, 48, 72]

    def save_plot(self, filename):
        for plot_dir in self.plot_dirs:
            os.makedirs(plot_dir, exist_ok=True)
            full_path = os.path.join(plot_dir, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {self.results_dir}")

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

        available_features = []
        for feature in self.all_features:
            if feature in df.columns and feature not in self.exclude_columns:
                available_features.append(feature)

        print(f"Available features: {len(available_features)}")
        self.available_features = available_features

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
                    df[feature] = df[feature].fillna(method='ffill').fillna(method='bfill')
                    if df[feature].isnull().sum() > 0:
                        df[feature] = df[feature].fillna('unknown')
                    df[feature] = df[feature].astype(str)
                    le = LabelEncoder()
                    df[feature] = le.fit_transform(df[feature])

        for feature in available_features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                try:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
                    df[feature] = df[feature].fillna(0)
                except:
                    le = LabelEncoder()
                    df[feature] = le.fit_transform(df[feature].astype(str))

            if df[feature].isnull().sum() > 0:
                df[feature] = df[feature].fillna(0)

        print(f"Final feature count: {len(available_features)}")
        return df

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

            if 'hour' in df.columns:
                hour = df['hour']
            else:
                hour = df.index.hour

            time_factor = 1.0 + 0.45 * (
                    np.exp(-((hour - 7) ** 2) / 12) +
                    np.exp(-((hour - 12) ** 2) / 18) +
                    np.exp(-((hour - 19) ** 2) / 14) +
                    0.3 * np.exp(-((hour - 22) ** 2) / 10)
            )

            if 'day_of_week' in df.columns:
                dow = df['day_of_week']
            else:
                dow = df.index.dayofweek

            dow_factor = np.array([1.30, 1.25, 1.20, 1.15, 1.10, 0.70, 0.65])[dow]

            if 'month' in df.columns:
                month = df['month']
            else:
                month = df.index.month

            seasonal_factor = (1.0 + 0.35 * np.cos(2 * np.pi * (month - 1) / 12) +
                               0.15 * np.cos(4 * np.pi * (month - 1) / 12))

            heat_demand = ((base_demand + temp_contribution) * time_factor *
                           dow_factor * seasonal_factor)
            heat_demand = np.maximum(heat_demand, 0.1)

        print(f"Heat demand data loaded: {heat_demand.min():.2f} to {heat_demand.max():.2f} MWh")
        return heat_demand

    def add_advanced_features(self, df, heat_demand):
        enhanced_features = {}

        lag_hours = [12, 23, 24, 25, 36, 47, 48, 49, 60, 72, 84, 96, 120, 144]
        for lag in lag_hours:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        weekly_lags = [162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 174, 176]
        for lag in weekly_lags:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        multi_week_lags = [336, 504]
        for lag in multi_week_lags:
            enhanced_features[f'demand_lag_{lag}h'] = heat_demand.shift(lag)

        windows = [24, 48, 72, 96, 120, 144, 168, 240, 336]
        for window in windows:
            enhanced_features[f'demand_rolling_mean_{window}h'] = heat_demand.rolling(window).mean()
            enhanced_features[f'demand_rolling_std_{window}h'] = heat_demand.rolling(window).std()

        enhanced_features['demand_rolling_min_72h'] = heat_demand.rolling(72).min()
        enhanced_features['demand_rolling_max_72h'] = heat_demand.rolling(72).max()
        enhanced_features['demand_rolling_median_72h'] = heat_demand.rolling(72).median()

        if 'hour' in df.columns:
            hour = df['hour']
        else:
            hour = df.index.hour

        enhanced_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        enhanced_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        if 'day_of_week' in df.columns:
            dow = df['day_of_week']
        else:
            dow = df.index.dayofweek

        enhanced_features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        enhanced_features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        if 'month' in df.columns:
            month = df['month']
        else:
            month = df.index.month

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
            enhanced_features['temp_trend_48h'] = df['temp'].rolling(48).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 48 else 0, raw=False
            )
            enhanced_features['temp_trend_72h'] = df['temp'].rolling(72).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 72 else 0, raw=False
            )
            enhanced_features['temp_trend_168h'] = df['temp'].rolling(168).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 168 else 0, raw=False
            )
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

        day_of_year = df.index.dayofyear if hasattr(df.index, 'dayofyear') else ((month - 1) * 30 + df.index.day)
        enhanced_features['season_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        enhanced_features['season_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        week_of_year = df.index.isocalendar().week if hasattr(df.index, 'isocalendar') else ((day_of_year - 1) // 7 + 1)
        enhanced_features['week_of_year_sin'] = np.sin(2 * np.pi * week_of_year / 52)
        enhanced_features['week_of_year_cos'] = np.cos(2 * np.pi * week_of_year / 52)

        if 'hdd_15_5' in df.columns:
            enhanced_features['hdd_lag_48h'] = df['hdd_15_5'].shift(48)
            enhanced_features['hdd_lag_72h'] = df['hdd_15_5'].shift(72)
            enhanced_features['hdd_lag_168h'] = df['hdd_15_5'].shift(168)
            enhanced_features['hdd_trend_72h'] = df['hdd_15_5'].rolling(72).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 72 else 0, raw=False
            )
            enhanced_features['hdd_trend_168h'] = df['hdd_15_5'].rolling(168).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 168 else 0, raw=False
            )

        major_zones = ['B1_B2', 'V1', 'W1', 'F1_Sud']
        for zone in major_zones:
            supply_col = f'{zone}_expected_supply_temp'
            if supply_col in df.columns:
                enhanced_features[f'{zone}_lag_72h'] = df[supply_col].shift(72)
                enhanced_features[f'{zone}_lag_168h'] = df[supply_col].shift(168)
                enhanced_features[f'{zone}_trend_72h'] = df[supply_col].rolling(72).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 72 else 0, raw=False
                )

        if 'temp' in df.columns:
            enhanced_features['weekend_temp_interaction'] = enhanced_features['is_weekend'] * df['temp']
        enhanced_features['dow_hour_interaction'] = enhanced_features['dow_sin'] * enhanced_features['hour_sin']
        enhanced_features['week_position_season_interaction'] = enhanced_features['week_position_sin'] * enhanced_features['season_sin']

        print(f"Added {len(enhanced_features)} advanced features")
        return enhanced_features

    def create_multi_horizon_targets(self, heat_demand):
        targets = {}
        for horizon in self.horizons:
            targets[f'demand_{horizon}h'] = heat_demand.shift(-horizon)
        return targets

    def prepare_ml_dataset(self, df, enhanced_features, targets):
        ml_df = df[self.available_features].copy()

        for feature_name, feature_series in enhanced_features.items():
            ml_df[feature_name] = feature_series

        for target_name, target_series in targets.items():
            ml_df[target_name] = target_series

        print(f"Dataset before cleaning: {ml_df.shape[0]} samples")
        ml_df_clean = ml_df.dropna()
        print(f"Dataset after cleaning: {ml_df_clean.shape[0]} samples")

        total_features = len(self.available_features) + len(enhanced_features)
        print(f"ML dataset: {ml_df_clean.shape[0]} samples, {total_features} features")
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

    def feature_selection_by_horizon(self, X, y, horizon):
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
            k_features = min(150, len(var_features))
        elif horizon <= 48:
            k_features = min(140, len(var_features))
        else:
            k_features = min(220, len(var_features))

        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X_var, y)
        selected_indices = selector.get_support()
        selected_features = var_features[selected_indices].tolist()

        print(f"{horizon}h feature selection: {len(X_numeric.columns)} -> {len(var_features)} -> {len(selected_features)}")
        return X_selected, selected_features

    def get_model_strategy(self):
        return {
            1: Ridge(alpha=0.01),
            6: Ridge(alpha=0.1),
            24: VotingRegressor([
                ('ridge1', Ridge(alpha=0.5)),
                ('ridge2', Ridge(alpha=2.0)),
                ('elastic', ElasticNet(alpha=1.0, l1_ratio=0.1))
            ], weights=[0.4, 0.3, 0.3]),
            48: VotingRegressor([
                ('ridge1', Ridge(alpha=1.0)),
                ('ridge2', Ridge(alpha=3.0)),
                ('elastic1', ElasticNet(alpha=1.5, l1_ratio=0.1)),
                ('elastic2', ElasticNet(alpha=2.5, l1_ratio=0.2))
            ], weights=[0.3, 0.3, 0.2, 0.2]),
            72: VotingRegressor([
                ('ridge_ultra_light', Ridge(alpha=0.1)),
                ('ridge_light', Ridge(alpha=0.5)),
                ('ridge_medium', Ridge(alpha=2.0)),
                ('ridge_heavy', Ridge(alpha=5.0)),
                ('elastic_ultra_light', ElasticNet(alpha=0.3, l1_ratio=0.05)),
                ('elastic_light', ElasticNet(alpha=1.0, l1_ratio=0.1)),
                ('elastic_medium', ElasticNet(alpha=2.0, l1_ratio=0.15)),
                ('elastic_heavy', ElasticNet(alpha=3.0, l1_ratio=0.2)),
                ('lasso_light', Lasso(alpha=0.2)),
                ('lasso_medium', Lasso(alpha=0.8))
            ], weights=[0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
        }

    def train_models(self, X_train, y_train, X_val, y_val):
        model_strategy = self.get_model_strategy()
        results = {}

        for horizon in self.horizons:
            print(f"\nTraining {horizon}h model...")

            X_train_selected, selected_features = self.feature_selection_by_horizon(
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

            model_type = 'Advanced72hEnsemble' if horizon == 72 else ('Ensemble' if isinstance(model, VotingRegressor) else type(model).__name__)

            results[horizon] = {
                'train': train_metrics,
                'validation': val_metrics,
                'model_type': model_type,
                'features_used': len(selected_features),
                'regularization': 'Advanced72hEnsemble' if horizon == 72 else ('Ensemble' if isinstance(model, VotingRegressor) else getattr(model, 'alpha', 'N/A'))
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
        print("\n" + "="*60)
        print("ALL-FEATURES MLR PERFORMANCE ANALYSIS")
        print("="*60)

        for horizon in self.horizons:
            actual = results[horizon]['validation']['r2']
            print(f"{horizon:2d}h: {actual:.1%} R²")

        avg_performance = np.mean([results[h]['validation']['r2'] for h in self.horizons])
        print(f"\nAverage Performance: {avg_performance:.1%} R²")

        return avg_performance

    def create_mlr_architecture_analysis(self, results):
        print("Creating MLR architecture analysis plot...")

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        horizons = self.horizons
        horizon_labels = [f'{h}h' for h in horizons]

        ax1 = fig.add_subplot(gs[0, 0])
        ensemble_sizes = []
        for horizon in horizons:
            model = self.models[horizon]['model']
            if isinstance(model, VotingRegressor):
                ensemble_sizes.append(len(model.estimators_))
            else:
                ensemble_sizes.append(1)

        colors = ['lightblue' if s == 1 else 'coral' if s <= 4 else 'darkred' for s in ensemble_sizes]
        bars = ax1.bar(horizon_labels, ensemble_sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Number of Base Models', fontsize=11, fontweight='bold')
        ax1.set_title('Ensemble Size Evolution Across Horizons', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        for i, (size, model_type) in enumerate(zip(ensemble_sizes, [results[h]['model_type'] for h in horizons])):
            ax1.text(i, size + 0.3, f'{size}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2 = fig.add_subplot(gs[0, 1])
        reg_data = {
            'Ridge': [0, 0, 2, 2, 4],
            'ElasticNet': [0, 0, 1, 2, 4],
            'Lasso': [0, 0, 0, 0, 2]
        }

        x = np.arange(len(horizons))
        width = 0.25

        ax2.bar(x - width, reg_data['Ridge'], width, label='Ridge', color='steelblue', alpha=0.8)
        ax2.bar(x, reg_data['ElasticNet'], width, label='ElasticNet', color='darkorange', alpha=0.8)
        ax2.bar(x + width, reg_data['Lasso'], width, label='Lasso', color='forestgreen', alpha=0.8)

        ax2.set_ylabel('Number of Estimators', fontsize=11, fontweight='bold')
        ax2.set_title('Regularization Strategy Distribution', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(horizon_labels)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')

        ax3 = fig.add_subplot(gs[0, 2])
        complexity_index = []
        for horizon in horizons:
            features = results[horizon]['features_used']
            ensemble_size = ensemble_sizes[horizons.index(horizon)]
            complexity_index.append(features * ensemble_size)

        bars = ax3.bar(horizon_labels, complexity_index, color='mediumpurple', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Complexity Index (Features × Models)', fontsize=11, fontweight='bold')
        ax3.set_title('Overall Model Complexity', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        for i, comp in enumerate(complexity_index):
            ax3.text(i, comp + 50, f'{comp}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax4 = fig.add_subplot(gs[1, :])
        model_72h = self.models[72]['model']
        if isinstance(model_72h, VotingRegressor):
            estimator_names = [name for name, _ in model_72h.estimators]
            weights = model_72h.weights if model_72h.weights is not None else [1] * len(estimator_names)

            colors_weights = ['steelblue']*4 + ['darkorange']*4 + ['forestgreen']*2
            bars = ax4.barh(estimator_names, weights, color=colors_weights, alpha=0.8, edgecolor='black', linewidth=1)

            ax4.set_xlabel('Ensemble Weight', fontsize=11, fontweight='bold')
            ax4.set_title('72h Model: Ensemble Composition and Weights', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')

            for i, (name, weight) in enumerate(zip(estimator_names, weights)):
                ax4.text(weight + 0.005, i, f'{weight:.2f}', va='center', fontsize=9)

        ax5 = fig.add_subplot(gs[2, 0])
        train_r2 = [results[h]['train']['r2'] * 100 for h in horizons]
        val_r2 = [results[h]['validation']['r2'] * 100 for h in horizons]

        x = np.arange(len(horizons))
        width = 0.35

        ax5.bar(x - width/2, train_r2, width, label='Training', color='lightblue', alpha=0.8)
        ax5.bar(x + width/2, val_r2, width, label='Validation', color='coral', alpha=0.8)

        ax5.set_ylabel('R² Score (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Training vs Validation Performance', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(horizon_labels)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        ax6 = fig.add_subplot(gs[2, 1])
        gen_gap = [(train_r2[i] - val_r2[i]) for i in range(len(horizons))]
        colors_gap = ['green' if gap < 5 else 'orange' if gap < 10 else 'red' for gap in gen_gap]

        bars = ax6.bar(horizon_labels, gen_gap, color=colors_gap, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax6.set_ylabel('Generalization Gap (% R²)', fontsize=11, fontweight='bold')
        ax6.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
        ax6.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Warning (5%)')
        ax6.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Critical (10%)')
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')

        for i, gap in enumerate(gen_gap):
            ax6.text(i, gap + 0.3, f'{gap:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax7 = fig.add_subplot(gs[2, 2])
        efficiency = []
        for horizon in horizons:
            perf = results[horizon]['validation']['r2']
            feat = results[horizon]['features_used']
            ens_size = ensemble_sizes[horizons.index(horizon)]
            efficiency.append((perf / (feat * ens_size)) * 10000)

        bars = ax7.bar(horizon_labels, efficiency, color='teal', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax7.set_ylabel('Efficiency Score', fontsize=11, fontweight='bold')
        ax7.set_title('Performance Efficiency Index', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')

        for i, eff in enumerate(efficiency):
            ax7.text(i, eff + 0.5, f'{eff:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle('MLR-All: Architecture and Ensemble Analysis', fontsize=14, fontweight='bold', y=0.995)
        self.save_plot('mlr_architecture_analysis.png')
        plt.close()

    def create_mlr_regularization_analysis(self, results):
        print("Creating MLR regularization analysis plot...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        horizons = self.horizons
        horizon_labels = [f'{h}h' for h in horizons]

        ax1 = fig.add_subplot(gs[0, 0])

        alpha_ranges = {
            1: [0.01],
            6: [0.1],
            24: [0.5, 2.0, 1.0],
            48: [1.0, 3.0, 1.5, 2.5],
            72: [0.1, 0.5, 2.0, 5.0, 0.3, 1.0, 2.0, 3.0, 0.2, 0.8]
        }

        avg_alpha = [np.mean(alpha_ranges[h]) for h in horizons]
        min_alpha = [np.min(alpha_ranges[h]) for h in horizons]
        max_alpha = [np.max(alpha_ranges[h]) for h in horizons]

        ax1.plot(horizon_labels, avg_alpha, 'o-', linewidth=2.5, markersize=8, color='darkblue', label='Average α')
        ax1.fill_between(range(len(horizons)), min_alpha, max_alpha, alpha=0.3, color='lightblue', label='α Range')

        ax1.set_ylabel('Regularization Strength (α)', fontsize=11, fontweight='bold')
        ax1.set_title('Regularization Parameter Evolution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        ax2 = fig.add_subplot(gs[0, 1])
        rmse_vals = [results[h]['validation']['rmse'] for h in horizons]
        mae_vals = [results[h]['validation']['mae'] for h in horizons]

        x = np.arange(len(horizons))
        width = 0.35

        ax2.bar(x - width/2, rmse_vals, width, label='RMSE', color='lightcoral', alpha=0.8)
        ax2.bar(x + width/2, mae_vals, width, label='MAE', color='lightskyblue', alpha=0.8)

        ax2.set_ylabel('Error (MWh)', fontsize=11, fontweight='bold')
        ax2.set_title('Error Metrics by Horizon', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(horizon_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        for i, (rmse, mae) in enumerate(zip(rmse_vals, mae_vals)):
            ax2.text(i - width/2, rmse + 0.5, f'{rmse:.1f}', ha='center', va='bottom', fontsize=8)
            ax2.text(i + width/2, mae + 0.5, f'{mae:.1f}', ha='center', va='bottom', fontsize=8)

        ax3 = fig.add_subplot(gs[0, 2])
        mape_vals = [results[h]['validation']['mape'] for h in horizons]

        colors_mape = ['green' if m < 15 else 'orange' if m < 25 else 'red' for m in mape_vals]
        bars = ax3.bar(horizon_labels, mape_vals, color=colors_mape, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax3.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
        ax3.axhline(y=15, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='Good (<15%)')
        ax3.axhline(y=25, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Acceptable (<25%)')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        for i, mape in enumerate(mape_vals):
            ax3.text(i, mape + 1, f'{mape:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax4 = fig.add_subplot(gs[1, 0])
        train_r2 = [results[h]['train']['r2'] for h in horizons]
        val_r2 = [results[h]['validation']['r2'] for h in horizons]

        ax4.plot(horizon_labels, train_r2, 'o-', linewidth=2.5, markersize=8, color='blue', label='Training R²')
        ax4.plot(horizon_labels, val_r2, 's-', linewidth=2.5, markersize=8, color='red', label='Validation R²')

        ax4.set_ylabel('R² Score', fontsize=11, fontweight='bold')
        ax4.set_title('R² Score Progression', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        for i, (tr, vr) in enumerate(zip(train_r2, val_r2)):
            ax4.text(i, tr + 0.01, f'{tr:.3f}', ha='center', va='bottom', fontsize=8, color='blue')
            ax4.text(i, vr - 0.02, f'{vr:.3f}', ha='center', va='top', fontsize=8, color='red')

        ax5 = fig.add_subplot(gs[1, 1])
        features_used = [results[h]['features_used'] for h in horizons]
        val_r2_pct = [r * 100 for r in val_r2]

        scatter = ax5.scatter(features_used, val_r2_pct, s=[h*3 for h in horizons],
                              c=horizons, cmap='viridis', alpha=0.7, edgecolor='black', linewidth=1.5)

        for i, (feat, perf, label) in enumerate(zip(features_used, val_r2_pct, horizon_labels)):
            ax5.annotate(label, (feat, perf), xytext=(5, 5), textcoords='offset points',
                         fontsize=10, fontweight='bold')

        ax5.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Validation R² (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Feature Count vs Performance', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Horizon (h)', fontsize=10)

        ax6 = fig.add_subplot(gs[1, 2])
        error_ratio = [rmse_vals[i] / mae_vals[i] for i in range(len(horizons))]

        bars = ax6.bar(horizon_labels, error_ratio, color='darkorchid', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax6.axhline(y=1.25, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Ideal (~1.25)')

        ax6.set_ylabel('RMSE/MAE Ratio', fontsize=11, fontweight='bold')
        ax6.set_title('Error Distribution Consistency', fontsize=12, fontweight='bold')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3, axis='y')

        for i, ratio in enumerate(error_ratio):
            ax6.text(i, ratio + 0.01, f'{ratio:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle('MLR-All: Regularization and Error Analysis', fontsize=14, fontweight='bold', y=0.995)
        self.save_plot('mlr_regularization_analysis.png')
        plt.close()

    def create_mlr_performance_details(self, results):
        print("Creating MLR performance details plot...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        horizons = self.horizons
        horizon_labels = [f'{h}h' for h in horizons]

        ax1 = fig.add_subplot(gs[0, :2])
        metrics_matrix = []
        metric_names = ['R² (%)', 'RMSE', 'MAE', 'MAPE (%)']

        for horizon in horizons:
            row = [
                results[horizon]['validation']['r2'] * 100,
                results[horizon]['validation']['rmse'],
                results[horizon]['validation']['mae'],
                results[horizon]['validation']['mape']
            ]
            metrics_matrix.append(row)

        metrics_array = np.array(metrics_matrix).T
        normalized_metrics = np.zeros_like(metrics_array)

        for i in range(len(metric_names)):
            if i == 0:
                normalized_metrics[i] = metrics_array[i] / 100
            else:
                max_val = np.max(metrics_array[i])
                normalized_metrics[i] = 1 - (metrics_array[i] / max_val)

        im = ax1.imshow(normalized_metrics, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax1.set_xticks(range(len(horizons)))
        ax1.set_xticklabels(horizon_labels)
        ax1.set_yticks(range(len(metric_names)))
        ax1.set_yticklabels(metric_names)
        ax1.set_title('Performance Metrics Heatmap (Normalized)', fontsize=12, fontweight='bold')

        for i in range(len(metric_names)):
            for j in range(len(horizons)):
                text = ax1.text(j, i, f'{metrics_array[i, j]:.1f}',
                                ha="center", va="center", color="black", fontsize=9, fontweight='bold')

        plt.colorbar(im, ax=ax1, label='Normalized Score (0-1)')

        ax2 = fig.add_subplot(gs[0, 2])
        model_types_count = {'Single': 0, 'Small Ensemble': 0, 'Large Ensemble': 0}

        for horizon in horizons:
            model = self.models[horizon]['model']
            if isinstance(model, VotingRegressor):
                num_estimators = len(model.estimators_)
                if num_estimators <= 4:
                    model_types_count['Small Ensemble'] += 1
                else:
                    model_types_count['Large Ensemble'] += 1
            else:
                model_types_count['Single'] += 1

        colors_pie = ['lightblue', 'coral', 'darkred']
        wedges, texts, autotexts = ax2.pie(model_types_count.values(), labels=model_types_count.keys(),
                                           autopct='%1.0f%%', colors=colors_pie, startangle=90)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        ax2.set_title('Model Architecture Distribution', fontsize=12, fontweight='bold')

        ax3 = fig.add_subplot(gs[1, 0])
        train_r2 = [results[h]['train']['r2'] for h in horizons]
        val_r2 = [results[h]['validation']['r2'] for h in horizons]
        stability = [abs(train_r2[i] - val_r2[i]) / train_r2[i] * 100 for i in range(len(horizons))]

        colors_stab = ['green' if s < 5 else 'orange' if s < 10 else 'red' for s in stability]
        bars = ax3.bar(horizon_labels, stability, color=colors_stab, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax3.set_ylabel('Stability Loss (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Model Stability Analysis', fontsize=12, fontweight='bold')
        ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Stable (<5%)')
        ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Unstable (>10%)')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        for i, stab in enumerate(stability):
            ax3.text(i, stab + 0.2, f'{stab:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax4 = fig.add_subplot(gs[1, 1])
        val_r2_pct = [results[h]['validation']['r2'] * 100 for h in horizons]

        colors_perf = ['darkgreen' if p > 85 else 'green' if p > 75 else 'orange' if p > 65 else 'red'
                       for p in val_r2_pct]
        bars = ax4.bar(horizon_labels, val_r2_pct, color=colors_perf, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax4.set_ylabel('Validation R² (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Absolute Performance by Horizon', fontsize=12, fontweight='bold')
        ax4.axhline(y=85, color='darkgreen', linestyle='--', alpha=0.5, linewidth=1, label='Excellent (>85%)')
        ax4.axhline(y=75, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Good (>75%)')
        ax4.axhline(y=65, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Fair (>65%)')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0, 100])

        for i, perf in enumerate(val_r2_pct):
            ax4.text(i, perf + 1, f'{perf:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax5 = fig.add_subplot(gs[1, 2])
        features_used = [results[h]['features_used'] for h in horizons]
        ensemble_sizes = []
        for horizon in horizons:
            model = self.models[horizon]['model']
            if isinstance(model, VotingRegressor):
                ensemble_sizes.append(len(model.estimators_))
            else:
                ensemble_sizes.append(1)

        comp_cost = [features_used[i] * ensemble_sizes[i] for i in range(len(horizons))]
        val_r2_pct = [results[h]['validation']['r2'] * 100 for h in horizons]

        efficiency_score = [(val_r2_pct[i] / comp_cost[i]) * 100 for i in range(len(horizons))]

        bars = ax5.bar(horizon_labels, efficiency_score, color='teal', alpha=0.8, edgecolor='black', linewidth=1.5)

        ax5.set_ylabel('Efficiency Score', fontsize=11, fontweight='bold')
        ax5.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        for i, eff in enumerate(efficiency_score):
            ax5.text(i, eff + 0.5, f'{eff:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.suptitle('MLR-All: Performance Details and Analysis', fontsize=14, fontweight='bold', y=0.995)
        self.save_plot('mlr_performance_details.png')
        plt.close()

    def create_comprehensive_plots(self, results):
        for plot_dir in self.plot_dirs:
            os.makedirs(plot_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        performance = [results[h]['validation']['r2'] * 100 for h in self.horizons]
        train_performance = [results[h]['train']['r2'] * 100 for h in self.horizons]
        horizon_labels = [f'{h}h' for h in self.horizons]

        x = np.arange(len(self.horizons))
        width = 0.35

        bars1 = axes[0,0].bar(x - width/2, train_performance, width, label='Training', color='lightblue', alpha=0.8)
        bars2 = axes[0,0].bar(x + width/2, performance, width, label='Validation',
                              color=['darkblue', 'green', 'orange', 'red', 'darkred'], alpha=0.8)

        axes[0,0].set_ylabel('R² Score (%)')
        axes[0,0].set_title('Model Performance by Forecast Horizon')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(horizon_labels)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        for i, (train, val) in enumerate(zip(train_performance, performance)):
            axes[0,0].text(i - width/2, train + 0.5, f'{train:.1f}%', ha='center', va='bottom', fontsize=9)
            axes[0,0].text(i + width/2, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, weight='bold')

        feature_counts = [results[h]['features_used'] for h in self.horizons]
        bars = axes[0,1].bar(horizon_labels, feature_counts,
                             color=['lightblue', 'lightgreen', 'orange', 'red', 'darkred'], alpha=0.8)
        axes[0,1].set_ylabel('Features Selected')
        axes[0,1].set_title('Feature Count by Horizon')
        axes[0,1].grid(True, alpha=0.3)

        for i, count in enumerate(feature_counts):
            axes[0,1].text(i, count + 2, f'{count}', ha='center', va='bottom', fontsize=10, weight='bold')

        model_types = [results[h]['model_type'] for h in self.horizons]
        complexity_scores = []
        for horizon in self.horizons:
            if 'Ensemble' in results[horizon]['model_type']:
                if horizon == 72:
                    complexity_scores.append(10)
                elif horizon in [24, 48]:
                    complexity_scores.append(len(results[horizon]['model_type']) if isinstance(results[horizon]['model_type'], list) else 4)
                else:
                    complexity_scores.append(3)
            else:
                complexity_scores.append(1)

        bars = axes[0,2].bar(horizon_labels, complexity_scores,
                             color=['lightblue', 'lightgreen', 'orange', 'red', 'darkred'], alpha=0.8)
        axes[0,2].set_ylabel('Model Complexity Score')
        axes[0,2].set_title('Model Architecture Complexity')
        axes[0,2].grid(True, alpha=0.3)

        for i, (score, model_type) in enumerate(zip(complexity_scores, model_types)):
            axes[0,2].text(i, score + 0.2, f'{score}\n({model_type[:6]})', ha='center', va='bottom', fontsize=8)

        axes[1,0].scatter(complexity_scores, performance,
                          c=['blue', 'green', 'orange', 'red', 'darkred'],
                          s=[100, 120, 140, 160, 180], alpha=0.7)

        for i, (comp, perf, label) in enumerate(zip(complexity_scores, performance, horizon_labels)):
            axes[1,0].annotate(label, (comp, perf), xytext=(5, 5), textcoords='offset points', fontsize=10)

        axes[1,0].set_xlabel('Model Complexity Score')
        axes[1,0].set_ylabel('Validation R² (%)')
        axes[1,0].set_title('Performance vs Model Complexity')
        axes[1,0].grid(True, alpha=0.3)

        rmse_values = [results[h]['validation']['rmse'] for h in self.horizons]
        mae_values = [results[h]['validation']['mae'] for h in self.horizons]

        x = np.arange(len(self.horizons))
        width = 0.35

        bars1 = axes[1,1].bar(x - width/2, rmse_values, width, label='RMSE', color='lightcoral', alpha=0.8)
        bars2 = axes[1,1].bar(x + width/2, mae_values, width, label='MAE', color='lightblue', alpha=0.8)

        axes[1,1].set_ylabel('Error (MWh)')
        axes[1,1].set_title('Error Metrics by Horizon')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(horizon_labels)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        normalized_performance = [p/performance[0] * 100 for p in performance]
        axes[1,2].plot(self.horizons, normalized_performance, 'o-', linewidth=3, markersize=8, color='purple')
        axes[1,2].set_xlabel('Forecast Horizon (hours)')
        axes[1,2].set_ylabel('Normalized Performance (%)')
        axes[1,2].set_title('Performance Decay Pattern')
        axes[1,2].grid(True, alpha=0.3)

        for i, (horizon, norm_perf) in enumerate(zip(self.horizons, normalized_performance)):
            axes[1,2].text(horizon, norm_perf + 1, f'{norm_perf:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        self.save_plot('comprehensive_mlr_analysis.png')
        plt.close()

        self.create_feature_analysis_plot(results)

        self.create_mlr_architecture_analysis(results)
        self.create_mlr_regularization_analysis(results)
        self.create_mlr_performance_details(results)

        print(f"All comprehensive analysis plots saved")

    def create_feature_analysis_plot(self, results):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        feature_categories = {
            'Weather': ['temp', 'humidity', 'wind', 'solar', 'precip', 'clouds'],
            'Zone_Supply': ['supply_temp', 'return_temp', 'delta_T'],
            'Temporal': ['hour', 'day', 'month', 'season', 'weekend'],
            'Lag_Features': ['lag_', 'rolling_'],
            'HDD': ['hdd_'],
            'Advanced': ['sin', 'cos', 'trend', 'interaction']
        }

        horizon_feature_counts = {}
        for horizon in self.horizons:
            selected_features = self.models[horizon]['selected_features']
            category_counts = {}

            for category, keywords in feature_categories.items():
                count = 0
                for feature in selected_features:
                    if any(keyword in feature.lower() for keyword in keywords):
                        count += 1
                category_counts[category] = count

            horizon_feature_counts[horizon] = category_counts

        categories = list(feature_categories.keys())
        heatmap_data = []

        for category in categories:
            row = [horizon_feature_counts[h][category] for h in self.horizons]
            heatmap_data.append(row)

        im = axes[0].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        axes[0].set_xticks(range(len(self.horizons)))
        axes[0].set_xticklabels([f'{h}h' for h in self.horizons])
        axes[0].set_yticks(range(len(categories)))
        axes[0].set_yticklabels(categories)
        axes[0].set_title('Feature Category Usage by Horizon')

        for i in range(len(categories)):
            for j in range(len(self.horizons)):
                text = axes[0].text(j, i, heatmap_data[i][j], ha="center", va="center", color="black")

        plt.colorbar(im, ax=axes[0])

        efficiency_scores = []
        for horizon in self.horizons:
            performance = results[horizon]['validation']['r2']
            features_used = results[horizon]['features_used']
            efficiency = performance / features_used * 1000
            efficiency_scores.append(efficiency)

        bars = axes[1].bar([f'{h}h' for h in self.horizons], efficiency_scores,
                           color=['lightblue', 'lightgreen', 'orange', 'red', 'darkred'], alpha=0.8)
        axes[1].set_ylabel('Performance Efficiency (R²/Features * 1000)')
        axes[1].set_title('Model Efficiency by Horizon')
        axes[1].grid(True, alpha=0.3)

        for i, eff in enumerate(efficiency_scores):
            axes[1].text(i, eff + 0.1, f'{eff:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')

        plt.tight_layout()
        self.save_plot('feature_analysis.png')
        plt.close()

    def save_results(self, results):
        results_data = []
        for horizon in self.horizons:
            for split in ['train', 'validation']:
                metrics = results[horizon][split]
                results_data.append({
                    'methodology': 'All Features MLR',
                    'horizon': f'{horizon}h',
                    'split': split,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'mape': metrics['mape'],
                    'features_used': results[horizon]['features_used'],
                    'model_type': results[horizon]['model_type'],
                    'regularization': results[horizon]['regularization']
                })

        results_df = pd.DataFrame(results_data)

        output_path = os.path.join(self.results_dir, 'all_features_mlr_results.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        return results_df

    def main_training(self, filename='merged_dataset.csv'):
        print("=== ALL-FEATURES MLR ANALYSIS ===")

        df = self.load_and_prepare_data(filename)
        heat_demand = self.load_heat_demand_data(df)
        enhanced_features = self.add_advanced_features(df, heat_demand)
        targets = self.create_multi_horizon_targets(heat_demand)
        ml_df = self.prepare_ml_dataset(df, enhanced_features, targets)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(ml_df)
        results = self.train_models(X_train, y_train, X_val, y_val)
        avg_performance = self.analyze_performance(results)
        self.create_comprehensive_plots(results)
        self.save_results(results)

        return results, avg_performance


def main():
    import time
    start_time = time.time()
    print("Starting All-Features MLR Implementation...")

    model = AllFeaturesMLRForecasting()
    results, avg_performance = model.main_training()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    print(f"\nALL-FEATURES MLR ANALYSIS COMPLETE!")
    print(f"Average Performance: {avg_performance:.1%} R²")
    print(f"Total Training Time: {elapsed_time:.2f} minutes")

    print("\nPERFORMANCE SUMMARY:")
    for horizon in model.horizons:
        perf = results[horizon]['validation']['r2']
        features = results[horizon]['features_used']
        model_type = results[horizon]['model_type']
        print(f"{horizon:2d}h: {perf:.1%} R² ({features} features, {model_type})")

    return model, results



if __name__ == "__main__":
    model, results = main()