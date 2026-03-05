import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional, Any
import pickle
from pathlib import Path
import warnings

class DistrictHeatingEnv(gym.Env):

    def __init__(self,
                 data_file: str,
                 svr_model_path: str,
                 episode_length: int = 24,
                 start_date: str = '2024-01-01'):

        super(DistrictHeatingEnv, self).__init__()

        self.data = self._load_and_prepare_data(data_file)

        self.svr_model, self.svr_scaler = self._load_svr_model(svr_model_path)

        self.episode_length = episode_length
        self.current_step = 0
        self.episode_start_idx = 0

        self.zones = ['B1_B2', 'F1_Nord', 'F1_Sud', 'Maintal', 'N1', 'N2',
                      'V1', 'V2', 'V6', 'W1', 'ZN']
        self.n_zones = len(self.zones)

        self.feature_columns = self._identify_feature_columns()
        print(f"Using {len(self.feature_columns)} features for SVR predictions")

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_zones,),
            dtype=np.float32
        )

        obs_dim = 11 + 5 + 11 + 4

        self.observation_space = spaces.Box(
            low=-50.0,
            high=200.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.base_cost = 45.0
        self.peak_multiplier = 1.5
        self.efficiency_penalty = 10.0

        self.zone_temps = None
        self.rng = np.random.default_rng()

        print("District Heating Environment initialized successfully!")

    def _load_and_prepare_data(self, data_file: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(data_file)
            print(f"Dataset loaded: {data.shape}")

            datetime_cols = ['datetime', 'timestamp', 'date', 'time']
            datetime_col = None
            for col in datetime_cols:
                if col in data.columns:
                    datetime_col = col
                    break

            if datetime_col:
                data[datetime_col] = pd.to_datetime(data[datetime_col])
                data = data.set_index(datetime_col)
                print(f"Using datetime column: {datetime_col}")
            else:
                data.index = pd.date_range(start='2021-01-01', periods=len(data), freq='H')
                print("Created synthetic datetime index")

            data = data.fillna(method='ffill').fillna(method='bfill')

            print(f"Data prepared with index range: {data.index[0]} to {data.index[-1]}")
            return data

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _load_svr_model(self, svr_model_path: str) -> Tuple[Any, Optional[Any]]:
        try:
            with open(svr_model_path, 'rb') as f:
                model_data = pickle.load(f)

            if isinstance(model_data, dict):
                svr_model = model_data['model']
                svr_scaler = model_data.get('scaler', None)
                print(f"Loaded SVR model: {model_data.get('model_info', {})}")
            else:
                svr_model = model_data
                svr_scaler = None
                print("Loaded legacy SVR model format")

            print("SVR model loaded successfully")
            return svr_model, svr_scaler

        except Exception as e:
            print(f"Error loading SVR model: {e}")
            raise

    def _identify_feature_columns(self) -> List[str]:
        numeric_columns = [col for col in self.data.columns
                           if self.data[col].dtype in ['int64', 'float64', 'int32', 'float32']]

        priority_features = [
            'temp', 'app_temp', 'dewpt', 'rh', 'pres', 'slp', 'wind_spd',
            'wind_dir', 'solar_rad', 'ghi', 'dhi', 'dni', 'clouds', 'vis',

            'B1_B2_expected_supply_temp', 'F1_Nord_expected_supply_temp',
            'F1_Sud_expected_supply_temp', 'Maintal_expected_supply_temp',
            'N1_expected_supply_temp', 'N2_expected_supply_temp',
            'V1_expected_supply_temp', 'V2_expected_supply_temp',
            'V6_expected_supply_temp', 'W1_expected_supply_temp',
            'ZN_expected_supply_temp',

            'hdd_18', 'hdd_15_5', 'hour', 'day_of_week', 'month', 'season',

            'temp_change', 'is_daytime', 'is_weekend', 'precip', 'snow'
        ]

        selected_features = [f for f in priority_features if f in numeric_columns]

        remaining_features = [col for col in numeric_columns if col not in selected_features]
        selected_features.extend(remaining_features[:33-len(selected_features)])

        if len(selected_features) < 33:
            for i in range(33 - len(selected_features)):
                selected_features.append(f'synthetic_feature_{i}')

        return selected_features[:33]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        max_start = len(self.data) - self.episode_length - 24
        if max_start <= 0:
            max_start = len(self.data) - self.episode_length

        self.episode_start_idx = self.rng.integers(0, max(1, max_start))
        self.current_step = 0

        self.zone_temps = np.array([75.0] * self.n_zones, dtype=np.float32)

        obs = self._get_observation()

        info = {
            'episode_start_idx': int(self.episode_start_idx),
            'current_time': str(self.data.index[self.episode_start_idx])
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        current_idx = self.episode_start_idx + self.current_step

        action = np.clip(action, 0, 1)
        production = action * 100.0

        actual_demand = self._get_actual_demand(current_idx)

        reward = self._calculate_reward(production, actual_demand, current_idx)

        self._update_zone_temperatures(production, actual_demand)

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        obs = self._get_observation()

        cost = self._calculate_cost(production, current_idx)
        efficiency = min(np.sum(production), np.sum(actual_demand)) / max(np.sum(production), 1.0)

        info = {
            'step': int(self.current_step),
            'cost': float(cost),
            'efficiency': float(efficiency),
            'total_production': float(np.sum(production)),
            'total_demand': float(np.sum(actual_demand)),
            'zone_temps': self.zone_temps.copy(),
            'current_time': str(self.data.index[min(current_idx, len(self.data) - 1)])
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        current_idx = self.episode_start_idx + self.current_step

        weather_row = self.data.iloc[min(current_idx, len(self.data) - 1)]
        current_time = self.data.index[current_idx]

        zone_temps_norm = self.zone_temps / 100.0

        outdoor_temp = self._safe_get_value(weather_row, 'temp', 10.0)
        wind_speed = self._safe_get_value(weather_row, 'wind_spd', 0.0)
        solar_rad = self._safe_get_value(weather_row, 'solar_rad', 0.0)
        humidity = self._safe_get_value(weather_row, 'rh', 50.0)
        pressure = self._safe_get_value(weather_row, 'pres', 1013.0)

        weather_features = np.array([
            outdoor_temp / 30.0,
            wind_speed / 20.0,
            solar_rad / 1000.0,
            humidity / 100.0,
            pressure / 1100.0
        ], dtype=np.float32)

        forecasts = self._get_demand_forecasts(current_idx)

        time_features = np.array([
            current_time.hour / 24.0,
            current_time.dayofweek / 7.0,
            current_time.month / 12.0,
            (current_time.hour >= 6 and current_time.hour <= 22)
        ], dtype=np.float32)

        obs = np.concatenate([zone_temps_norm, weather_features, forecasts, time_features])

        return obs.astype(np.float32)

    def _safe_get_value(self, row: pd.Series, column: str, default: float) -> float:
        if column in row.index:
            try:
                value = float(row[column])
                return value if not np.isnan(value) else default
            except (ValueError, TypeError):
                return default
        return default

    def _get_demand_forecasts(self, current_idx: int) -> np.ndarray:
        try:
            forecast_features = self._prepare_forecast_features(current_idx)

            if self.svr_scaler is not None:
                forecast_features_scaled = self.svr_scaler.transform(forecast_features.reshape(1, -1))
            else:
                forecast_features_scaled = forecast_features.reshape(1, -1)

            predicted_demand = self.svr_model.predict(forecast_features_scaled)
            total_demand = max(predicted_demand[0], 1.0)

        except Exception as e:
            warnings.warn(f"SVR prediction failed: {e}. Using fallback method.")
            weather_row = self.data.iloc[min(current_idx, len(self.data) - 1)]
            outdoor_temp = self._safe_get_value(weather_row, 'temp', 10.0)
            total_demand = max(5.0, (15.5 - outdoor_temp) * 3.5)

        zone_weights = np.array([0.12, 0.08, 0.15, 0.09, 0.07, 0.06,
                                 0.11, 0.08, 0.05, 0.10, 0.09], dtype=np.float32)

        zone_forecasts = total_demand * zone_weights

        return zone_forecasts / 100.0

    def _prepare_forecast_features(self, current_idx: int) -> np.ndarray:
        row = self.data.iloc[min(current_idx, len(self.data) - 1)]
        current_time = self.data.index[current_idx]

        features = []

        for feature_col in self.feature_columns:
            if feature_col.startswith('synthetic_feature_'):
                features.append(current_time.hour / 24.0)
            elif feature_col in row.index:
                features.append(self._safe_get_value(row, feature_col, 0.0))
            else:
                if 'temp' in feature_col.lower():
                    features.append(10.0)
                elif 'hour' in feature_col.lower():
                    features.append(current_time.hour)
                elif 'day' in feature_col.lower():
                    features.append(current_time.dayofweek)
                elif 'month' in feature_col.lower():
                    features.append(current_time.month)
                else:
                    features.append(0.0)

        features = features[:33] + [0.0] * max(0, 33 - len(features))

        return np.array(features, dtype=np.float32)

    def _get_actual_demand(self, current_idx: int) -> np.ndarray:
        row = self.data.iloc[min(current_idx, len(self.data) - 1)]

        demand_cols = ['heat_demand', 'demand', 'load', 'consumption']
        total_demand = None

        for col in demand_cols:
            if col in row.index:
                total_demand = self._safe_get_value(row, col, None)
                if total_demand is not None:
                    break

        if total_demand is None:
            hdd = self._safe_get_value(row, 'hdd_15_5', None)
            if hdd is not None:
                total_demand = max(5.0, hdd * 3.5)
            else:
                outdoor_temp = self._safe_get_value(row, 'temp', 10.0)
                total_demand = max(5.0, (15.5 - outdoor_temp) * 3.5)

        zone_weights = np.array([0.12, 0.08, 0.15, 0.09, 0.07, 0.06,
                                 0.11, 0.08, 0.05, 0.10, 0.09], dtype=np.float32)

        return max(total_demand, 1.0) * zone_weights

    def _calculate_reward(self, production: np.ndarray, demand: np.ndarray,
                          current_idx: int) -> float:
        cost = self._calculate_cost(production, current_idx)
        cost_penalty = -cost / 1000.0

        total_prod = np.sum(production)
        total_demand = np.sum(demand)

        if total_prod > 0:
            efficiency = min(total_prod, total_demand) / total_prod
            efficiency_reward = efficiency * 10.0
        else:
            efficiency_reward = -10.0

        demand_satisfaction = 1.0 - abs(total_prod - total_demand) / max(total_demand, 1.0)
        demand_reward = demand_satisfaction * 5.0

        temp_penalty = 0
        for temp in self.zone_temps:
            if temp < 60 or temp > 120:
                temp_penalty -= abs(temp - 85) * 0.1

        reward = cost_penalty + efficiency_reward + demand_reward + temp_penalty

        return float(reward)

    def _calculate_cost(self, production: np.ndarray, current_idx: int) -> float:
        current_time = self.data.index[min(current_idx, len(self.data) - 1)]
        hour = current_time.hour

        if 6 <= hour <= 10 or 17 <= hour <= 21:
            cost_multiplier = self.peak_multiplier
        else:
            cost_multiplier = 1.0

        total_production = np.sum(production)
        cost = total_production * self.base_cost * cost_multiplier / 1000.0

        return float(cost)

    def _update_zone_temperatures(self, production: np.ndarray, demand: np.ndarray):
        for i in range(self.n_zones):
            heat_balance = production[i] - demand[i]

            temp_change = heat_balance * 0.5

            self.zone_temps[i] += temp_change * 0.3

            self.zone_temps[i] = np.clip(self.zone_temps[i], 40, 140)


if __name__ == "__main__":
    try:
        env = DistrictHeatingEnv(
            data_file="data/schweinfurt_data.csv",
            svr_model_path="data/svr_model_33features.pkl"
        )

        print("Environment created successfully!")
        print(f"Observation space shape: {env.observation_space.shape}")
        print(f"Action space shape: {env.action_space.shape}")

        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step completed successfully!")
        print(f"Reward: {reward:.2f}")
        print(f"Info: {info}")

    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()