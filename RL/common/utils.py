import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
import os

SCHWEINFURT_ZONES = ['B1_B2', 'F1_Nord', 'F1_Sud', 'Maintal', 'N1', 'N2',
                     'V1', 'V2', 'V6', 'W1', 'ZN']

ZONE_WEIGHTS = np.array([0.12, 0.08, 0.15, 0.09, 0.07, 0.06,
                         0.11, 0.08, 0.05, 0.10, 0.09], dtype=np.float32)

DEFAULT_FEATURE_PRIORITIES = [
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

class DataLoader:

    @staticmethod
    def load_schweinfurt_data(data_file: str) -> pd.DataFrame:
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

    @staticmethod
    def split_data_by_date(data: pd.DataFrame,
                           train_end: str = '2023-12-31',
                           val_end: str = '2024-06-30') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data = data[data.index <= train_end]
        val_data = data[(data.index > train_end) & (data.index <= val_end)]
        test_data = data[data.index > val_end]

        print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data

class FeatureSelector:

    @staticmethod
    def select_features(data: pd.DataFrame, n_features: int = 33,
                        priority_features: List[str] = None) -> List[str]:
        if priority_features is None:
            priority_features = DEFAULT_FEATURE_PRIORITIES

        numeric_columns = [col for col in data.columns
                           if data[col].dtype in ['int64', 'float64', 'int32', 'float32']]

        selected_features = [f for f in priority_features if f in numeric_columns]

        remaining_features = [col for col in numeric_columns if col not in selected_features]
        selected_features.extend(remaining_features[:n_features-len(selected_features)])

        if len(selected_features) < n_features:
            for i in range(n_features - len(selected_features)):
                selected_features.append(f'synthetic_feature_{i}')

        return selected_features[:n_features]

    @staticmethod
    def create_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()

        if 'hour' not in data_copy.columns:
            data_copy['hour'] = data_copy.index.hour
        if 'day_of_week' not in data_copy.columns:
            data_copy['day_of_week'] = data_copy.index.dayofweek
        if 'month' not in data_copy.columns:
            data_copy['month'] = data_copy.index.month
        if 'season' not in data_copy.columns:
            data_copy['season'] = (data_copy.index.month % 12) // 3

        if 'hour_sin' not in data_copy.columns:
            data_copy['hour_sin'] = np.sin(2 * np.pi * data_copy['hour'] / 24)
        if 'hour_cos' not in data_copy.columns:
            data_copy['hour_cos'] = np.cos(2 * np.pi * data_copy['hour'] / 24)

        if 'is_weekend' not in data_copy.columns:
            data_copy['is_weekend'] = (data_copy.index.dayofweek >= 5).astype(int)

        if 'is_daytime' not in data_copy.columns:
            data_copy['is_daytime'] = ((data_copy.index.hour >= 6) &
                                       (data_copy.index.hour < 18)).astype(int)

        return data_copy

class ModelUtils:

    @staticmethod
    def load_svr_model(model_path: str) -> Tuple[Any, Optional[Any]]:
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            if isinstance(model_data, dict):
                svr_model = model_data['model']
                svr_scaler = model_data.get('scaler', None)
                model_info = model_data.get('model_info', {})
                print(f"Loaded SVR model: {model_info}")
            else:
                svr_model = model_data
                svr_scaler = None
                print("Loaded legacy SVR model format")

            print("SVR model loaded successfully")
            return svr_model, svr_scaler

        except Exception as e:
            print(f"Error loading SVR model: {e}")
            raise

    @staticmethod
    def save_model_results(results: Dict[str, Any], filepath: str):
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to: {filepath}")

    @staticmethod
    def load_results(filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            results = json.load(f)
        return results

class HeatDemandUtils:

    @staticmethod
    def calculate_hdd(temperature: Union[float, np.ndarray],
                      base_temp: float = 15.5) -> Union[float, np.ndarray]:
        return np.maximum(base_temp - temperature, 0)

    @staticmethod
    def estimate_demand_from_temperature(temperature: Union[float, np.ndarray],
                                         hdd_multiplier: float = 3.5,
                                         base_demand: float = 5.0) -> Union[float, np.ndarray]:
        hdd = HeatDemandUtils.calculate_hdd(temperature)
        demand = base_demand + hdd * hdd_multiplier
        return np.maximum(demand, 0.1)

    @staticmethod
    def distribute_demand_across_zones(total_demand: Union[float, np.ndarray],
                                       zone_weights: np.ndarray = None) -> np.ndarray:
        if zone_weights is None:
            zone_weights = ZONE_WEIGHTS

        return total_demand * zone_weights

    @staticmethod
    def calculate_demand_satisfaction(production: np.ndarray,
                                      demand: np.ndarray) -> float:
        total_production = np.sum(production)
        total_demand = np.sum(demand)

        if total_demand == 0:
            return 1.0 if total_production == 0 else 0.0

        return min(total_production, total_demand) / total_demand

class ConfigManager:

    @staticmethod
    def get_ppo_config() -> Dict[str, Any]:
        return {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'policy_kwargs': {'net_arch': [256, 256]}
        }

    @staticmethod
    def get_sac_config() -> Dict[str, Any]:
        return {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'gamma': 0.99,
            'tau': 0.005,
            'ent_coef': 'auto',
            'target_entropy': 'auto',
            'policy_kwargs': {'net_arch': [256, 256]}
        }

    @staticmethod
    def get_td3_config() -> Dict[str, Any]:
        return {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'policy_kwargs': {'net_arch': [256, 256]}
        }

    @staticmethod
    def get_training_config() -> Dict[str, Any]:
        return {
            'total_timesteps': 100000,
            'eval_freq': 5000,
            'n_eval_episodes': 5,
            'episode_length': 24,
            'train_start_date': '2021-01-01',
            'eval_start_date': '2024-01-01'
        }

class ValidationUtils:

    @staticmethod
    def validate_environment_setup(data_file: str, svr_model_path: str) -> bool:
        issues = []

        if not os.path.exists(data_file):
            issues.append(f"Dataset not found: {data_file}")

        if not os.path.exists(svr_model_path):
            issues.append(f"SVR model not found: {svr_model_path}")

        if issues:
            print("Validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("Environment setup validation passed!")
        return True

    @staticmethod
    def test_environment_compatibility(env_class, env_kwargs: Dict) -> bool:
        try:
            env = env_class(**env_kwargs)

            obs, info = env.reset()
            assert obs.shape == env.observation_space.shape

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print("Environment compatibility test passed!")
            return True

        except Exception as e:
            print(f"Environment compatibility test failed: {e}")
            return False

def estimate_training_time(timesteps: int, algorithm: str) -> int:
    base_times = {'PPO': 0.5, 'SAC': 0.8, 'TD3': 0.9, 'A2C': 0.3}
    return int((timesteps / 1000) * base_times.get(algorithm.upper(), 0.5))

def safe_get_value(row: pd.Series, column: str, default: float) -> float:
    if column in row.index:
        try:
            value = float(row[column])
            return value if not np.isnan(value) else default
        except (ValueError, TypeError):
            return default
    return default

def create_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(log_dir: str, algorithm: str) -> str:
    log_path = Path(log_dir) / algorithm.lower()
    log_path.mkdir(parents=True, exist_ok=True)
    return str(log_path)

if __name__ == "__main__":
    print("District Heating Utilities Module")
    print("Available utilities:")
    print("- DataLoader: Load and preprocess Schweinfurt data")
    print("- FeatureSelector: Feature selection and engineering")
    print("- ModelUtils: Model loading/saving utilities")
    print("- HeatDemandUtils: Heat demand calculations")
    print("- ConfigManager: Algorithm configurations")
    print("- ValidationUtils: Environment validation")
    print("\nExample:")
    print("from common.utils import DataLoader, ConfigManager")
    print("data = DataLoader.load_schweinfurt_data('data/schweinfurt_data.csv')")
    print("sac_config = ConfigManager.get_sac_config()")