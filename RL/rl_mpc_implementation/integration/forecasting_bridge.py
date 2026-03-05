import numpy as np
import pandas as pd
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.mpc_config import get_mpc_config, MPCConfig

class SVRForecastingBridge:

    def __init__(self, config: Optional[MPCConfig] = None,
                 svr_model_path: Optional[str] = None,
                 data_path: Optional[str] = None):

        self.config = config if config is not None else get_mpc_config()
        self.forecast_params = self.config.forecasting_params

        if svr_model_path is None:
            parent_dir = Path(__file__).parent.parent.parent.parent
            svr_model_path = str(parent_dir / "04_svr_implementation" / "results" / "33features" / "svr_model_24h_20250825_191319.pkl")

        if data_path is None:
            parent_dir = Path(__file__).parent.parent.parent.parent
            data_path = str(parent_dir / "01_data" / "processed_data" / "merged_dataset.csv")

        self.svr_models = {}
        self.svr_scaler = None
        self.feature_columns = []
        self.data = None

        self._load_svr_models(svr_model_path)
        self._load_data(data_path)
        self._initialize_feature_columns()

        self.zone_weights = self.forecast_params['zone_demand_weights']
        self.forecast_cache = {}
        self.cache_validity_hours = 1

        print("SVR Forecasting Bridge initialized")
        print(f"Available horizons: {list(self.svr_models.keys())} hours")
        print(f"Primary horizon: {self.forecast_params['primary_horizon']}h")
        print(f"Feature columns: {len(self.feature_columns)}")

    def _load_svr_models(self, model_path: str):
        try:
            horizons = self.forecast_params['svr_horizons']

            for horizon in horizons:
                possible_paths = [
                    str(Path(model_path).parent / f"svr_model_{horizon}h_{Path(model_path).stem.split('_')[-1]}.pkl"),
                    str(Path(model_path).parent / f"svr_model_{horizon}h_20250825_191319.pkl"),
                    model_path if f"{horizon}h" in model_path else None
                ]

                for path in possible_paths:
                    if path and Path(path).exists():
                        try:
                            with open(path, 'rb') as f:
                                model_data = pickle.load(f)

                            if isinstance(model_data, dict):
                                self.svr_models[horizon] = model_data.get('model', model_data)
                                if self.svr_scaler is None:
                                    self.svr_scaler = model_data.get('scaler')
                            else:
                                self.svr_models[horizon] = model_data

                            print(f"Loaded SVR model for {horizon}h horizon from {Path(path).name}")
                            break
                        except Exception as e:
                            continue

            if not self.svr_models:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                if isinstance(model_data, dict):
                    primary_horizon = self.forecast_params['primary_horizon']
                    self.svr_models[primary_horizon] = model_data.get('model', model_data)
                    self.svr_scaler = model_data.get('scaler')
                else:
                    primary_horizon = self.forecast_params['primary_horizon']
                    self.svr_models[primary_horizon] = model_data

                print(f"Loaded single SVR model as {primary_horizon}h horizon")

            if self.svr_scaler is None:
                scaler_path = str(Path(model_path).parent / f"svr_scaler_{Path(model_path).stem.split('_')[-1]}.pkl")
                if Path(scaler_path).exists():
                    with open(scaler_path, 'rb') as f:
                        self.svr_scaler = pickle.load(f)
                    print(f"Loaded SVR scaler from {Path(scaler_path).name}")

        except Exception as e:
            print(f"Warning: Could not load SVR models: {e}")
            print("Falling back to synthetic demand estimation")

    def _load_data(self, data_path: str):
        try:
            if Path(data_path).exists():
                self.data = pd.read_csv(data_path)

                datetime_cols = ['datetime', 'timestamp', 'date', 'time']
                datetime_col = None
                for col in datetime_cols:
                    if col in self.data.columns:
                        datetime_col = col
                        break

                if datetime_col:
                    self.data[datetime_col] = pd.to_datetime(self.data[datetime_col])
                    self.data = self.data.set_index(datetime_col)
                else:
                    self.data.index = pd.date_range(start='2021-01-01', periods=len(self.data), freq='H')

                self.data = self.data.fillna(method='ffill').fillna(method='bfill')

                print(f"Loaded data: {self.data.shape}, range: {self.data.index[0]} to {self.data.index[-1]}")
            else:
                print(f"Warning: Data file not found: {data_path}")
                self.data = None

        except Exception as e:
            print(f"Warning: Could not load data: {e}")
            self.data = None

    def _initialize_feature_columns(self):
        if self.data is None:
            self.feature_columns = [
                'temp', 'app_temp', 'dewpt', 'rh', 'pres', 'slp', 'wind_spd',
                'wind_dir', 'solar_rad', 'ghi', 'dhi', 'dni', 'clouds', 'vis',
                'B1_B2_expected_supply_temp', 'F1_Nord_expected_supply_temp',
                'F1_Sud_expected_supply_temp', 'Maintal_expected_supply_temp',
                'N1_expected_supply_temp', 'N2_expected_supply_temp',
                'V1_expected_supply_temp', 'V2_expected_supply_temp',
                'V6_expected_supply_temp', 'W1_expected_supply_temp',
                'ZN_expected_supply_temp',
                'hdd_18', 'hdd_15_5', 'hour', 'day_of_week', 'month', 'season',
                'temp_change', 'is_daytime', 'is_weekend'
            ]
            return

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

        self.feature_columns = selected_features[:33]

    def extract_features(self, current_time: pd.Timestamp,
                         outdoor_temp: float, hour: int) -> np.ndarray:
        features = np.zeros(33)

        if self.data is not None:
            try:
                time_diffs = np.abs((self.data.index - current_time).total_seconds())
                closest_idx = np.argmin(time_diffs)
                row = self.data.iloc[closest_idx]

                for i, feature_name in enumerate(self.feature_columns[:33]):
                    if feature_name in row.index:
                        value = row[feature_name]
                        if not pd.isna(value):
                            features[i] = float(value)

            except Exception as e:
                warnings.warn(f"Feature extraction from data failed: {e}")

        for i, feature_name in enumerate(self.feature_columns[:33]):
            if features[i] == 0:
                if 'temp' in feature_name.lower():
                    features[i] = outdoor_temp
                elif 'hour' in feature_name.lower():
                    features[i] = hour
                elif 'day' in feature_name.lower():
                    features[i] = current_time.dayofweek if hasattr(current_time, 'dayofweek') else 1
                elif 'month' in feature_name.lower():
                    features[i] = current_time.month if hasattr(current_time, 'month') else 6
                elif 'hdd' in feature_name.lower():
                    features[i] = max(0, 15.5 - outdoor_temp)
                elif 'rh' in feature_name.lower():
                    features[i] = 60.0
                elif 'pres' in feature_name.lower():
                    features[i] = 1013.0
                elif 'wind' in feature_name.lower():
                    features[i] = 5.0
                elif 'solar' in feature_name.lower():
                    features[i] = 200.0 if 6 <= hour <= 18 else 0.0
                else:
                    features[i] = 0.0

        return features

    def predict_demand_svr(self, features: np.ndarray, horizon: int) -> float:
        if horizon not in self.svr_models:
            available_horizons = list(self.svr_models.keys())
            if not available_horizons:
                return None

            closest_horizon = min(available_horizons, key=lambda x: abs(x - horizon))
            model = self.svr_models[closest_horizon]
        else:
            model = self.svr_models[horizon]

        try:
            if self.svr_scaler is not None:
                features_scaled = self.svr_scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)

            predicted_demand = model.predict(features_scaled)[0]

            return max(predicted_demand, 1.0)

        except Exception as e:
            warnings.warn(f"SVR prediction failed: {e}")
            return None

    def predict_demand_fallback(self, outdoor_temp: float, hour: int) -> float:
        base_demand = max(5.0, (self.forecast_params['base_demand_temperature'] - outdoor_temp) *
                          self.forecast_params['demand_temperature_slope'])

        time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24)
        total_demand = base_demand * time_factor

        return max(total_demand, self.forecast_params['minimum_demand'])

    def get_demand_forecast(self, current_time: pd.Timestamp,
                            outdoor_temp: float, hour: int,
                            horizon: int) -> Dict[str, Any]:
        cache_key = f"{current_time}_{outdoor_temp}_{hour}_{horizon}"
        if cache_key in self.forecast_cache:
            cached_result = self.forecast_cache[cache_key]
            if (pd.Timestamp.now() - cached_result['timestamp']).total_seconds() < 3600:
                return cached_result['forecast']

        features = self.extract_features(current_time, outdoor_temp, hour)

        svr_demand = None
        if self.svr_models:
            svr_demand = self.predict_demand_svr(features, horizon)

        if svr_demand is not None:
            total_demand = svr_demand
            prediction_method = 'svr'
            confidence = 0.85
        else:
            total_demand = self.predict_demand_fallback(outdoor_temp, hour)
            prediction_method = 'fallback'
            confidence = 0.6

        zone_demands = total_demand * self.zone_weights

        forecast = {
            'total_demand': float(total_demand),
            'zone_demands': zone_demands,
            'horizon': horizon,
            'prediction_method': prediction_method,
            'confidence': confidence,
            'outdoor_temp': outdoor_temp,
            'hour': hour,
            'features_used': len(features)
        }

        self.forecast_cache[cache_key] = {
            'forecast': forecast,
            'timestamp': pd.Timestamp.now()
        }

        if len(self.forecast_cache) > 100:
            oldest_keys = sorted(self.forecast_cache.keys())[:50]
            for key in oldest_keys:
                del self.forecast_cache[key]

        return forecast

    def get_multi_horizon_forecast(self, current_time: pd.Timestamp,
                                   outdoor_temp: float, hour: int,
                                   horizons: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
        if horizons is None:
            horizons = self.forecast_params['svr_horizons']

        forecasts = {}

        for horizon in horizons:
            try:
                forecast = self.get_demand_forecast(current_time, outdoor_temp, hour, horizon)
                forecasts[horizon] = forecast
            except Exception as e:
                warnings.warn(f"Failed to get forecast for {horizon}h horizon: {e}")
                fallback_demand = self.predict_demand_fallback(outdoor_temp, hour)
                forecasts[horizon] = {
                    'total_demand': fallback_demand,
                    'zone_demands': fallback_demand * self.zone_weights,
                    'horizon': horizon,
                    'prediction_method': 'fallback_error',
                    'confidence': 0.3,
                    'outdoor_temp': outdoor_temp,
                    'hour': hour,
                    'features_used': 0
                }

        return forecasts

    def get_mpc_prediction_horizon(self, current_time: pd.Timestamp,
                                   outdoor_temp: float, hour: int,
                                   prediction_horizon: int = 24) -> np.ndarray:
        primary_horizon = self.forecast_params['primary_horizon']
        backup_horizon = self.forecast_params['backup_horizon']

        try:
            forecast = self.get_demand_forecast(current_time, outdoor_temp, hour, primary_horizon)
            if forecast['confidence'] >= self.forecast_params['forecast_confidence_threshold']:
                base_demand = forecast['zone_demands']
            else:
                raise ValueError(f"Primary forecast confidence too low: {forecast['confidence']}")
        except:
            try:
                forecast = self.get_demand_forecast(current_time, outdoor_temp, hour, backup_horizon)
                base_demand = forecast['zone_demands']
            except:
                fallback_total = self.predict_demand_fallback(outdoor_temp, hour)
                base_demand = fallback_total * self.zone_weights

        horizon_demands = np.zeros((prediction_horizon, len(self.zone_weights)))

        for t in range(prediction_horizon):
            hour_future = (hour + t) % 24
            time_factor = 1.0 + 0.2 * np.sin(2 * np.pi * hour_future / 24)

            if self.forecast_params['demand_smoothing']:
                noise_factor = 1.0 + 0.05 * np.random.normal(0, 1)
                time_factor *= noise_factor

            horizon_demands[t] = base_demand * time_factor

        return horizon_demands

    def validate_forecast_quality(self, forecast: Dict[str, Any]) -> bool:
        if forecast['confidence'] < self.forecast_params['forecast_confidence_threshold']:
            return False

        total_demand = forecast['total_demand']
        if total_demand < self.forecast_params['minimum_demand'] or total_demand > 200.0:
            return False

        zone_demands = forecast['zone_demands']
        if np.any(zone_demands < 0) or np.sum(zone_demands) <= 0:
            return False

        return True

def test_forecasting_bridge():
    print("Testing SVR Forecasting Bridge...")
    print("=" * 50)

    try:
        bridge = SVRForecastingBridge()

        print(f"Forecasting bridge created successfully!")
        print(f"Available models: {len(bridge.svr_models)}")
        print(f"Has scaler: {bridge.svr_scaler is not None}")
        print(f"Has data: {bridge.data is not None}")

        current_time = pd.Timestamp('2024-03-15 14:00:00')
        outdoor_temp = 8.0
        hour = 14

        features = bridge.extract_features(current_time, outdoor_temp, hour)
        print(f"\nFeature extraction test:")
        print(f"Features shape: {features.shape}")
        print(f"Feature sample: {features[:5]}")

        horizon = 24
        forecast = bridge.get_demand_forecast(current_time, outdoor_temp, hour, horizon)

        print(f"\nSingle horizon forecast ({horizon}h):")
        print(f"Total demand: {forecast['total_demand']:.1f} MW")
        print(f"Method: {forecast['prediction_method']}")
        print(f"Confidence: {forecast['confidence']:.2f}")
        print(f"Zone demands (first 3): {forecast['zone_demands'][:3]}")

        horizons = [1, 24, 48]
        multi_forecast = bridge.get_multi_horizon_forecast(current_time, outdoor_temp, hour, horizons)

        print(f"\nMulti-horizon forecast:")
        for h, f in multi_forecast.items():
            print(f"  {h}h: {f['total_demand']:.1f} MW ({f['prediction_method']})")

        mpc_horizon = bridge.get_mpc_prediction_horizon(current_time, outdoor_temp, hour, 12)

        print(f"\nMPC prediction horizon test:")
        print(f"Horizon shape: {mpc_horizon.shape}")
        print(f"Demand range: {np.min(mpc_horizon):.1f} - {np.max(mpc_horizon):.1f} MW")
        print(f"Average total demand: {np.mean(np.sum(mpc_horizon, axis=1)):.1f} MW")

        is_valid = bridge.validate_forecast_quality(forecast)
        print(f"\nForecast validation: {'PASS' if is_valid else 'FAIL'}")

        print(f"\nCache test:")
        print(f"Cache entries before: {len(bridge.forecast_cache)}")

        forecast2 = bridge.get_demand_forecast(current_time, outdoor_temp, hour, horizon)
        print(f"Cache entries after: {len(bridge.forecast_cache)}")
        print(f"Same result: {abs(forecast['total_demand'] - forecast2['total_demand']) < 0.001}")

        print(f"\nForecasting bridge test completed successfully!")
        return True

    except Exception as e:
        print(f"Forecasting bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_forecasting_bridge()

    if success:
        print(f"\nSVR Forecasting Bridge ready!")
        print(f"Key capabilities:")
        print(f"  - Multi-horizon SVR integration")
        print(f"  - Feature extraction from your data")
        print(f"  - Fallback demand estimation")
        print(f"  - MPC horizon prediction")
        print(f"  - Forecast caching and validation")
    else:
        print(f"\nForecasting bridge test failed!")
        print(f"Check SVR model paths and data availability.")