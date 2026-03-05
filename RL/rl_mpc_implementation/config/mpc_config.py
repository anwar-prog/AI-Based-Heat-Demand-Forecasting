import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path

class MPCConfig:

    def __init__(self):
        self.zones = ['B1_B2', 'F1_Nord', 'F1_Sud', 'Maintal', 'N1', 'N2',
                      'V1', 'V2', 'V6', 'W1', 'ZN']
        self.n_zones = len(self.zones)
        self.zone_parameters = self._define_zone_parameters()
        self.optimization_params = self._define_optimization_parameters()
        self.physics_params = self._define_physics_parameters()
        self.cost_params = self._define_cost_parameters()
        self.constraints = self._define_constraints()
        self.forecasting_params = self._define_forecasting_parameters()

        print("MPC Configuration initialized for Schweinfurt District Heating Network")
        print(f"Zones configured: {self.n_zones}")

    def _define_zone_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'B1_B2': {
                'max_operating_temp': 130,
                'max_supply_temp': 120,
                'min_supply_temp': 70,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 10,
                'max_diff_pressure': 4.0,
                'elevation': 206,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.12,
                'thermal_capacity': 2.5,
                'heat_loss_coefficient': 0.15
            },
            'F1_Nord': {
                'max_operating_temp': 130,
                'max_supply_temp': 110,
                'min_supply_temp': 70,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 16,
                'max_diff_pressure': 2.4,
                'elevation': 225,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.08,
                'thermal_capacity': 1.8,
                'heat_loss_coefficient': 0.12
            },
            'F1_Sud': {
                'max_operating_temp': 130,
                'max_supply_temp': 120,
                'min_supply_temp': 70,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 25,
                'max_diff_pressure': 9.0,
                'elevation': 225,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.15,
                'thermal_capacity': 3.2,
                'heat_loss_coefficient': 0.18
            },
            'Maintal': {
                'max_operating_temp': 110,
                'max_supply_temp': 110,
                'min_supply_temp': 70,
                'max_return_temp': 50,
                'temp_breakpoint': None,
                'design_pressure': 16,
                'max_diff_pressure': 4.0,
                'elevation': 209,
                'control_strategy': 'constant',
                'zone_weight': 0.09,
                'thermal_capacity': 2.0,
                'heat_loss_coefficient': 0.13
            },
            'N1': {
                'max_operating_temp': 130,
                'max_supply_temp': 90,
                'min_supply_temp': 65,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 10,
                'max_diff_pressure': 3.0,
                'elevation': 220,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.07,
                'thermal_capacity': 1.5,
                'heat_loss_coefficient': 0.10
            },
            'N2': {
                'max_operating_temp': 90,
                'max_supply_temp': 80,
                'min_supply_temp': 60,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 10,
                'max_diff_pressure': 3.0,
                'elevation': 214,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.06,
                'thermal_capacity': 1.2,
                'heat_loss_coefficient': 0.08
            },
            'V1': {
                'max_operating_temp': 130,
                'max_supply_temp': 110,
                'min_supply_temp': 70,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 16,
                'max_diff_pressure': 3.5,
                'elevation': 229,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.11,
                'thermal_capacity': 2.3,
                'heat_loss_coefficient': 0.14
            },
            'V2': {
                'max_operating_temp': 130,
                'max_supply_temp': 75,
                'min_supply_temp': 65,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 25,
                'max_diff_pressure': 3.0,
                'elevation': 237,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.08,
                'thermal_capacity': 1.6,
                'heat_loss_coefficient': 0.11
            },
            'V6': {
                'max_operating_temp': 95,
                'max_supply_temp': 85,
                'min_supply_temp': 65,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 16,
                'max_diff_pressure': 2.0,
                'elevation': 237,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.05,
                'thermal_capacity': 1.0,
                'heat_loss_coefficient': 0.07
            },
            'W1': {
                'max_operating_temp': 130,
                'max_supply_temp': 110,
                'min_supply_temp': 70,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 25,
                'max_diff_pressure': 3.4,
                'elevation': 214,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.10,
                'thermal_capacity': 2.2,
                'heat_loss_coefficient': 0.13
            },
            'ZN': {
                'max_operating_temp': 130,
                'max_supply_temp': 120,
                'min_supply_temp': 70,
                'max_return_temp': 50,
                'temp_breakpoint': 8,
                'design_pressure': 25,
                'max_diff_pressure': 8.5,
                'elevation': 211,
                'control_strategy': 'bi_linear',
                'zone_weight': 0.09,
                'thermal_capacity': 2.0,
                'heat_loss_coefficient': 0.12
            }
        }

    def _define_optimization_parameters(self) -> Dict[str, Any]:
        return {
            'prediction_horizon': 24,
            'control_horizon': 12,
            'sampling_time': 1.0,
            'solver': 'OSQP',
            'solver_options': {
                'max_iter': 1000,
                'eps_abs': 1e-6,
                'eps_rel': 1e-6,
                'verbose': False
            },
            'cost_weight': 1.0,
            'comfort_weight': 2.0,
            'efficiency_weight': 0.5,
            'stability_weight': 0.3,
            'terminal_constraint': True,
            'terminal_cost_weight': 0.1,
            'uncertainty_margin': 0.05,
            'constraint_softening': True,
            'soft_constraint_penalty': 1000.0
        }

    def _define_physics_parameters(self) -> Dict[str, Any]:
        return {
            'ambient_temperature': 10.0,
            'pipe_thermal_resistance': 0.1,
            'network_thermal_inertia': 0.8,
            'supply_heat_transfer': 0.85,
            'return_heat_transfer': 0.75,
            'distribution_losses': 0.02,
            'max_flow_rate': 100.0,
            'min_flow_rate': 5.0,
            'pump_efficiency': 0.85,
            'mixing_efficiency': 0.95,
            'thermal_lag': 0.5,
            'water_specific_heat': 4.18,
            'water_density': 1000.0,
            'conversion_factor': 3.6
        }

    def _define_cost_parameters(self) -> Dict[str, Any]:
        return {
            'base_heat_cost': 45.0,
            'peak_hour_multiplier': 1.5,
            'efficiency_penalty': 10.0,
            'morning_peak_start': 6,
            'morning_peak_end': 10,
            'evening_peak_start': 17,
            'evening_peak_end': 21,
            'zone_cost_factors': {
                'B1_B2': 1.0,   'F1_Nord': 1.1,  'F1_Sud': 1.0,
                'Maintal': 0.9,  'N1': 1.2,      'N2': 1.1,
                'V1': 1.0,      'V2': 1.3,      'V6': 1.2,
                'W1': 1.0,      'ZN': 1.0
            },
            'target_daily_cost': 25.0,
            'acceptable_daily_cost': 30.0,
            'cost_tolerance': 5.0,
            'demand_shortage_penalty': 100.0,
            'oversupply_penalty': 20.0,
            'temperature_violation_penalty': 50.0
        }

    def _define_constraints(self) -> Dict[str, Any]:
        return {
            'hard_constraints': {
                'min_demand_satisfaction': 0.85,
                'max_temperature_violations': 0,
                'min_zone_production': 0.0,
                'max_zone_production': 100.0,
                'max_total_production': 800.0
            },
            'soft_constraints': {
                'preferred_demand_satisfaction': 0.95,
                'preferred_efficiency': 0.80,
                'temperature_comfort_margin': 5.0,
                'production_smoothness': 10.0
            },
            'zone_constraints': self._get_zone_constraints(),
            'operational_constraints': {
                'max_temperature_ramp': 10.0,
                'max_production_ramp': 20.0,
                'min_system_pressure': 2.0,
                'max_system_pressure': 30.0,
                'emergency_shutdown_temp': 140.0
            }
        }

    def _get_zone_constraints(self) -> Dict[str, Dict[str, float]]:
        zone_constraints = {}

        for zone_name, params in self.zone_parameters.items():
            zone_constraints[zone_name] = {
                'min_supply_temp': params['min_supply_temp'],
                'max_supply_temp': params['max_supply_temp'],
                'max_operating_temp': params['max_operating_temp'],
                'max_return_temp': params['max_return_temp'],
                'max_pressure': params['design_pressure'] * 1.1,
                'min_pressure': params['design_pressure'] * 0.8,
                'max_heat_production': 100.0 * params['zone_weight'],
                'min_heat_production': 0.0
            }

        return zone_constraints

    def _define_forecasting_parameters(self) -> Dict[str, Any]:
        return {
            'use_svr_forecasts': True,
            'svr_horizons': [1, 6, 24, 48, 72],
            'primary_horizon': 24,
            'backup_horizon': 48,
            'forecast_confidence_threshold': 0.8,
            'uncertainty_bands': {
                '1h': 0.05,   '6h': 0.10,   '24h': 0.15,
                '48h': 0.20,  '72h': 0.25
            },
            'zone_demand_weights': np.array([
                params['zone_weight'] for params in self.zone_parameters.values()
            ]),
            'fallback_demand_model': 'temperature_based',
            'base_demand_temperature': 15.5,
            'demand_temperature_slope': 3.5,
            'minimum_demand': 5.0,
            'demand_smoothing': True,
            'smoothing_window': 3,
            'outlier_detection': True,
            'outlier_threshold': 3.0
        }

    def get_zone_parameter(self, zone_name: str, parameter: str) -> Any:
        if zone_name not in self.zone_parameters:
            raise ValueError(f"Unknown zone: {zone_name}")
        if parameter not in self.zone_parameters[zone_name]:
            raise ValueError(f"Unknown parameter: {parameter}")
        return self.zone_parameters[zone_name][parameter]

    def get_optimization_weights(self) -> Dict[str, float]:
        return {
            'cost_weight': self.optimization_params['cost_weight'],
            'comfort_weight': self.optimization_params['comfort_weight'],
            'efficiency_weight': self.optimization_params['efficiency_weight'],
            'stability_weight': self.optimization_params['stability_weight']
        }

    def update_optimization_weights(self, new_weights: Dict[str, float]):
        for weight_name, value in new_weights.items():
            if weight_name in self.optimization_params:
                self.optimization_params[weight_name] = max(0.0, float(value))
            else:
                raise ValueError(f"Unknown weight: {weight_name}")

    def get_zone_names(self) -> List[str]:
        return self.zones.copy()

    def get_constraint_bounds(self, constraint_type: str = 'hard') -> Dict[str, Tuple[float, float]]:
        if constraint_type == 'hard':
            constraints = self.constraints['hard_constraints']
        else:
            constraints = self.constraints['soft_constraints']

        bounds = {}
        for constraint_name, value in constraints.items():
            if 'min_' in constraint_name:
                bounds[constraint_name] = (value, float('inf'))
            elif 'max_' in constraint_name:
                bounds[constraint_name] = (0.0, value)
            else:
                bounds[constraint_name] = (value, value)

        return bounds

    def validate_configuration(self) -> bool:
        try:
            total_weight = sum(p['zone_weight'] for p in self.zone_parameters.values())
            if abs(total_weight - 1.0) > 0.01:
                print(f"Warning: Zone weights sum to {total_weight:.3f}, not 1.0")

            for zone_name, params in self.zone_parameters.items():
                if params['min_supply_temp'] >= params['max_supply_temp']:
                    print(f"Error: {zone_name} min supply temp >= max supply temp")
                    return False

                if params['max_supply_temp'] > params['max_operating_temp']:
                    print(f"Error: {zone_name} max supply temp > max operating temp")
                    return False

            cost_params = self.cost_params
            if cost_params['base_heat_cost'] <= 0:
                print("Error: Base heat cost must be positive")
                return False

            print("MPC configuration validation passed")
            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

def get_mpc_config() -> MPCConfig:
    config = MPCConfig()

    if not config.validate_configuration():
        raise RuntimeError("MPC configuration validation failed")

    return config

DEFAULT_ZONES = ['B1_B2', 'F1_Nord', 'F1_Sud', 'Maintal', 'N1', 'N2',
                 'V1', 'V2', 'V6', 'W1', 'ZN']

DEFAULT_OPTIMIZATION_WEIGHTS = {
    'cost_weight': 1.0,
    'comfort_weight': 2.0,
    'efficiency_weight': 0.5,
    'stability_weight': 0.3
}

TARGET_PERFORMANCE = {
    'daily_cost_target': 25.0,
    'demand_satisfaction_min': 0.85,
    'efficiency_target': 0.80,
    'cost_variance_max': 5.0
}

if __name__ == "__main__":
    print("MPC Configuration for Schweinfurt District Heating Network")
    print("=" * 60)

    config = get_mpc_config()

    print(f"\nConfiguration Summary:")
    print(f"  - Zones: {config.n_zones}")
    print(f"  - Prediction horizon: {config.optimization_params['prediction_horizon']} hours")
    print(f"  - Target daily cost: {config.cost_params['target_daily_cost']} €")
    print(f"  - Minimum demand satisfaction: {config.constraints['hard_constraints']['min_demand_satisfaction']*100}%")

    print(f"\nZone Configuration:")
    for zone in config.zones[:3]:
        params = config.zone_parameters[zone]
        print(f"  {zone}: {params['min_supply_temp']}-{params['max_supply_temp']}°C, weight={params['zone_weight']:.2f}")

    print(f"\nOptimization Weights:")
    weights = config.get_optimization_weights()
    for name, value in weights.items():
        print(f"  {name}: {value:.1f}")

    print(f"\nMPC configuration ready for physics modeling!")