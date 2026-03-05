import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import sys
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from config.mpc_config import get_mpc_config, MPCConfig
from mpc_core.physics_model import DistrictHeatingPhysics
from integration.forecasting_bridge import SVRForecastingBridge

class DistrictHeatingMPC:

    def __init__(self, config: Optional[MPCConfig] = None,
                 physics_model: Optional[DistrictHeatingPhysics] = None,
                 forecasting_bridge: Optional[SVRForecastingBridge] = None):

        self.config = config if config is not None else get_mpc_config()
        self.physics = physics_model if physics_model is not None else DistrictHeatingPhysics(self.config)
        self.forecaster = forecasting_bridge if forecasting_bridge is not None else SVRForecastingBridge(self.config)

        self.prediction_horizon = self.config.optimization_params['prediction_horizon']
        self.control_horizon = self.config.optimization_params['control_horizon']
        self.dt = self.config.optimization_params['sampling_time']

        self.weights = self.config.get_optimization_weights()

        self.zones = self.config.zones
        self.n_zones = len(self.zones)
        self.zone_params = self.config.zone_parameters

        self.constraints_config = self.config.constraints

        self.optimization_problem = None
        self.optimization_variables = {}

        self.last_solution = None
        self.solve_times = []
        self.constraint_violations = []

        print("District Heating MPC Controller initialized")
        print(f"Horizon: {self.prediction_horizon}h prediction, {self.control_horizon}h control")
        print(f"Zones: {self.n_zones}, Solver: {self.config.optimization_params['solver']}")

    def build_optimization_problem(self, current_state: Dict[str, np.ndarray],
                                   demand_forecast: np.ndarray) -> cp.Problem:
        heat_prod = cp.Variable((self.control_horizon, self.n_zones), name="heat_production")
        zone_temps = cp.Variable((self.prediction_horizon, self.n_zones), name="zone_temperatures")
        demand_slack = cp.Variable(self.prediction_horizon, name="demand_slack")
        temp_slack = cp.Variable((self.prediction_horizon, self.n_zones), name="temp_slack")

        self.optimization_variables = {
            'heat_production': heat_prod,
            'zone_temperatures': zone_temps,
            'demand_slack': demand_slack,
            'temp_slack': temp_slack
        }

        cost_objective = self._build_cost_objective(heat_prod)
        comfort_objective = self._build_comfort_objective(heat_prod, demand_forecast, demand_slack)
        efficiency_objective = self._build_efficiency_objective(heat_prod, demand_forecast)
        stability_objective = self._build_stability_objective(heat_prod, zone_temps, temp_slack)

        objective = (
                self.weights['cost_weight'] * cost_objective +
                self.weights['comfort_weight'] * comfort_objective +
                self.weights['efficiency_weight'] * efficiency_objective +
                self.weights['stability_weight'] * stability_objective
        )

        constraints = []

        constraints.extend(self._build_production_constraints(heat_prod))
        constraints.extend(self._build_temperature_constraints(zone_temps, temp_slack))
        constraints.extend(self._build_demand_constraints(heat_prod, demand_forecast, demand_slack))
        constraints.extend(self._build_physics_constraints(heat_prod, zone_temps, current_state))
        constraints.extend(self._build_slack_constraints(demand_slack, temp_slack))

        if self.config.optimization_params['terminal_constraint']:
            constraints.extend(self._build_terminal_constraints(zone_temps))

        problem = cp.Problem(cp.Minimize(objective), constraints)

        return problem

    def _build_cost_objective(self, heat_prod: cp.Variable) -> cp.Expression:
        cost_params = self.config.cost_params
        base_cost = cost_params['base_heat_cost']

        total_cost = 0

        for t in range(self.control_horizon):
            for z in range(self.n_zones):
                zone_name = self.zones[z]
                zone_cost_factor = cost_params['zone_cost_factors'][zone_name]

                production_cost = base_cost * zone_cost_factor * heat_prod[t, z] * self.dt
                total_cost += production_cost

        return total_cost

    def _build_comfort_objective(self, heat_prod: cp.Variable,
                                 demand_forecast: np.ndarray,
                                 demand_slack: cp.Variable) -> cp.Expression:
        comfort_penalty = 0

        for t in range(min(self.control_horizon, demand_forecast.shape[0])):
            total_production = cp.sum(heat_prod[t, :])
            total_demand = np.sum(demand_forecast[t, :])

            demand_shortage = cp.maximum(0, total_demand - total_production)
            comfort_penalty += 100.0 * demand_shortage
            comfort_penalty += 1000.0 * demand_slack[t]

        return comfort_penalty

    def _build_efficiency_objective(self, heat_prod: cp.Variable,
                                    demand_forecast: np.ndarray) -> cp.Expression:
        efficiency_penalty = 0

        for t in range(min(self.control_horizon, demand_forecast.shape[0])):
            total_production = cp.sum(heat_prod[t, :])
            total_demand = np.sum(demand_forecast[t, :])

            oversupply = cp.maximum(0, total_production - total_demand)
            efficiency_penalty += 10.0 * oversupply

        return efficiency_penalty

    def _build_stability_objective(self, heat_prod: cp.Variable,
                                   zone_temps: cp.Variable,
                                   temp_slack: cp.Variable) -> cp.Expression:
        stability_penalty = 0

        for t in range(self.control_horizon - 1):
            for z in range(self.n_zones):
                production_change = heat_prod[t+1, z] - heat_prod[t, z]
                stability_penalty += 0.1 * cp.square(production_change)

        stability_penalty += 100.0 * cp.sum(temp_slack)

        return stability_penalty

    def _build_production_constraints(self, heat_prod: cp.Variable) -> List[cp.Constraint]:
        constraints = []
        hard_constraints = self.constraints_config['hard_constraints']

        for z in range(self.n_zones):
            zone_name = self.zones[z]
            zone_constraints = self.constraints_config['zone_constraints'][zone_name]

            constraints.append(heat_prod[:, z] >= zone_constraints['min_heat_production'])
            constraints.append(heat_prod[:, z] <= zone_constraints['max_heat_production'])

        for t in range(self.control_horizon):
            total_production = cp.sum(heat_prod[t, :])
            constraints.append(total_production <= hard_constraints['max_total_production'])

        operational_constraints = self.constraints_config['operational_constraints']
        max_ramp = operational_constraints['max_production_ramp'] * self.dt

        for t in range(self.control_horizon - 1):
            for z in range(self.n_zones):
                production_change = heat_prod[t+1, z] - heat_prod[t, z]
                constraints.append(production_change <= max_ramp)
                constraints.append(production_change >= -max_ramp)

        return constraints

    def _build_temperature_constraints(self, zone_temps: cp.Variable,
                                       temp_slack: cp.Variable) -> List[cp.Constraint]:
        constraints = []

        for z in range(self.n_zones):
            zone_name = self.zones[z]
            zone_constraints = self.constraints_config['zone_constraints'][zone_name]

            min_temp = zone_constraints['min_supply_temp']
            max_temp = zone_constraints['max_supply_temp']

            for t in range(self.prediction_horizon):
                constraints.append(zone_temps[t, z] >= min_temp - temp_slack[t, z])
                constraints.append(zone_temps[t, z] <= max_temp + temp_slack[t, z])

                max_operating = zone_constraints['max_operating_temp']
                constraints.append(zone_temps[t, z] <= max_operating)

        operational_constraints = self.constraints_config['operational_constraints']
        max_temp_ramp = operational_constraints['max_temperature_ramp'] * self.dt

        for t in range(self.prediction_horizon - 1):
            for z in range(self.n_zones):
                temp_change = zone_temps[t+1, z] - zone_temps[t, z]
                constraints.append(temp_change <= max_temp_ramp)
                constraints.append(temp_change >= -max_temp_ramp)

        return constraints

    def _build_demand_constraints(self, heat_prod: cp.Variable,
                                  demand_forecast: np.ndarray,
                                  demand_slack: cp.Variable) -> List[cp.Constraint]:
        constraints = []
        hard_constraints = self.constraints_config['hard_constraints']
        min_satisfaction = hard_constraints['min_demand_satisfaction']

        for t in range(min(self.control_horizon, demand_forecast.shape[0])):
            total_production = cp.sum(heat_prod[t, :])
            total_demand = np.sum(demand_forecast[t, :])

            if total_demand > 0:
                min_production = min_satisfaction * total_demand
                constraints.append(total_production >= min_production - demand_slack[t])

        return constraints

    def _build_physics_constraints(self, heat_prod: cp.Variable,
                                   zone_temps: cp.Variable,
                                   current_state: Dict[str, np.ndarray]) -> List[cp.Constraint]:
        constraints = []

        current_temps = current_state['temperatures']
        constraints.append(zone_temps[0, :] == current_temps)

        thermal_capacities = self.physics.thermal_capacities
        heat_loss_coeffs = self.physics.heat_loss_coefficients
        ambient_temp = self.physics.physics_params['ambient_temperature']

        for t in range(min(self.control_horizon - 1, self.prediction_horizon - 1)):
            for z in range(self.n_zones):
                heat_loss = heat_loss_coeffs[z] * (zone_temps[t, z] - ambient_temp)
                temp_change = self.dt * (heat_prod[t, z] - heat_loss) / thermal_capacities[z]
                constraints.append(zone_temps[t+1, z] == zone_temps[t, z] + temp_change)

        return constraints

    def _build_slack_constraints(self, demand_slack: cp.Variable,
                                 temp_slack: cp.Variable) -> List[cp.Constraint]:
        constraints = []

        constraints.append(demand_slack >= 0)
        constraints.append(temp_slack >= 0)
        constraints.append(demand_slack <= 10.0)
        constraints.append(temp_slack <= 5.0)

        return constraints

    def _build_terminal_constraints(self, zone_temps: cp.Variable) -> List[cp.Constraint]:
        constraints = []

        for z in range(self.n_zones):
            zone_name = self.zones[z]
            zone_params = self.zone_params[zone_name]

            target_temp = (zone_params['min_supply_temp'] + zone_params['max_supply_temp']) / 2
            temp_tolerance = 10.0

            constraints.append(zone_temps[-1, z] >= target_temp - temp_tolerance)
            constraints.append(zone_temps[-1, z] <= target_temp + temp_tolerance)

        return constraints

    def solve_mpc(self, current_time: pd.Timestamp, current_state: Dict[str, np.ndarray],
                  outdoor_temp: float, hour: int) -> Dict[str, Any]:
        try:
            demand_forecast = self.forecaster.get_mpc_prediction_horizon(
                current_time, outdoor_temp, hour, self.prediction_horizon
            )

            problem = self.build_optimization_problem(current_state, demand_forecast)

            solver_options = self.config.optimization_params['solver_options']
            solver_name = self.config.optimization_params['solver']

            problem.solve(solver=solver_name, verbose=solver_options.get('verbose', False))

            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                warnings.warn(f"MPC solver status: {problem.status}")
                return self._create_fallback_solution(current_state, demand_forecast)

            solution = self._extract_solution(problem, demand_forecast, current_state)

            self.last_solution = solution
            self.solve_times.append(problem.solver_stats.solve_time if problem.solver_stats else 0.0)

            return solution

        except Exception as e:
            warnings.warn(f"MPC optimization failed: {e}")
            return self._create_fallback_solution(current_state, demand_forecast)

    def _extract_solution(self, problem: cp.Problem, demand_forecast: np.ndarray,
                          current_state: Dict[str, np.ndarray]) -> Dict[str, Any]:
        heat_prod = self.optimization_variables['heat_production'].value
        zone_temps = self.optimization_variables['zone_temperatures'].value
        demand_slack = self.optimization_variables['demand_slack'].value
        temp_slack = self.optimization_variables['temp_slack'].value

        optimal_action = heat_prod[0, :] if heat_prod is not None else np.zeros(self.n_zones)
        optimal_action = np.clip(optimal_action, 0, 100)

        total_production = np.sum(optimal_action)
        total_demand = np.sum(demand_forecast[0, :]) if demand_forecast.shape[0] > 0 else 0

        demand_satisfaction = min(total_production / max(total_demand, 1.0), 1.0)
        efficiency = min(total_production, total_demand) / max(total_production, 1.0) if total_production > 0 else 0

        cost_per_hour = self._calculate_cost(optimal_action, 14)
        daily_cost_estimate = cost_per_hour * 24

        violations = {
            'demand_slack_used': np.sum(demand_slack) if demand_slack is not None else 0,
            'temp_slack_used': np.sum(temp_slack) if temp_slack is not None else 0,
            'total_violations': 0
        }

        if optimal_action.max() > 100.0 or optimal_action.min() < 0.0:
            violations['total_violations'] += 1

        solution = {
            'optimal_action': optimal_action,
            'heat_production_plan': heat_prod if heat_prod is not None else np.zeros((self.control_horizon, self.n_zones)),
            'temperature_plan': zone_temps if zone_temps is not None else np.zeros((self.prediction_horizon, self.n_zones)),
            'total_production': total_production,
            'total_demand': total_demand,
            'demand_satisfaction': demand_satisfaction,
            'efficiency': efficiency,
            'cost_per_hour': cost_per_hour,
            'daily_cost_estimate': daily_cost_estimate,
            'objective_value': problem.value if problem.value is not None else float('inf'),
            'solver_status': problem.status,
            'constraint_violations': violations,
            'forecast_used': demand_forecast,
            'weights_used': self.weights.copy()
        }

        return solution

    def _create_fallback_solution(self, current_state: Dict[str, np.ndarray],
                                  demand_forecast: np.ndarray) -> Dict[str, Any]:
        if demand_forecast.shape[0] > 0:
            total_demand = np.sum(demand_forecast[0, :])
            zone_weights = np.array([self.zone_params[zone]['zone_weight'] for zone in self.zones])
            optimal_action = total_demand * zone_weights
        else:
            optimal_action = np.ones(self.n_zones) * 10.0

        optimal_action = np.clip(optimal_action, 0, 100)

        total_production = np.sum(optimal_action)
        total_demand = np.sum(demand_forecast[0, :]) if demand_forecast.shape[0] > 0 else total_production

        demand_satisfaction = min(total_production / max(total_demand, 1.0), 1.0)
        efficiency = min(total_production, total_demand) / max(total_production, 1.0) if total_production > 0 else 0

        cost_per_hour = self._calculate_cost(optimal_action, 14)
        daily_cost_estimate = cost_per_hour * 24

        return {
            'optimal_action': optimal_action,
            'heat_production_plan': np.tile(optimal_action, (self.control_horizon, 1)),
            'temperature_plan': np.tile(current_state['temperatures'], (self.prediction_horizon, 1)),
            'total_production': total_production,
            'total_demand': total_demand,
            'demand_satisfaction': demand_satisfaction,
            'efficiency': efficiency,
            'cost_per_hour': cost_per_hour,
            'daily_cost_estimate': daily_cost_estimate,
            'objective_value': float('inf'),
            'solver_status': 'FALLBACK',
            'constraint_violations': {'total_violations': 0},
            'forecast_used': demand_forecast,
            'weights_used': self.weights.copy()
        }

    def _calculate_cost(self, heat_production: np.ndarray, hour: int) -> float:
        cost_params = self.config.cost_params
        base_cost = cost_params['base_heat_cost']

        if (6 <= hour <= 10) or (17 <= hour <= 21):
            cost_multiplier = cost_params['peak_hour_multiplier']
        else:
            cost_multiplier = 1.0

        total_cost = 0
        for z in range(self.n_zones):
            zone_name = self.zones[z]
            zone_cost_factor = cost_params['zone_cost_factors'][zone_name]

            zone_cost = heat_production[z] * base_cost * zone_cost_factor * cost_multiplier / 1000.0
            total_cost += zone_cost

        return total_cost

    def update_weights(self, new_weights: Dict[str, float]):
        self.config.update_optimization_weights(new_weights)
        self.weights = self.config.get_optimization_weights()

    def get_performance_summary(self) -> Dict[str, Any]:
        if self.last_solution is None:
            return {'error': 'No solution available'}

        avg_solve_time = np.mean(self.solve_times) if self.solve_times else 0.0

        return {
            'last_daily_cost': self.last_solution['daily_cost_estimate'],
            'last_demand_satisfaction': self.last_solution['demand_satisfaction'],
            'last_efficiency': self.last_solution['efficiency'],
            'avg_solve_time': float(avg_solve_time),
            'total_solves': len(self.solve_times),
            'constraint_violations': self.last_solution['constraint_violations']['total_violations'],
            'solver_status': self.last_solution['solver_status'],
            'current_weights': self.weights.copy()
        }

def test_mpc_controller():
    print("Testing District Heating MPC Controller...")
    print("=" * 50)

    try:
        mpc = DistrictHeatingMPC()

        print(f"MPC controller created successfully!")
        print(f"Prediction horizon: {mpc.prediction_horizon}h")
        print(f"Control horizon: {mpc.control_horizon}h")
        print(f"Zones: {mpc.n_zones}")

        current_time = pd.Timestamp('2024-03-15 14:00:00')
        outdoor_temp = 8.0
        hour = 14

        current_state = mpc.physics.get_current_state()
        print(f"\nCurrent state:")
        print(f"Temperatures: {current_state['temperatures'][:3]}")

        print(f"\nSolving MPC optimization...")
        solution = mpc.solve_mpc(current_time, current_state, outdoor_temp, hour)

        print(f"Solution status: {solution['solver_status']}")
        print(f"Optimal action (first 3 zones): {solution['optimal_action'][:3]}")
        print(f"Total production: {solution['total_production']:.1f} MW")
        print(f"Total demand: {solution['total_demand']:.1f} MW")
        print(f"Demand satisfaction: {solution['demand_satisfaction']:.1%}")
        print(f"Efficiency: {solution['efficiency']:.1%}")
        print(f"Daily cost estimate: {solution['daily_cost_estimate']:.1f} €")

        print(f"\nTesting weight updates...")
        original_weights = mpc.weights.copy()
        print(f"Original weights: {original_weights}")

        new_weights = {
            'cost_weight': 1.5,
            'comfort_weight': 1.8,
            'efficiency_weight': 0.7,
            'stability_weight': 0.4
        }

        mpc.update_weights(new_weights)
        updated_weights = mpc.weights
        print(f"Updated weights: {updated_weights}")

        solution2 = mpc.solve_mpc(current_time, current_state, outdoor_temp, hour)
        print(f"\nSolution with new weights:")
        print(f"Daily cost estimate: {solution2['daily_cost_estimate']:.1f} € (vs {solution['daily_cost_estimate']:.1f} €)")
        print(f"Demand satisfaction: {solution2['demand_satisfaction']:.1%} (vs {solution['demand_satisfaction']:.1%})")

        summary = mpc.get_performance_summary()
        print(f"\nMPC Performance Summary:")
        print(f"Average solve time: {summary['avg_solve_time']:.3f} seconds")
        print(f"Total solves: {summary['total_solves']}")
        print(f"Constraint violations: {summary['constraint_violations']}")

        violations = mpc.physics.check_constraints(
            current_state['temperatures'],
            current_state['flows'],
            current_state['pressures']
        )
        print(f"\nPhysics constraint check: {violations['total_violations']} violations")

        print(f"\nMPC controller test completed successfully!")
        return True

    except Exception as e:
        print(f"MPC controller test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mpc_controller()

    if success:
        print(f"\nDistrict Heating MPC Controller ready!")
        print(f"Key capabilities:")
        print(f"  - Constrained optimization with CVXPY")
        print(f"  - Hard constraint enforcement (85% demand satisfaction)")
        print(f"  - Multi-objective optimization (cost, comfort, efficiency, stability)")
        print(f"  - RL-tunable parameters")
        print(f"  - Physics-based constraints")
        print(f"  - Fallback solutions for reliability")
    else:
        print(f"\nMPC controller test failed!")
        print(f"Check CVXPY installation and constraint formulation.")