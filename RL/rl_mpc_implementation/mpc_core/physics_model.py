import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.mpc_config import get_mpc_config, MPCConfig

class DistrictHeatingPhysics:

    def __init__(self, config: Optional[MPCConfig] = None):
        self.config = config if config is not None else get_mpc_config()

        self.zones = self.config.zones
        self.n_zones = len(self.zones)
        self.zone_params = self.config.zone_parameters
        self.physics_params = self.config.physics_params

        self.zone_temperatures = self._initialize_zone_temperatures()
        self.zone_flows = np.zeros(self.n_zones)
        self.zone_pressures = self._initialize_zone_pressures()

        self.thermal_capacities = self._get_thermal_capacities()
        self.heat_loss_coefficients = self._get_heat_loss_coefficients()
        self.coupling_matrix = self._build_coupling_matrix()

        self.dt = self.config.optimization_params['sampling_time']

        print("District Heating Physics Model initialized")
        print(f"Zones: {self.n_zones}, Time step: {self.dt}h")

    def _initialize_zone_temperatures(self) -> np.ndarray:
        temps = np.zeros(self.n_zones)

        for i, zone_name in enumerate(self.zones):
            zone_params = self.zone_params[zone_name]
            min_temp = zone_params['min_supply_temp']
            max_temp = zone_params['max_supply_temp']
            temps[i] = (min_temp + max_temp) / 2.0

        return temps

    def _initialize_zone_pressures(self) -> np.ndarray:
        pressures = np.zeros(self.n_zones)

        for i, zone_name in enumerate(self.zones):
            pressures[i] = self.zone_params[zone_name]['design_pressure']

        return pressures

    def _get_thermal_capacities(self) -> np.ndarray:
        capacities = np.zeros(self.n_zones)

        for i, zone_name in enumerate(self.zones):
            capacities[i] = self.zone_params[zone_name]['thermal_capacity']

        return capacities

    def _get_heat_loss_coefficients(self) -> np.ndarray:
        coefficients = np.zeros(self.n_zones)

        for i, zone_name in enumerate(self.zones):
            coefficients[i] = self.zone_params[zone_name]['heat_loss_coefficient']

        return coefficients

    def _build_coupling_matrix(self) -> np.ndarray:
        coupling = np.eye(self.n_zones)
        coupling_strength = 0.05

        for i in range(self.n_zones - 1):
            coupling[i, i + 1] = coupling_strength
            coupling[i + 1, i] = coupling_strength

        return coupling

    def calculate_expected_supply_temperature(self, zone_idx: int, outdoor_temp: float) -> float:
        zone_name = self.zones[zone_idx]
        zone_params = self.zone_params[zone_name]

        if zone_params['control_strategy'] == 'constant':
            return zone_params['max_supply_temp']

        min_temp = zone_params['min_supply_temp']
        max_temp = zone_params['max_supply_temp']
        breakpoint = zone_params['temp_breakpoint']

        min_outdoor_temp = -14.0

        if outdoor_temp >= breakpoint:
            return min_temp
        else:
            temp_slope = (max_temp - min_temp) / (breakpoint - min_outdoor_temp)
            supply_temp = min_temp + (breakpoint - outdoor_temp) * temp_slope
            return np.clip(supply_temp, min_temp, max_temp)

    def calculate_heat_demand(self, outdoor_temp: float, hour: int,
                              zone_weights: Optional[np.ndarray] = None) -> np.ndarray:
        if zone_weights is None:
            zone_weights = np.array([self.zone_params[zone]['zone_weight'] for zone in self.zones])

        base_demand = max(5.0, (15.5 - outdoor_temp) * 3.5)
        time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour / 24)
        seasonal_factor = 1.0

        total_demand = base_demand * time_factor * seasonal_factor
        zone_demands = total_demand * zone_weights

        return zone_demands

    def update_thermal_dynamics(self, heat_production: np.ndarray,
                                outdoor_temp: float, demand: np.ndarray) -> Dict[str, np.ndarray]:
        heat_production = np.array(heat_production)
        demand = np.array(demand)

        heat_balance = heat_production - demand

        ambient_temp = self.physics_params['ambient_temperature']
        heat_losses = self.heat_loss_coefficients * (self.zone_temperatures - ambient_temp)

        net_heat = heat_balance - heat_losses
        coupled_heat = self.coupling_matrix @ net_heat

        temp_derivatives = coupled_heat / self.thermal_capacities
        new_temperatures = self.zone_temperatures + self.dt * temp_derivatives

        new_temperatures = self._apply_temperature_constraints(new_temperatures)
        new_flows = self._calculate_flows(heat_production, new_temperatures)
        new_pressures = self._calculate_pressures(new_flows, new_temperatures)

        self.zone_temperatures = new_temperatures
        self.zone_flows = new_flows
        self.zone_pressures = new_pressures

        return {
            'temperatures': new_temperatures.copy(),
            'flows': new_flows.copy(),
            'pressures': new_pressures.copy(),
            'heat_balance': heat_balance.copy(),
            'heat_losses': heat_losses.copy()
        }

    def _apply_temperature_constraints(self, temperatures: np.ndarray) -> np.ndarray:
        constrained_temps = temperatures.copy()

        for i, zone_name in enumerate(self.zones):
            zone_params = self.zone_params[zone_name]
            min_temp = zone_params['min_supply_temp']
            max_temp = zone_params['max_operating_temp']

            constrained_temps[i] = np.clip(constrained_temps[i], min_temp, max_temp)

        return constrained_temps

    def _calculate_flows(self, heat_production: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
        flows = np.zeros(self.n_zones)

        cp = self.physics_params['water_specific_heat'] / 1000.0
        dt_supply_return = 30.0

        for i in range(self.n_zones):
            if heat_production[i] > 0 and dt_supply_return > 0:
                flows[i] = (heat_production[i] * 1000.0) / (cp * 1000.0 * dt_supply_return)
            else:
                flows[i] = self.physics_params['min_flow_rate']

            min_flow = self.physics_params['min_flow_rate']
            max_flow = self.physics_params['max_flow_rate']
            flows[i] = np.clip(flows[i], min_flow, max_flow)

        return flows

    def _calculate_pressures(self, flows: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
        pressures = np.zeros(self.n_zones)

        for i, zone_name in enumerate(self.zones):
            zone_params = self.zone_params[zone_name]

            elevation_pressure = zone_params['elevation'] * 0.01

            density = self.physics_params['water_density']
            flow_pressure = 0.5 * density * (flows[i] / 1000.0) ** 2 / 100000.0

            total_pressure = zone_params['design_pressure'] + elevation_pressure + flow_pressure

            min_pressure = zone_params['design_pressure'] * 0.8
            max_pressure = zone_params['design_pressure'] * 1.2
            pressures[i] = np.clip(total_pressure, min_pressure, max_pressure)

        return pressures

    def check_constraints(self, temperatures: np.ndarray, flows: np.ndarray,
                          pressures: np.ndarray) -> Dict[str, Any]:
        violations = {
            'temperature_violations': [],
            'pressure_violations': [],
            'flow_violations': [],
            'total_violations': 0
        }

        for i, zone_name in enumerate(self.zones):
            zone_params = self.zone_params[zone_name]

            if temperatures[i] < zone_params['min_supply_temp']:
                violations['temperature_violations'].append({
                    'zone': zone_name,
                    'type': 'min_temperature',
                    'value': temperatures[i],
                    'limit': zone_params['min_supply_temp'],
                    'violation': zone_params['min_supply_temp'] - temperatures[i]
                })

            if temperatures[i] > zone_params['max_operating_temp']:
                violations['temperature_violations'].append({
                    'zone': zone_name,
                    'type': 'max_temperature',
                    'value': temperatures[i],
                    'limit': zone_params['max_operating_temp'],
                    'violation': temperatures[i] - zone_params['max_operating_temp']
                })

            min_pressure = zone_params['design_pressure'] * 0.8
            max_pressure = zone_params['design_pressure'] * 1.2

            if pressures[i] < min_pressure:
                violations['pressure_violations'].append({
                    'zone': zone_name,
                    'type': 'min_pressure',
                    'value': pressures[i],
                    'limit': min_pressure,
                    'violation': min_pressure - pressures[i]
                })

            if pressures[i] > max_pressure:
                violations['pressure_violations'].append({
                    'zone': zone_name,
                    'type': 'max_pressure',
                    'value': pressures[i],
                    'limit': max_pressure,
                    'violation': pressures[i] - max_pressure
                })

            if flows[i] < self.physics_params['min_flow_rate']:
                violations['flow_violations'].append({
                    'zone': zone_name,
                    'type': 'min_flow',
                    'value': flows[i],
                    'limit': self.physics_params['min_flow_rate'],
                    'violation': self.physics_params['min_flow_rate'] - flows[i]
                })

            if flows[i] > self.physics_params['max_flow_rate']:
                violations['flow_violations'].append({
                    'zone': zone_name,
                    'type': 'max_flow',
                    'value': flows[i],
                    'limit': self.physics_params['max_flow_rate'],
                    'violation': flows[i] - self.physics_params['max_flow_rate']
                })

        violations['total_violations'] = (
                len(violations['temperature_violations']) +
                len(violations['pressure_violations']) +
                len(violations['flow_violations'])
        )

        return violations

    def calculate_demand_satisfaction(self, heat_production: np.ndarray,
                                      heat_demand: np.ndarray) -> float:
        total_production = np.sum(heat_production)
        total_demand = np.sum(heat_demand)

        if total_demand <= 0:
            return 1.0

        satisfaction = min(total_production / total_demand, 1.0)
        return satisfaction

    def calculate_efficiency(self, heat_production: np.ndarray,
                             heat_demand: np.ndarray) -> float:
        total_production = np.sum(heat_production)
        total_demand = np.sum(heat_demand)

        if total_production <= 0:
            return 0.0

        efficiency = min(total_production, total_demand) / total_production
        return efficiency

    def reset_state(self):
        self.zone_temperatures = self._initialize_zone_temperatures()
        self.zone_flows = np.zeros(self.n_zones)
        self.zone_pressures = self._initialize_zone_pressures()

    def get_current_state(self) -> Dict[str, np.ndarray]:
        return {
            'temperatures': self.zone_temperatures.copy(),
            'flows': self.zone_flows.copy(),
            'pressures': self.zone_pressures.copy(),
            'thermal_capacities': self.thermal_capacities.copy(),
            'heat_loss_coefficients': self.heat_loss_coefficients.copy()
        }

    def set_state(self, temperatures: np.ndarray, flows: np.ndarray, pressures: np.ndarray):
        self.zone_temperatures = np.array(temperatures)
        self.zone_flows = np.array(flows)
        self.zone_pressures = np.array(pressures)

def test_physics_model():
    print("Testing District Heating Physics Model...")
    print("=" * 50)

    try:
        physics = DistrictHeatingPhysics()

        print(f"Physics model created successfully!")
        print(f"Zones: {physics.n_zones}")
        print(f"Initial temperatures: {physics.zone_temperatures[:3]}")

        outdoor_temp = 5.0
        for i in range(3):
            expected_temp = physics.calculate_expected_supply_temperature(i, outdoor_temp)
            zone_name = physics.zones[i]
            print(f"Zone {zone_name}: Expected supply temp at {outdoor_temp}°C = {expected_temp:.1f}°C")

        hour = 14
        zone_demands = physics.calculate_heat_demand(outdoor_temp, hour)
        total_demand = np.sum(zone_demands)
        print(f"\nHeat demand at {outdoor_temp}°C, hour {hour}: {total_demand:.1f} MW total")
        print(f"Zone demands (first 3): {zone_demands[:3]}")

        heat_production = np.array([8.0, 5.0, 12.0, 7.0, 4.0, 3.0, 9.0, 5.0, 3.0, 8.0, 7.0])
        demand = zone_demands

        print(f"\nTesting thermal dynamics update...")
        print(f"Heat production: {np.sum(heat_production):.1f} MW")
        print(f"Heat demand: {np.sum(demand):.1f} MW")

        results = physics.update_thermal_dynamics(heat_production, outdoor_temp, demand)

        print(f"Updated temperatures (first 3): {results['temperatures'][:3]}")
        print(f"Flows (first 3): {results['flows'][:3]}")
        print(f"Pressures (first 3): {results['pressures'][:3]}")

        violations = physics.check_constraints(
            results['temperatures'],
            results['flows'],
            results['pressures']
        )

        print(f"\nConstraint violations: {violations['total_violations']}")
        if violations['temperature_violations']:
            print(f"Temperature violations: {len(violations['temperature_violations'])}")

        demand_satisfaction = physics.calculate_demand_satisfaction(heat_production, demand)
        efficiency = physics.calculate_efficiency(heat_production, demand)

        print(f"\nPerformance metrics:")
        print(f"Demand satisfaction: {demand_satisfaction:.2%}")
        print(f"Efficiency: {efficiency:.2%}")

        initial_state = physics.get_current_state()
        physics.reset_state()
        reset_state = physics.get_current_state()

        print(f"\nState management test:")
        print(f"Temperature difference after reset: {np.mean(np.abs(initial_state['temperatures'] - reset_state['temperatures'])):.2f}°C")

        print(f"\nPhysics model test completed successfully!")
        return True

    except Exception as e:
        print(f"Physics model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_physics_model()

    if success:
        print(f"\nDistrict Heating Physics Model ready!")
        print(f"Key capabilities:")
        print(f"  - Thermal dynamics simulation")
        print(f"  - Control curve implementation")
        print(f"  - Constraint violation detection")
        print(f"  - Performance metric calculation")
        print(f"  - State management")
    else:
        print(f"\nPhysics model test failed!")
        print(f"Check error messages and fix issues.")