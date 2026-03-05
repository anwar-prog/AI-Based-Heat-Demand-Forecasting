import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import calculate_expected_supply_temp, plot_and_save, create_directory

def generate_expected_temperature_data(weather_data, zone_parameters):

    print("Generating expected temperature data for each zone...")

    zone_temp_data = {}

    zone_temp_data['datetime'] = weather_data['datetime']

    for zone_name, params in zone_parameters.items():
        print(f"Processing zone: {zone_name}")

        zone_temp_data[f'{zone_name}_expected_supply_temp'] = weather_data['temp'].apply(
            lambda x: calculate_expected_supply_temp(x, params)
        )

        zone_temp_data[f'{zone_name}_expected_return_temp'] = 50.0

        zone_temp_data[f'{zone_name}_expected_delta_T'] = (
                zone_temp_data[f'{zone_name}_expected_supply_temp'] -
                zone_temp_data[f'{zone_name}_expected_return_temp']
        )

    expected_temp_df = pd.DataFrame(zone_temp_data)

    return expected_temp_df

def plot_temperature_curves(zone_parameters, output_dir="plots/temperature_curves"):

    print("Plotting temperature curves for each zone...")

    create_directory(output_dir)

    outdoor_temps = np.linspace(-14, 20, 100)

    for zone_name, params in zone_parameters.items():
        print(f"Creating temperature curve plot for zone: {zone_name}")

        supply_temps = [calculate_expected_supply_temp(t, params) for t in outdoor_temps]

        plt.figure(figsize=(10, 6))
        plt.plot(outdoor_temps, supply_temps, 'r-', linewidth=2, label="Supply Temperature")

        return_temps = [params['max_return_temp']] * len(outdoor_temps)
        plt.plot(outdoor_temps, return_temps, 'b--', linewidth=2, label="Return Temperature")

        if params.get('temp_breakpoint') is not None:
            breakpoint_temp = params['temp_breakpoint']
            breakpoint_supply = calculate_expected_supply_temp(breakpoint_temp, params)
            plt.axvline(x=breakpoint_temp, color='gray', linestyle='--', alpha=0.7)
            plt.scatter([breakpoint_temp], [breakpoint_supply], color='black', s=50)
            plt.text(breakpoint_temp, breakpoint_supply+2, f"Breakpoint: {breakpoint_temp}°C",
                     horizontalalignment='center')

        plt.xlabel('Outdoor Temperature (°C)')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Temperature Curve for Zone {zone_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plot_filepath = f"{output_dir}/{zone_name}_temperature_curve.png"
        plot_and_save(plt, plot_filepath)

if __name__ == "__main__":
    from district_heating_parameters import create_district_heating_parameters

    _, zone_parameters = create_district_heating_parameters()

    plot_temperature_curves(zone_parameters)