import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_weather_data(filepath):
    try:
        df = pd.read_csv(filepath)

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])

        print(f"Successfully loaded weather data with {len(df)} records")
        return df
    except Exception as e:
        print(f"Error loading weather data: {str(e)}")
        return None

def calculate_expected_supply_temp(outdoor_temp, zone_params):
    min_supply = zone_params['min_supply_temp']
    max_supply = zone_params['max_supply_temp']
    breakpoint_temp = zone_params.get('temp_breakpoint')

    if breakpoint_temp is None:
        return max_supply

    if outdoor_temp >= breakpoint_temp:
        return min_supply
    else:
        min_outdoor = -14.0
        slope = (max_supply - min_supply) / (min_outdoor - breakpoint_temp)
        return min_supply + slope * (outdoor_temp - breakpoint_temp)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_dataframe(df, filepath, index=False):
    try:
        directory = os.path.dirname(filepath)
        if directory:
            create_directory(directory)

        df.to_csv(filepath, index=index)
        print(f"Successfully saved data to: {filepath}")
    except Exception as e:
        print(f"Error saving DataFrame: {str(e)}")

def plot_and_save(plt, filepath, dpi=300):
    try:
        directory = os.path.dirname(filepath)
        if directory:
            create_directory(directory)

        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {filepath}")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")