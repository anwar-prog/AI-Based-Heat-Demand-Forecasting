import pandas as pd
import os

def create_district_heating_parameters():

    zone_parameters = {
        'B1_B2': {
            'max_operating_temp': 130,
            'max_supply_temp': 120,
            'min_supply_temp': 70,
            'max_return_temp': 50,
            'temp_breakpoint': 8,
            'design_pressure': 10,
            'max_diff_pressure': 4.0,
            'elevation': 206,
            'last_updated': '01.05.2021',
            'control_strategy': 'Bi-linear'
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
            'last_updated': '01.05.2021',
            'control_strategy': 'Bi-linear'
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
            'last_updated': '01.10.2023',
            'control_strategy': 'Bi-linear'
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
            'last_updated': '06.02.2025',
            'control_strategy': 'Constant'
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
            'last_updated': '01.10.2023',
            'control_strategy': 'Bi-linear'
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
            'last_updated': '01.10.2023',
            'control_strategy': 'Bi-linear'
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
            'last_updated': '01.05.2021',
            'control_strategy': 'Bi-linear'
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
            'last_updated': '01.10.2023',
            'control_strategy': 'Bi-linear'
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
            'last_updated': '01.05.2021',
            'control_strategy': 'Bi-linear'
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
            'last_updated': '01.05.2021',
            'control_strategy': 'Bi-linear'
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
            'last_updated': '01.05.2021',
            'control_strategy': 'Bi-linear'
        }
    }

    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame.from_dict(zone_parameters, orient='index')
    df['zone'] = df.index
    df = df.reset_index(drop=True)

    output_file = f"{output_dir}/district_heating_parameters.csv"
    df.to_csv(output_file, index=False)
    print(f"District heating parameters saved to: {output_file}")

    return df, zone_parameters

if __name__ == "__main__":

    df, params = create_district_heating_parameters()
    print(f"Created parameters for {len(params)} heating zones")
    print("Sample data:")
    print(df.head())