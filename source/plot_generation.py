import os
from helpers import (
    generate_line_plot,
    load_config
)

"""DATASET_NAMES = ["air quality", "border crossing", "crime", "demography"]#, "heart rate"]   
TS_PATH = "/home/ubuntu/thesis/data/samples/time series"
PLOT_HEIGHT = None
PLOT_WIDTH = None"""


TITLE_MAP = {
        "air quality": "value across hours",
        "border crossing": "crossing number across days",
        "crime": "crime count across days",
        "demography": "values across years",
        "heart rate": "values at different measurements",
        "road injuries": "values across years"
    }
X_LABEL_MAP = {
        "air quality": "hourly timestep",
        "border crossing": "montly timestep",
        "crime": "daily timestep",
        "demography": "yearly timestep",
        "heart rate": "measurement",
        "road injuries": "yearly timestep"
    }
Y_LABEL_MAP = {
        "air quality": "value",
        "border crossing": "crossing number",
        "crime": "crime count",
        "demography": "value",
        "heart rate": "value",
        "road injuries": "accident count"
    }


def main():
    config = load_config()
    dataset_names = config['data']['dataset_names']
    ts_folder_path = config['path']['ts_folder_path']
    save_folder_path = config['path']['plot_folder_path']
    plot_height = config['plot']['height']
    plot_width = config['plot']['width']

    for dataset_name in dataset_names:
        print("\nGenerating plots for", dataset_name)
        for filename in os.listdir(ts_folder_path):
            if dataset_name in filename:
                filepath = os.path.join(ts_folder_path, filename)
                with open(filepath, 'r') as file:
                    #ts = [float(line.strip()) for line in file.read().split('_'*80) if line.strip()]
                    ts = [float(line.strip()) for line in file if line.strip()]

                savepath = f"{save_folder_path}/{filename[:-4]}.jpeg" 
                generate_line_plot(ts=ts, 
                                    xlabel=X_LABEL_MAP[dataset_name],
                                    ylabel=Y_LABEL_MAP[dataset_name],
                                    title=TITLE_MAP[dataset_name],
                                    savepath=savepath,
                                    height=plot_height,
                                    width=plot_width
                                    )
                    


if __name__ == "__main__":
    main()