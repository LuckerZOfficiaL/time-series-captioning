import os
from helpers import (
    generate_line_plot
)

DATASET_NAMES = ["air quality", "border crossing", "crime", "demography", "heart rate"]   
TS_PATH = "/home/ubuntu/thesis/data/samples/time series"
PLOT_HEIGHT = None
PLOT_WIDTH = None

TITLE_MAP = {
        "air quality": "value across time",
        "border crossing": "crossing number across time",
        "crime": "crime count across time",
        "demography": "values across time",
        "heart rate": "values at different measurements"
    }
X_LABEL_MAP = {
        "air quality": "hourly timestep",
        "border crossing": "montly timestep",
        "crime": "daily timestep",
        "demography": "yearly timestep",
        "heart rate": "measurement"
    }
Y_LABEL_MAP = {
        "air quality": "value",
        "border crossing": "crossing number",
        "crime": "crime count",
        "demography": "value",
        "heart rate": "value"
    }


def main(dataset_names):
    for dataset_name in dataset_names:
        print("\nGenerating plots for", dataset_name)
        for filename in os.listdir(TS_PATH):
            if dataset_name in dataset_names:
                filepath = os.path.join(TS_PATH, filename)
                with open(filepath, 'r') as file:
                    ts = [float(line.strip()) for line in file if line.strip()]

                savepath = f"/home/ubuntu/thesis/data/samples/plots/{filename[:-5]}.jpeg" 
                generate_line_plot(ts=ts, 
                                    xlabel=X_LABEL_MAP[dataset_name],
                                    ylabel=Y_LABEL_MAP[dataset_name],
                                    title=TITLE_MAP[dataset_name],
                                    savepath=savepath,
                                    height=PLOT_HEIGHT,
                                    width=PLOT_WIDTH
                                    )
                    


if __name__ == "__main__":
    main(DATASET_NAMES)