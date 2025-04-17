from curses import meta
import os
from helpers import (
    generate_line_plot,
    load_config
)
import random
import json



TITLE_MAP = {
        "air quality": "value across hours",
        "border crossing": "crossing number across days",
        "crime": "crime count across days",
        "demography": "values across years",
        "heart rate": "values at different measurements",
        "road injuries": "values across years",
        "covid": "value count across days",
        "co2": "co2 emission in million metric tons across years",
        "diet": "average daily kilocalories consumed per person across years",
        "walmart": "weekly sales in USD across weeks",
        "online retail": "weekly value across weeks",
        "agriculture": "values across years",
        
    }
X_LABEL_MAP = {
        "air quality": "hourly timestamp",
        "border crossing": "montly timestamp",
        "crime": "daily timestamp",
        "demography": "yearly timestamp",
        "heart rate": "measurement",
        "road injuries": "yearly timestamp",
        "covid": "daily timestamp",
        "co2": "yearly timestamp",
        "diet": "yearly timestamp",
        "walmart": "weekly timestamp",
        "online retail": "weekly timestamp",
        "agriculture": "yearly timestamp",
    }
Y_LABEL_MAP = {
        "air quality": "value",
        "border crossing": "crossing number",
        "crime": "crime count",
        "demography": "value",
        "heart rate": "value",
        "road injuries": "accident count",
        "covid": "daily count",
        "co2": "co2 emission in million metric tons",
        "diet": "average daily kilocalories consumed per person",
        "walmart": "weekly sales in USD",
        "online retail": "weekly value",
        "agriculture": "yearly value",
    }


def main():
    config = load_config()
    dataset_names = config['data']['dataset_names']
    """ts_folder_path = config['path']['ts_folder_path']
    metadata_folder_path = config['path']['metadata_folder_path']
    save_folder_path = config['path']['plot_folder_path']"""
    #plot_height = config['plot']['height']
    #plot_width = config['plot']['width']
    
    ts_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/train/time series"
    metadata_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/train/metadata"
    save_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/train/plots"
    

    
    for dataset_name in dataset_names:
        print("\nGenerating plots for", dataset_name)
        for filename in os.listdir(ts_folder_path):
            if dataset_name in filename:
                filepath = os.path.join(ts_folder_path, filename)
                with open(filepath, 'r') as file:
                    #ts = [float(line.strip()) for line in file.read().split('_'*80) if line.strip()]
                    ts = [float(line.strip()) for line in file if line.strip()]
                
                metadata_path = os.path.join(metadata_folder_path, f"{filename[:-4]}.json")
                with open(metadata_path, 'r') as metadata_file:
                    metadata = json.load(metadata_file)
                      
                #################### Get start and end time####################
                start_time_keys = [key for key in metadata.keys() if "start" in key]
                end_time_keys = [key for key in metadata.keys() if "end" in key]
                          
                if len(start_time_keys) == 1:
                    start_time=metadata[start_time_keys[0]]
                else: start_time = None
                
                if len(end_time_keys) == 1:
                    end_time=metadata[end_time_keys[0]]
                else: end_time = None
                #################################################################
                
                savepath = f"{save_folder_path}/{filename[:-4]}.jpeg"
                
                
                ########################## Random Plot Configs ###############################
                plot_height = random.randint(3, 7)  # Height in inches
                plot_width = random.randint(7, 12)  # Width in inches

                # Random color
                colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'black']
                color = random.choice(colors)

                # Random line width
                linewidth = random.randint(1, 3)
                
                grid=random.choice([True, False])
                
                show_nums_on_line = random.choice([True, False])

                # Random marker
                markers = [None, ".", 'o', 's', 'D']
                marker = random.choice(markers)

                # Random line style
                linestyles = ['-']
                linestyle = random.choice(linestyles)
    
                ################################################################################
    
                generate_line_plot(ts=ts, 
                                    xlabel=X_LABEL_MAP[dataset_name],
                                    ylabel=Y_LABEL_MAP[dataset_name],
                                    title=TITLE_MAP[dataset_name],
                                    savepath=savepath,
                                    height=plot_height,
                                    width=plot_width,
                                    color=color,
                                    linewidth=linewidth,
                                    marker=marker,
                                    linestyle=linestyle,
                                    grid=grid,
                                    show_nums_on_line=show_nums_on_line,
                                    x_start=start_time,
                                    x_end=end_time
                                    )
                    


if __name__ == "__main__":
    main()