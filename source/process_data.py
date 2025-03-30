# Run this script to generate to preprocess the raw datasets into json dictionaries.

import json
import os
import pandas as pd
import numpy as np

DATASETS_PATH = '/home/ubuntu/thesis/data/datasets'

def preprocess_heart_rate():
    hr_path = DATASETS_PATH + '/Heart Rate Oscillations during Meditation'
    char_to_group = {'C': 'chi', 'Y': 'yoga', 'N': 'normal', 'M': 'metron', 'I': 'ironman'}
    hr_data = {}

    for group in os.listdir(hr_path):
        if os.path.isdir(os.path.join(hr_path, group)): # if is a folder
            for filename in os.listdir(os.path.join(hr_path, group)):
                stuff = filename.split(".")

                if len(stuff) == 3:
                    id = stuff[0] + "." + stuff[1]
                else:
                    id = stuff[0]

                hr_data[id] = {}
                hr_data[id]["data"] = pd.read_csv(os.path.join(hr_path, group, filename), sep='\s+', header=None, names=["time (s)", "heart rate"])
                hr_data[id]["metadata"] = {}
                hr_data[id]["metadata"]["group"] = char_to_group[filename[0]]
                hr_data[id]["metadata"]["series length"] = len(hr_data[id]['data'])
                hr_data[id]["metadata"]["mean"] = hr_data[id]['data']["heart rate"].mean()
                hr_data[id]["metadata"]["min"] = hr_data[id]['data']["heart rate"].min()
                hr_data[id]["metadata"]["max"] = hr_data[id]['data']["heart rate"].max()
                hr_data[id]["metadata"]["std"] = hr_data[id]['data']["heart rate"].std()
                hr_data[id]["data"] = hr_data[id]["data"].to_dict(orient='list') # convert pandas df to dict
                #print(f"File: {filename}, Length: {len(hr_data[id]['data'])}")

    # Save the dictionary as JSON
    with open('/home/ubuntu/thesis/data/processed/hr_data.json', 'w') as file:
        json.dump(hr_data, file, indent=4, sort_keys=True)

def preprocess_air_quality():
    info_df = pd.read_csv(DATASETS_PATH + '/Time Series Air Quality Data of India/stations_info.csv')
    info_df.rename(columns={'file_name': 'station id'}, inplace=True)

    def df_to_dict(df):
        result = {}
        for col in df.columns:
            result[col] = df[col].tolist()
        return result

    aq_path = DATASETS_PATH + '/Time Series Air Quality Data of India'
    aq_data = {}

    count = 0

    for filename in os.listdir(aq_path):
        if filename == "stations_info.csv": continue
        if filename == "DL017.csv": continue # a file that causes problems
        if filename == "HR016.csv": continue # a file that causes problems

        #print(filename, "in progress...")
        id = filename.split(".")[0]
        metadata_dict = info_df[info_df["station id"] == id].to_dict('records')[0]

        df = pd.read_csv(os.path.join(aq_path, filename))
        numeric_data_dict = df_to_dict(df)
        aq_data[id] = {}
        aq_data[id]["metadata"] = metadata_dict
        aq_data[id].update(numeric_data_dict)
        aq_data[id]["metadata"]["series length"] = len(df)
        aq_data[id]["metadata"]["starting time"] = df["From Date"].tolist()
        aq_data[id]["metadata"]["end time"] = df.tail(1)["To Date"].values[0]
        del aq_data[id]['From Date']
        del aq_data[id]['To Date']

        aq_data[id]["metadata"]["mean"] = {}
        aq_data[id]["metadata"]["min"] = {}
        aq_data[id]["metadata"]["max"] = {}
        aq_data[id]["metadata"]["std"] = {}

        for col in numeric_data_dict:
            if col != "From Date" and col != "To Date":
                aq_data[id]["metadata"]["mean"][col] = df[col].mean()
                aq_data[id]["metadata"]["min"][col] = df[col].min()
                aq_data[id]["metadata"]["max"][col] = df[col].max()
                aq_data[id]["metadata"]["std"][col] = df[col].std()
                aq_data[id]["metadata"]['nans'] = int(df[col].isna().sum())

    aq_data[id]["metadata"]['sampling frequency'] = "hourly"
 
    with open('/home/ubuntu/thesis/data/processed/aq.json', 'w') as file:
        json.dump(aq_data, file, indent=4, sort_keys=True)

def preprocess_demographics():
    info_df = pd.read_csv(DATASETS_PATH + '/Population Collapse Time Series Data of the World/country_groups_all.csv')
    info_df.head(10)

    def df_to_dict(df):
        result = {}

        for _, row in df.iterrows():
            country_id = row.iloc[0]
            country_name = row.iloc[1]
            group = row.iloc[2]
            group_type = row.iloc[3]
            capital = row.iloc[5]
            longitude = row.iloc[6]
            latitude = row.iloc[7]

            if country_id not in result:
                result[country_id] = {}

            result[country_id]["country name"] = country_name
            result[country_id]["capital city"] = capital
            result[country_id]["longitude"] = longitude
            result[country_id]["latitude"] = latitude

            if group_type not in result[country_id]:
                result[country_id][group_type] = []
            if group != "World": # exclude this group because it's the same for all countries
                result[country_id][group_type].append(group)

        return result
    country_info_dict = df_to_dict(info_df)

    birth_rate_path = DATASETS_PATH + "/Population Collapse Time Series Data of the World/total_birth_rate.csv"
    death_rate_path = DATASETS_PATH + "/Population Collapse Time Series Data of the World/total_death_rate.csv"
    median_age_path = DATASETS_PATH + "/Population Collapse Time Series Data of the World/total_median_age.csv"
    # there are more datasets beyond these

    birth_rate_df = pd.read_csv(birth_rate_path)
    death_rate_df = pd.read_csv(death_rate_path)
    median_age_df = pd.read_csv(median_age_path)

    def df_rows_to_ts(df, attribute, demo_dict): # iterates through df rows and adds the ts to the dict, matching the country IDs
        for _, row in df.iterrows():
            country_id = row.iloc[0]
            if country_id in demo_dict:
                demo_dict[country_id][attribute] = row.iloc[1:].tolist()

    demo_dict = {}
    for country_id in country_info_dict:
        demo_dict[country_id] = {}
        demo_dict[country_id]["metadata"] = country_info_dict[country_id].copy()
        df_rows_to_ts(birth_rate_df, attribute="birth rate", demo_dict=demo_dict)
        df_rows_to_ts(death_rate_df, attribute="death rate", demo_dict=demo_dict)
        df_rows_to_ts(median_age_df, attribute="median age", demo_dict=demo_dict)


        demo_dict[country_id]["metadata"]['series length'] = len(demo_dict[country_id]["birth rate"])
        demo_dict[country_id]["metadata"]['start year of the series'] = 2000 # all ts start from year 2000

        demo_dict[country_id]["metadata"]['mean'] = {}
        demo_dict[country_id]["metadata"]['min'] = {}
        demo_dict[country_id]["metadata"]['max'] = {}
        demo_dict[country_id]["metadata"]['std'] = {}

        for attr in demo_dict[country_id]:
            if attr != "metadata":
                demo_dict[country_id]["metadata"]['mean'][attr] = np.mean(demo_dict[country_id][attr])
                demo_dict[country_id]["metadata"]['min'][attr] = np.min(demo_dict[country_id][attr])
                demo_dict[country_id]["metadata"]['max'][attr] = np.max(demo_dict[country_id][attr])
                demo_dict[country_id]["metadata"]['std'][attr] = np.std(demo_dict[country_id][attr])

        demo_dict[country_id]["metadata"]['sampling frequency'] = "yearly"

    with open('/home/ubuntu/thesis/data/processed/demographics.json', 'w') as file:
        json.dump(demo_dict, file, indent=4, sort_keys=True)

def preprocess_crime():
    df = pd.read_csv(DATASETS_PATH + '/US Gov/Crime_Data_from_2020_to_Present.csv')
    grouped_df = df.groupby(['AREA NAME', 'DATE OCC']).size().reset_index(name='counts')
    grouped_df = grouped_df.sort_values(by='DATE OCC')

    crime_dict = {}
    towns = grouped_df['AREA NAME'].unique()

    for town in towns:
        crime_dict[town] = {}
        town_df = grouped_df[grouped_df['AREA NAME'] == town]
        town_df['DATE OCC'] = pd.to_datetime(town_df['DATE OCC'])
        town_df = town_df.sort_values(by="DATE OCC").reset_index(drop=True)

        crime_dict[town]['data'] = town_df['counts'].to_list()
        crime_dict[town]['metadata'] = {'town': town}
        crime_dict[town]['metadata']['series length'] = len(town_df)
        crime_dict[town]['metadata']['start date'] = str(town_df['DATE OCC'].min())
        crime_dict[town]['metadata']['end date'] = str(town_df['DATE OCC'].max())
        crime_dict[town]['metadata']['mean'] = float(town_df['counts'].mean())
        crime_dict[town]['metadata']['std'] = float(town_df['counts'].std())
        crime_dict[town]['metadata']['min'] = int(town_df['counts'].min())
        crime_dict[town]['metadata']['max'] = int(town_df['counts'].max())
        
        crime_dict[town]['metadata']['sampling frequency'] = "daily"

    with open('/home/ubuntu/thesis/data/processed/crime.json', 'w') as file:
        json.dump(crime_dict, file, indent=4, sort_keys=True)

def preprocess_border_crossing():
    df = pd.read_csv(DATASETS_PATH + '/US Gov/Border_Crossing_Entry_Data.csv')
    port_names = list(df['Port Name'].unique())
    port_names.remove("Algonac")
    port_names.remove("Cross Border Xpress")
    measures = df['Measure'].unique()

    crossing_data = {}

    for port_name in port_names:
        port_df = df[df['Port Name'] == port_name]
        crossing_data[port_name] = {}
        crossing_data[port_name]['data'] = {}
        for measure in measures:
            measure_df = port_df[port_df['Measure'] == measure].sort_values(by="Date")
            measure_df['Date'] = pd.to_datetime(measure_df['Date'])
            measure_df = measure_df.sort_values(by="Date").reset_index(drop=True)
            crossing_data[port_name]['data'][measure] = measure_df['Value'].to_list()

        crossing_data[port_name]['metadata'] = {'port name': port_name}
        crossing_data[port_name]['metadata']['series length'] = measure_df.size
        crossing_data[port_name]['metadata']['state'] = measure_df['State'].unique()[0]
        crossing_data[port_name]['metadata']['border'] = measure_df['Border'].unique()[0]
        crossing_data[port_name]['metadata']['latitude'] = measure_df['Latitude'].unique()[0]
        crossing_data[port_name]['metadata']['longitude'] = measure_df['Longitude'].unique()[0]
        crossing_data[port_name]['metadata']['start date'] = str(measure_df['Date'].min())
        crossing_data[port_name]['metadata']['end date'] = str(measure_df['Date'].max())
        crossing_data[port_name]['metadata']['mean'] = {}
        crossing_data[port_name]['metadata']['min'] = {}
        crossing_data[port_name]['metadata']['max'] = {}
        crossing_data[port_name]['metadata']['std'] = {}

        for measure in measures:
            crossing_data[port_name]['metadata']['mean'][measure] = np.mean(crossing_data[port_name]['data'][measure])
            crossing_data[port_name]['metadata']['std'][measure] = np.std(crossing_data[port_name]['data'][measure])
            try:
                crossing_data[port_name]['metadata']['min'][measure] = int(np.min(crossing_data[port_name]['data'][measure]))
                crossing_data[port_name]['metadata']['max'][measure] = int(np.max(crossing_data[port_name]['data'][measure]))
            except:
                print("Error in generating min and max for border crossing", port_name, measure)
                pass
    crossing_data[port_name]['metadata']['sampling frequency'] = "monthly"

    with open('/home/ubuntu/thesis/data/processed/border_crossing.json', 'w') as file:
        json.dump(crossing_data, file, indent=4, sort_keys=True)

def preprocess_road_injuries():
    file_path = DATASETS_PATH + '/US Gov/road-traffic-injuries-2002-2010.csv'
    df = pd.read_csv(file_path)
    df = df[['reportyear', 'geotype', 'geoname', 'mode', 'severity', 'injuries', 'totalpop']]
    

    

    df = df[df['reportyear'].astype(str).str.len() == 4] # remove multi-year intervals and retain only single-year values
    geotype_mapping = { # Convert geotype using the mapping
        'CT': 'Census tract',
        'PL': 'Place',
        'CO': 'County',
        'CD': 'County division',
        'R4': 'Consolidated Statistical Metropolitan Area',
        'RE': 'Region',
        'CA': 'State'
    }
    df['geotype'] = df['geotype'].map(geotype_mapping)
    
    #print(df.head(2))
    #print(df['geoname'].unique())

    df_collapsed = df.groupby(['geotype', 'geoname', 'mode', 'severity', 'totalpop'])['injuries'].apply(list).reset_index()

    data_dict = {}
    for index, row in df_collapsed.iterrows():
        geotype = row['geotype']
        geoname = row['geoname']
        mode = row['mode']
        severity = row['severity']
        totalpop = row['totalpop']
        injuries = [x for x in row['injuries'] if not pd.isna(x)]  # Filter out NaN values

        if not injuries:  # Skip adding empty lists
            continue

        if geoname not in data_dict:
            data_dict[geoname] = {
                'metadata': {
                    'totalpop': totalpop,
                    'geotype': geotype,
                    'start year of the series': 2002,
                    'end year of the series': 2010
                },
                'data': {}
            }
        
        if mode not in data_dict[geoname]['data']:
            data_dict[geoname]['data'][mode] = {}
        
        if severity not in data_dict[geoname]['data'][mode]:
            data_dict[geoname]['data'][mode][severity] = injuries
        else:
            data_dict[geoname]['data'][mode][severity].extend(injuries)

    for geoname in data_dict:
        data_dict[geoname]['metadata']['mean'] = {}
        data_dict[geoname]['metadata']['standard deviation'] = {}
        data_dict[geoname]['metadata']['min'] = {}
        data_dict[geoname]['metadata']['max'] = {}
        
        for mode in data_dict[geoname]['data']:
            data_dict[geoname]['metadata']['mean'][mode] = {}
            data_dict[geoname]['metadata']['standard deviation'][mode] = {}
            data_dict[geoname]['metadata']['min'][mode] = {}
            data_dict[geoname]['metadata']['max'][mode] = {}
            
            for severity in data_dict[geoname]['data'][mode]:
                injuries = data_dict[geoname]['data'][mode][severity]
                if injuries:  # Ensure there are valid values
                    data_dict[geoname]['metadata']['mean'][mode][severity] = float(round(np.mean(injuries), 2))
                    data_dict[geoname]['metadata']['standard deviation'][mode][severity] = float(round(np.std(injuries), 2))
                    data_dict[geoname]['metadata']['min'][mode][severity] = float(round(np.min(injuries), 2))
                    data_dict[geoname]['metadata']['max'][mode][severity] = float(round(np.max(injuries), 2))

    with open('/home/ubuntu/thesis/data/processed/road_injuries.json', 'w') as file:
        json.dump(data_dict, file, indent=4, sort_keys=True)


def main():
    #preprocess_heart_rate()
    #preprocess_air_quality()
    #preprocess_demographics()
    #preprocess_crime()
    #preprocess_border_crossing()
    #preprocess_road_injuries()

    

if __name__ == "__main__":
    main()
