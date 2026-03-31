# use with query.json (neighborhoods)
# and trip_data_(id).csv

#################################################################
# Run with the dataset id [1-12] as parameter
#################################################################

import pandas as pd
import json
from scipy import spatial
import swifter
import sys
from datetime import datetime

def read_neighborhoods(path):
    lons = []
    lats = []
    with open(path) as f:
        data = json.load(f)
        for a in data["features"]:
            lons.append(a["geometry"]["coordinates"][0])
            lats.append(a["geometry"]["coordinates"][1])

    names = range(len(lons))
    nbhds = pd.DataFrame({'name': names, 'lon': lons, 'lat': lats})
    return nbhds

def get_nearest_values(row, nbhds):
    nearest = nbhds[['lon', 'lat']].values[spatial.KDTree(nbhds[['lon', 'lat']].values).query([row['pickup_longitude'], row['pickup_latitude']])[1]]
    nbhd = nbhds[(nbhds['lon'] == nearest[0]) & (nbhds['lat'] == nearest[1])].iloc[0]['name']
    return nbhd

def read_dataset(path, time, long, lat, nbhds):
    df = pd.read_csv(path, engine='python')
    df.columns = df.columns.str.strip() # broken csv header workaround
    df = df[df["vendor_id"] != "VTS"]
    df = df[[time, long, lat, "medallion", "hack_license"]]

    # sanity check
    df = df[(-74.77 < df[long]) &
            (df[long] < -73.18) &
            (41.42 > df[lat]) &
            (df[lat] > 39.99)]

    df["neighborhood"] = df.swifter.apply(get_nearest_values, nbhds=nbhds, axis=1)
    df = df[[time, "neighborhood", "medallion", "hack_license"]]
    df.to_csv(path[:-4] + "_processed.csv", index=False)

def abbr(row):
    return datetime.strptime(row["pickup_datetime"], '%Y-%m-%d %H:%M:%S')

argum = sys.argv[1]
nbhds = read_neighborhoods("query.json")
read_dataset("trip_data_"+argum+".csv", "pickup_datetime", "pickup_longitude", "pickup_latitude", nbhds)

# this selects taxis by medallion and splits into ts
df = pd.read_csv("trip_data_"+str(argum)+"_processed.csv")
df["datetime"] = df.swifter.apply(abbr, axis=1)

ts_collection = []

for taxi in df["medallion"].unique():
    df_taxi = df[df["medallion"] == taxi]
    df_taxi = df_taxi.sort_values(by="pickup_datetime")

    df_taxi['group_id'] = (df_taxi['datetime'].diff() >= pd.Timedelta(hours=12)).cumsum()

    for _, group in df_taxi.groupby('group_id'):
        if len(group) > 10 and group['datetime'].is_unique:
            group_df = group[["datetime", "neighborhood"]]
            ts_collection.append(group_df.T.to_json(orient="values"))

# write jsonlines
with open("full_preproc_"+str(argum)+".jsonl", 'w', encoding='utf-8') as f:
    for s in ts_collection:
        f.write(s + '\n')

