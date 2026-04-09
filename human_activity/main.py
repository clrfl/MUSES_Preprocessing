import pandas as pd

with open("./casas/new_labeled_data/aruba.txt", 'r', encoding="utf-8") as f:
    lines = f.readlines()

    all_markers = ['Meal_Preparation', 'Leave_Home', 'Enter_Home', 'Work', 'Respirate', 'Relax', 'Sleeping', 'Housekeeping', 'Eating', 'Bed_to_Toilet', 'Wash_Dishes']
    markerdict = {'Meal_Preparation': 0, 'Leave_Home': 1, 'Enter_Home': 2, 'Work': 3, 'Respirate': 4, 'Relax': 5, 'Sleeping': 6, 'Housekeeping': 7, 'Eating': 8, 'Bed_to_Toilet': 9, 'Wash_Dishes': 10}

    datetimes = []
    days = set()
    markers = []

    results = []


    for line in lines:
        line = [w for w in line.strip().replace('\t',' ').split(' ') if w]
        if len(line) == 4 or line[-1] == 'end':
            continue

        if line[0] not in days and len(datetimes) != 0:

            df = pd.DataFrame({'datetime':datetimes, 'type_event':markers})

            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
            first_time = df['datetime'][0]
            df['time_since_start'] = (df['datetime'] - first_time).dt.total_seconds() / 3600  # convert to hours
            df['time_since_last_event'] = (df['datetime'].diff().dt.total_seconds().fillna(0)) / 3600
            df['seq_len'] = len(df)
            results.append([df['datetime'].to_list(), len(df), df['time_since_start'].to_list(), df['time_since_last_event'].to_list(), df['type_event'].to_list()])

            datetimes = []
            markers = []

        days.add(line[0])

        datetimes.append(line[0] + ' ' + line[1])
        markers.append(markerdict[line[-2]])

df = pd.DataFrame(results, columns = ['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])
df['dim_process'] = len(markerdict)

print(len(df))
train = df.sample(frac=0.8, random_state=42)
train['seq_idx'] = range(len(train))
train.to_parquet('train.parquet', index=False)
remain = df.drop(train.index)
print(len(train), len(remain))
val = remain.sample(frac=0.5, random_state=42)
val['seq_idx'] = range(len(val))
test = remain.drop(val.index)
test['seq_idx'] = range(len(test))
val.to_parquet('validation.parquet', index=False)
test.to_parquet('test.parquet', index=False)
print(len(val), len(test))


