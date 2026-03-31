from datetime import timedelta
import pandas as pd

def richter_scale(magnitude):
    # values start at 2.5
    if magnitude < 3:
        return 0 # minor
    if magnitude < 4:
        return 1 # slight
    if magnitude < 5:
        return 2 # light
    if magnitude < 6:
        return 3 # moderate
    if magnitude < 7:
        return 4 # strong
    if magnitude < 8:
        return 5 # major
    else:
        print("Something went wrong")
    # if magnitude < 9:
    #     return 6 # great
    # return 7 # extreme

df = pd.read_csv('ComCat_catalog.csv')

df['type_event'] = df['magnitude'].apply(richter_scale)
df['datetime'] = pd.to_datetime(df['time'])
df['group_id'] = (df['datetime'].diff() >= pd.Timedelta(hours=24)).cumsum()

results = []

for _, group in df.groupby('group_id'):
    if len(group) < 3:
        continue

    first_time = group['datetime'].iloc[0]
    group['time_since_start'] = (group['datetime'] - first_time).dt.total_seconds() / 3600  # convert to hours
    group['time_since_last_event'] = (group['datetime'].diff().dt.total_seconds().fillna(0)) / 3600

    results.append(
        [group['datetime'].to_list(), len(group), group['time_since_start'].to_list(), group['time_since_last_event'].to_list(),
         group['type_event'].to_list()])

df = pd.DataFrame(results, columns=['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])

df['dim_process'] = 6

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