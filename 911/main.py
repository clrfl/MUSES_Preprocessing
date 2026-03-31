import pandas as pd
from datetime import timedelta

event_types = {'EMS': 0, 'Fire': 1, 'Traffic': 2}
df = pd.read_csv("911.csv")

df['type_event'] = df['title'].apply(lambda x: event_types[x.split(':')[0]])
df['datetime'] = pd.to_datetime(df['timeStamp'])
df = df[['datetime', 'type_event']]
df = df.sort_values(by="datetime")

df['group_id'] = df['datetime'].dt.date

results = []
for _, group in df.groupby('group_id'):
    first_time = group['datetime'].iloc[0]
    group['time_since_start'] = (group['datetime'] - first_time).dt.total_seconds() / 3600  # convert to hours
    group['time_since_last_event'] = (group['datetime'].diff().dt.total_seconds().fillna(0)) / 3600
    results.append([group['datetime'].to_list(), len(group), group['time_since_start'].to_list(), group['time_since_last_event'].to_list(), group['type_event'].to_list()])

gesamt_df = pd.DataFrame(results, columns=['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])
gesamt_df['dim_process'] = 3

df = gesamt_df

total_len = len(df)
print(total_len)

train_end = int(0.8 * total_len)
val_end = int(0.9 * total_len)

train = df.iloc[:train_end].copy()
train['seq_idx'] = range(len(train))
train.to_parquet('train.parquet', index=False)

val = df.iloc[train_end:val_end].copy()
val['seq_idx'] = range(len(val))

test = df.iloc[val_end:].copy()
test['seq_idx'] = range(len(test))

val.to_parquet('validation.parquet', index=False)
test.to_parquet('test.parquet', index=False)
print(len(val), len(test))

