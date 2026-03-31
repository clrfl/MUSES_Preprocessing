import pandas as pd
from datetime import timedelta

df = pd.read_csv('../SNAP/MOOC student actions/mooc_actions.tsv', sep='\t')

event_ids = set()
df['TARGETID'].apply(lambda x: event_ids.add(x))

results = []
for _, group in df.groupby('USERID'):
    first_time = group['TIMESTAMP'].iloc[0]

    group['time_since_start'] = (group['TIMESTAMP'] - first_time) / 3600  # convert to hours
    group['time_since_last_event'] = (group['TIMESTAMP'].diff().fillna(0)) / 3600

    results.append([len(group), group['time_since_start'].to_list(), group['time_since_last_event'].to_list(), group['TARGETID'].to_list()])

gesamt_df = pd.DataFrame(results, columns=['seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])

gesamt_df['dim_process'] = len(event_ids)

df = gesamt_df

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



