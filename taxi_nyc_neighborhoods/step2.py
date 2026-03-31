import sys

#################################################################
# Run beyond this point only once for all datasets, not with parameters!
#################################################################

import pandas as pd
from datetime import datetime
import swifter
import json
from tqdm.auto import tqdm

gesamt_df = pd.DataFrame(columns = ['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])

results = []
for i in [1,2,6,7,8,9,10,11,12]:
    with (open('full_preproc_'+str(i)+'.jsonl', 'r', encoding='utf-8') as f):
        for line in tqdm(f):
            df = pd.read_json(line).T
            df.columns = ['datetime', 'neighborhood']
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            df['type_event'] = df['neighborhood'] #.apply(lambda x: nbhddict[x])
            first_time = df['datetime'][0]
            df['time_since_start'] = (df['datetime'] - first_time).dt.total_seconds()/3600 # convert to hours
            df['time_since_last_event'] = (df['datetime'].diff().dt.total_seconds().fillna(0)) / 3600
            results.append([df['datetime'].to_list(), len(df), df['time_since_start'].to_list(), df['time_since_last_event'].to_list(), df['type_event'].to_list()])

    gesamt_df = pd.DataFrame(results, columns = ['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])
    gesamt_df.to_parquet('taxis'+str(i)+'.parquet')

df = pd.concat([pd.read_parquet('taxis'+str(i)+'.parquet') for i in [1,2,6,7,8,9,10,11,12]], ignore_index=True)

df['dim_process'] = 306

# run previous part with 3,4,5 for extra, then use this to export:
#df['seq_idx'] = range(len(df))
#df.to_parquet('taxis_extra.parquet', index=False)

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

