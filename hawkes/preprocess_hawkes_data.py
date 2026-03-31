import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

synth_ds = [['/home/user/neural-tpps/difficult_data/hawkes/train.json',150],
            ['/home/user/neural-tpps/enguehard_data/hawkes/train.json',2],
            ['/home/user/neural-tpps/omi_data/hawkes/train.json',1]]

for ds in synth_ds:

    results = []
    data = json.load(open(ds[0]))

    for item in data:
        if len(item) < 2: continue
        datetimes = []
        type_events = []
        for mark in item:
            datetimes.append(mark['time'])
            type_events.append(mark['labels'][0])

        df = pd.DataFrame({'datetime': datetimes, 'type_event': type_events})
        first_time = df['datetime'][0]

        df['time_since_start'] = df['datetime'] - first_time
        df['time_since_last_event'] = (df['datetime'].diff().fillna(0))

        results.append([len(df), df['time_since_start'].to_list(), df['time_since_last_event'].to_list(), df['type_event'].to_list()])

    df = pd.DataFrame(results, columns=['seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])

    df['dim_process'] = ds[1]

    base=ds[0].split('/')[4]
    print(len(df))
    train = df.sample(frac=0.8, random_state=42)
    train['seq_idx'] = range(len(train))
    train.to_parquet(base+'-train.parquet', index=False)
    remain = df.drop(train.index)
    print(len(train), len(remain))
    val = remain.sample(frac=0.5, random_state=42)
    val['seq_idx'] = range(len(val))
    test = remain.drop(val.index)
    test['seq_idx'] = range(len(test))
    val.to_parquet(base+'-validation.parquet', index=False)
    test.to_parquet(base+'-test.parquet', index=False)
    print(len(val), len(test))




