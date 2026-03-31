import pandas as pd
import numpy as np

paths = [
    '../memetrack',
    '../stackoverflow',
    '../taxi_nyc_neighborhoods',
    '../synthea',
    '../spiketrains',
    '../crypto_transactions',
    '../human_activity',
    '../911',
    '../mooc',
    '../wikipedia',
    '../hawkes_1',
    '../hawkes_dependent'
]

files = [
    '../taxi_nyc_neighborhoods/extra.parquet'
]

for path in paths:
    files.append(path+'/validation.parquet')
    files.append(path+'/test.parquet')
    files.append(path+'/train.parquet')

for file in files:
    print("Asserting " + file)

    assert file.split('/')[-1].split('.')[0] in ['test', 'train', 'validation', 'extra'], "Filename incorrect."
    df = pd.read_parquet(file)
    print("Contains", len(df), 'TS // Längste Zeitreihe', df["seq_len"].max(), '// dim process', df['dim_process'][0])
    assert (set(df.columns) == {'seq_len', 'time_since_start', 'time_since_last_event', 'type_event', 'seq_idx', 'dim_process', 'datetime'} or
            set(df.columns) == {'seq_len', 'time_since_start', 'time_since_last_event', 'type_event', 'seq_idx', 'dim_process'}), \
            "Columns incorrect: " + str(df.columns)

    assert type(df['seq_len'][0]) == np.int64, "seq_len wrong type: " + str(type(df['seq_len'][0]))
    assert type(df['time_since_start'][0][0]) == np.float64, "time_since_start wrong type: " + str(type(df['time_since_start'][0][0]))
    assert type(df['time_since_last_event'][0][0]) == np.float64, "time_since_last_event wrong type: " + str(type(df['time_since_last_event'][0][0]))
    assert type(df['type_event'][0][0]) == np.int64, "type_event wrong type: " + str(type(df['type_event'][0][0]))
    assert type(df['seq_idx'][0]) == np.int64, "seq_idx wrong type: " + str(type(df['seq_idx'][0]))
    assert type(df['dim_process'][0]) == np.int64, "dim_process wrong type: " + str(type(df['dim_process'][0]))
    if 'datetime' in df.columns:
        assert type(df['datetime'][0][0]) == np.datetime64, "datetime wrong type: " + str(type(df['datetime'][0][0]))
