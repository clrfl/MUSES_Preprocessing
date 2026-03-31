import pandas as pd
import matplotlib.pyplot as plt
import swifter
import numpy as np

def val_to_type_event(x):
    return [y%2 for y in x]

def val_to_datetime(x):
    return [pd.to_datetime((y-(y%2))/2,unit='s') for y in x]

def transform_val(x):
    return [(y-(y%2))/2 for y in x]

def time_since_start_from_val(x):
    first_time = x[0]
    returnval = [(y-first_time)/3600 for y in x]
    return returnval

def time_since_last_event_from_val(x):
    return (pd.Series(x).diff().fillna(0) / 3600).tolist()

df = pd.read_parquet('transfers.parquet')

df['seq_len'] = df['val'].apply(len)
df = df[df['seq_len'] > 2]
df = df[df['seq_len'] < 991] # cut off longest 0.1%
df = df[~df['address'].str.contains('0000000000', na=False)]
df = df[['val', 'seq_len']]
df['val'] = df['val'].swifter.apply(np.sort)
#df.to_parquet('filtered_sorted.parquet')

df['type_event'] = df['val'].swifter.apply(val_to_type_event)
df['datetime'] = df['val'].swifter.apply(val_to_datetime)
#df.to_parquet('filtered_sorted_preprocessed.parquet')

df['val'] = df['val'].swifter.apply(transform_val)
#df.to_parquet('filtered_sorted_preprocessed_valfix.parquet')

df['time_since_start'] = df['val'].swifter.apply(time_since_start_from_val)
#df.to_parquet('filtered_sorted_preprocessed_valfix_timesincestart.parquet')

df['time_since_last_event'] = df['val'].swifter.apply(time_since_last_event_from_val)
#df.to_parquet('filtered_sorted_preprocessed_valfix_timesincestart_timesincelastevent.parquet')

df = df[['datetime', 'time_since_start', 'time_since_last_event', 'seq_len', 'type_event']]

df['dim_process'] = 2

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
