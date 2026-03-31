import pandas as pd
from tqdm.auto import tqdm

gesamt_df = pd.DataFrame(columns = ['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])


for i in [1,2,3,4,5,6]:

    results = []
    with (open('full_preproc_'+str(i)+'.jsonl', 'r', encoding='utf-8') as f):
        for line in tqdm(f):
            df = pd.read_json(line).T
            df.columns = ['type_event', 'time_s']

            df['type_event'] = df['type_event'].astype(int)

            first_time = df['time_s'][0]

            df['time_since_start'] = (df['time_s'] - first_time) /3600 # convert to hours

            df['time_since_last_event'] = df['time_since_start'].diff().fillna(0)

            results.append([len(df), df['time_since_start'].to_list(), df['time_since_last_event'].to_list(), df['type_event'].to_list()])

    gesamt_df = pd.DataFrame(results, columns = ['seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])
    gesamt_df.to_parquet('neurons'+str(i)+'.parquet')


df = pd.concat([pd.read_parquet('neurons'+str(i)+'.parquet') for i in [1,2,3,4,5,6]], ignore_index=True)

df['dim_process'] = 9

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

