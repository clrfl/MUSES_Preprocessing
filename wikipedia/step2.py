import pandas as pd

df = pd.read_csv('wikipedia_out.csv')
users = df['userid'].value_counts().reset_index()
users = users.iloc[:50]
top50users = users['userid'].to_list()
df = df[df['userid'].isin(top50users)]
tslength = df['articleid'].value_counts().reset_index()
tslength = tslength[tslength['count'] < 2]
tooshort = tslength['articleid'].to_list()
df = df[~df['articleid'].isin(tooshort)]

df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%SZ')
userdict = {}
for idx, usr in enumerate(top50users):
    userdict[usr] = idx

df['type_event'] = df['userid'].apply(lambda x: userdict[x])

results = []
for _, group in df.groupby('articleid'):
    first_time = group['datetime'].iloc[0]
    group['time_since_start'] = (group['datetime'] - first_time).dt.total_seconds() / 3600  # convert to hours
    group['time_since_last_event'] = (group['datetime'].diff().dt.total_seconds().fillna(0)) / 3600
    results.append([group['datetime'].to_list(), len(group), group['time_since_start'].to_list(), group['time_since_last_event'].to_list(), group['type_event'].to_list()])

gesamt_df = pd.DataFrame(results, columns=['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])
gesamt_df['dim_process'] = 50

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

# {433328: 0, 1215485: 1, 527862: 2, 1324179: 3, 271058: 4, 1392310: 5, 75561: 6, 1079367: 7, 228773: 8, 279219: 9, 82835: 10, 1574574: 11, 4005189: 12, 4928500: 13, 6120: 14, 24902: 15, 621721: 16, 2198658: 17, 12978: 18, 112114: 19, 15708: 20, 44020: 21, 4477315: 22, 4626: 23, 1951353: 24, 1575512: 25, 1979668: 26, 25667: 27, 603177: 28, 7580: 29, 28438: 30, 293907: 31, 3808: 32, 1022601: 33, 754619: 34, 1616157: 35, 203434: 36, 66: 37, 97951: 38, 1591: 39, 564742: 40, 91310: 41, 294180: 42, 3813685: 43, 104523: 44, 5478189: 45, 117878: 46, 51235: 47, 156441: 48, 308437: 49}