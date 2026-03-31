from datasets import load_dataset
import pandas as pd
import os

easytpp_datasets = [
    'taobao',
    'retweet',
    'volcano',
    #'earthquake',
    #'stackoverflow',
    'amazon',
    'taxi'
]


for dsname in easytpp_datasets:
    ds = load_dataset('easytpp/'+dsname)
    split = ds.keys()
    dfs = []
    for splitkey in split:
        dfs.append(ds[splitkey].to_pandas())
    df = pd.concat([x for x in dfs], ignore_index=True)
    df.drop(columns=["seq_idx"], axis=1, inplace=True)

    print(dsname, df['dim_process'][0])

    # further processing
    df = df[df['seq_len'] > 2]

    if not os.path.exists(dsname):
        os.mkdir(dsname)
    print(len(df))
    train = df.sample(frac=0.8, random_state=42)
    train['seq_idx'] = range(len(train))
    train.to_parquet(dsname+'/train.parquet', index=False)
    remain = df.drop(train.index)
    print(len(train), len(remain))
    val = remain.sample(frac=0.5, random_state=42)
    val['seq_idx'] = range(len(val))
    test = remain.drop(val.index)
    test['seq_idx'] = range(len(test))
    val.to_parquet(dsname + '/validation.parquet', index=False)
    test.to_parquet(dsname+'/test.parquet', index=False)
    print(len(val), len(test))
