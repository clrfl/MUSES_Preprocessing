from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def normalized_shannon_entropy(series):
    p = series.value_counts(normalize=True)
    if len(p) <= 1: return 0.0
    entr = -(p * np.log(p)).sum()
    norm_entr = entr / np.log(len(p))
    return norm_entr

dataset_configs = [
    'earthquake',
    'memetrack',
    'stackoverflow',
    'taxi_nyc_neighborhoods',
    'synthea',
    'spiketrains',
    'crypto_transactions',
    'human_activity',
    '911',
    'mooc',
    'amazon_easytpp',
    'wikipedia',
    'retweet_easytpp',
    'taobao_easytpp',
    'taxi_easytpp',
    'volcano_easytpp',
    'hawkes_dependent',
    'hawkes_1',
    'hawkes_difficult'
]

with open('analysisdoc.csv', 'w', encoding='utf-8') as f:
    print('dataset,parameter,value', file=f)
    for ds in dataset_configs:
        print('Running ' + ds)

        data = load_dataset("ddrg/MUSES", ds)

        df = pd.DataFrame()
        df = pd.concat(
            [data[split].to_pandas() for split in data.keys()],
            ignore_index=True
        )

        print(ds,'class_count',df['dim_process'][0], sep=',', file=f)
        print(ds,'ts_len_avg',round(df['seq_len'].mean(),2), sep=',', file=f)
        print(ds,'ts_len_min', df['seq_len'].min(), sep=',', file=f)
        print(ds,'ts_len_max', df['seq_len'].max(), sep=',', file=f)
        print(ds,'ts_len_std', round(df['seq_len'].std(),2), sep=',', file=f)
        print(ds,'event_count_total', df['seq_len'].sum(), sep=',', file=f)
        print(ds,'ts_count_total', len(df), sep=',', file=f)
        print(ds,'entropy_class_bal', round(normalized_shannon_entropy(df['type_event'].explode()),2), sep=',', file=f)

        classes = df['type_event'].explode().value_counts().sort_index().to_dict()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.75))

        ax1.bar(list(classes.keys()), list(classes.values()), align='center', bottom=1, log=True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_title('Class distribution of dataset ' + ds)
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of events (log scale)')
        if len(classes.keys()) == 1:
            single_val = 0
            ax1.set_xlim(single_val - 1, single_val + 1)

        bins = min(50,(df['seq_len'].max()-df['seq_len'].min())+1)
        ax2.hist(df['seq_len'], bins=bins, align='mid', bottom=1, log=True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_title('Time series length distribution of dataset ' + ds)
        ax2.set_xlabel('Sequence Length (Histogram)')
        ax2.set_ylabel('Number of time series (log scale)')

        plt.tight_layout()

        dateiname = f'plots/{ds}_plot.pdf'
        plt.savefig(dateiname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'.    --> Plot gespeichert unter: {dateiname}')
