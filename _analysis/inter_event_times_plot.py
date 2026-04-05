import numpy as np
from datasets import load_dataset
import pandas as pd
from matplotlib import pyplot as plt

dsnames = [
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
    'hawkes_1'
]

times = []

for dsname in dsnames:
    data = load_dataset("ddrg/MUSES", dsname)
    df = pd.concat(
        [data[split].to_pandas() for split in data.keys()],
        ignore_index=True
    )
    inter_event_times = pd.Series(df['time_since_last_event']).explode()
    inter_event_times = inter_event_times[inter_event_times != 0.0]
    avg_time = np.mean(inter_event_times.tolist())

    times.append([dsname, avg_time])

#times = eval("[['earthquake', np.float64(3.7451098307720208)], ['memetrack', np.float64(56.513071402178)], ['stackoverflow', np.float64(1159.9011034596433)], ['taxi_nyc_neighborhoods', np.float64(0.5474640387377626)], ['synthea', np.float64(7381.070420134902)], ['spiketrains', np.float64(0.0007087594440836774)], ['crypto_transactions', np.float64(108.99860718649897)], ['human_activity', np.float64(0.7088379688163456)], ['911', np.float64(0.06285007971777994)], ['mooc', np.float64(4.85206162724305)], ['amazon_easytpp', np.float64(0.5196823759377474)], ['wikipedia', np.float64(2695.8874309547805)], ['retweet_easytpp', np.float64(2703.605205919073)], ['taobao_easytpp', np.float64(0.04638794128971427)], ['taxi_easytpp', np.float64(0.22379690905839614)], ['volcano_easytpp', np.float64(883.0893565576514)], ['hawkes_dependent', np.float64(4.051875708182595)], ['hawkes_1', np.float64(0.9579868619773451)]]")

df_times = pd.DataFrame(times, columns=['dataset', 'avg_time'])
df_sorted = df_times.dropna().sort_values(by='avg_time', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(df_sorted['dataset'], df_sorted['avg_time'])

ax.set_yscale('log')

ax.set_title('Average Inter-Event Time [hours] per Dataset', fontsize=14, fontweight='bold')
ax.set_ylabel('Time (log-scale)')
ax.set_xlabel('Dataset')

ax.set_xticks(range(len(df_sorted['dataset'])))
ax.set_xticklabels(df_sorted['dataset'], rotation=45, ha='right')

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('plot_avg_inter_event_times.pdf', dpi=300)