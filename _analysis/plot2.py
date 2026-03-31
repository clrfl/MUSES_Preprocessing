import pandas as pd
import matplotlib.pyplot as plt
pretty = {
    'ts_len_avg':'Average time series length',
    'ts_count_total':'Time series count',
    'class_count':'Class count',
    'event_count_total':'Event count',
}
df = pd.read_csv('analysisdoc.csv', sep=',')
df_pivot = df.pivot(index='dataset', columns='parameter', values='value').reset_index()

cols_to_convert = df_pivot.columns.drop('dataset')
df_pivot[cols_to_convert] = df_pivot[cols_to_convert].apply(pd.to_numeric, errors='coerce')

metriken = ['ts_len_avg', 'ts_count_total', 'class_count', 'event_count_total']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
axes = axes.flatten()

for i, metrik in enumerate(metriken):

    df_sorted = df_pivot[['dataset', metrik]].dropna().sort_values(by=metrik, ascending=False)
    axes[i].bar(df_sorted['dataset'], df_sorted[metrik])
    axes[i].set_yscale('log')

    axes[i].set_title(f'{pretty[metrik]}', fontsize=14, fontweight='bold')
    axes[i].set_ylabel('Value (log-scale)')

    axes[i].set_xticks(range(len(df_sorted['dataset'])))
    axes[i].set_xticklabels(df_sorted['dataset'], rotation=45, ha='right')

    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('plot2.pdf', dpi=300)