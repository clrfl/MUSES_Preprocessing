import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

df = pd.read_csv('analysisdoc.csv', sep=',')

target_params = ['ts_len_avg', 'ts_count_total', 'class_count', 'event_count_total']
pretty = {
    'ts_len_avg':'average time series lengths',
    'ts_count_total':'time series count',
    'class_count':'class count',
    'event_count_total':'event count',
}
df_filtered = df[df['parameter'].isin(target_params)].copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, param in enumerate(target_params):
    data = df_filtered[df_filtered['parameter'] == param]['value']
    sns.histplot(data, bins=30, kde=False, ax=axes[i], log_scale=True, color='C0')

    axes[i].set_title(f'Distribution of {pretty[param]} over datasets', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Value (log-scale)')
    axes[i].set_ylabel('Dataset count')

    axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    axes[i].grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('plot3.pdf', dpi=300)