import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, axs = plt.subplots(4, 3, figsize=(20, 20))
    axs = axs.flatten()

    for i in range(1, 13):

        file_path = f'./taxis{i}.parquet'

        df = pd.read_parquet(file_path, columns=['datetime'])
        seconds = pd.to_datetime(df["datetime"].explode()).dt.second

        sec_counts = seconds.value_counts().sort_index()

        ax = axs[i - 1]
        ax.bar(sec_counts.index, sec_counts.values)
        ax.set_title(f"Dataset {i}")
        ax.set_xlabel('Second')
        ax.set_ylabel('Quantity')

    plt.tight_layout()
    plt.savefig('NYTax_plot.pdf', dpi=300, bbox_inches='tight')