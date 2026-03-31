from datasets import load_dataset_builder, load_dataset

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
    #'hawkes_difficult'
]

sum_download_size = 0
sum_dataset_size = 0
sum_total_size = 0

for ds in dataset_configs:

    dataset = load_dataset("ddrg/MUSES", ds)
    builder = load_dataset_builder("ddrg/MUSES", ds)

    info = builder.info

    BYTES_TO_MB = 1E6

    download_size = (info.download_size or 0) / BYTES_TO_MB
    dataset_size = (info.dataset_size or 0) / BYTES_TO_MB
    total_size = (info.size_in_bytes or 0) / BYTES_TO_MB

    sum_download_size += download_size
    sum_dataset_size += dataset_size
    sum_total_size += total_size

    print(f"#### {ds}\n")
    print(f"- **Size of downloaded dataset files:** {download_size:.2f} MB")
    print(f"- **Size of the generated dataset:** {dataset_size:.2f} MB")
    print(f"- **Total amount of disk used:** {total_size:.2f} MB\n")

print('sum_download_size:', sum_download_size)
print('sum_dataset_size:', sum_dataset_size)
print('sum_total_size:', sum_total_size)