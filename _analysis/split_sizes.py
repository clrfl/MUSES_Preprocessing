from datasets import load_dataset

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
    'hawkes_1'
]

for ds in dataset_configs:

    dataset = load_dataset("ddrg/MUSES", ds)
    print(f"|{ds}|{len(dataset['train'])}|{len(dataset['validation'])}|{len(dataset['test'])}|")