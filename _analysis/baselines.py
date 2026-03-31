import numpy as np
from sklearn.metrics import root_mean_squared_error, f1_score
from datasets import load_dataset
import pandas as pd

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
    'hawkes_1',
    'hawkes_difficult'
]
with open('baselines.csv', 'w', encoding='utf-8') as f:
    print("dataset_name,rmse_time,acc_A,acc_B,f1_macro_A,f1_macro_B", file=f)
    for dsname in dsnames:
        print("Loading dataset " + dsname)
        ds = load_dataset("ddrg/MUSES", dsname)
        time_delta_seqs = ds['test']["time_since_last_event"]
        type_seqs = ds['test']["type_event"]
        print(' >',ds.keys())

        print("Calculating average inter event time...")
        # train_avg_time is the average inter event time of all sequences. We discard 0.0 as it is sequence start.
        train_inter_event_times = pd.Series(ds['train']['time_since_last_event']).explode()
        train_inter_event_times = train_inter_event_times[train_inter_event_times != 0.0]
        train_avg_time = np.mean(train_inter_event_times.tolist())
        print(' >',train_avg_time)

        print("Calculating most common type...")
        train_types = pd.Series(ds['train']['type_event'])
        train_most_pop_type = train_types.explode().value_counts().idxmax()
        print(' >',train_most_pop_type)

        print("Creating successor dict...")
        train_sequence_matrix = np.zeros((ds['train']['dim_process'][0], ds['train']['dim_process'][0]))

        from_nodes = np.concatenate([ts[:-1] for ts in train_types])
        to_nodes = np.concatenate([ts[1:] for ts in train_types])
        np.add.at(train_sequence_matrix, (from_nodes, to_nodes), 1)

        successor_dict = dict()
        for i in range(ds['train']['dim_process'][0]):
            i_successors = train_sequence_matrix[i]
            if i_successors.sum() == 0:
                successor_dict[i] = train_most_pop_type # if element doesnt have successors, use most common type
                continue
            max_amount = i_successors.max()
            successor_ids = [i for i, v in enumerate(i_successors) if v == max_amount]
            successor_dict[i] = successor_ids[np.random.choice(len(successor_ids))] # choose random item out of most common successors
        print(' >',successor_dict)

        print("Creating type predictions...")
        total_predictions = 0
        baseline_B_types = []
        true_types = []
        true_times = []
        for row in ds['test']:
            total_predictions += row['seq_len']-1

            for item in row['type_event'][:-1]:
                baseline_B_types.append(successor_dict[item])

            true_types.extend(row['type_event'][1:])
            true_times.extend(row['time_since_last_event'][1:])


        print("Calculating metrics...")
        rmse = root_mean_squared_error(true_times, [train_avg_time] * total_predictions)

        acc_A = float(np.mean(np.array(true_types) == train_most_pop_type))
        acc_B = float(np.mean(np.array(baseline_B_types) == np.array(true_types)))

        f1_macro_A = f1_score(true_types, [train_most_pop_type] * total_predictions, average='macro')
        f1_macro_B = f1_score(true_types, baseline_B_types, average='macro')

        print(dsname,rmse,acc_A,acc_B,f1_macro_A,f1_macro_B, file=f, sep=',')
