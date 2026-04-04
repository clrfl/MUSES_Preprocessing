import json
import numpy as np
import os

from argparse import Namespace
from pathlib import Path
from tqdm import tqdm

from tpp.processes.multi_class_dataset import generate_points
from tpp.utils.cli import parse_args
from tpp.utils.run import make_deterministic
from tpp.utils.record import hawkes_seq_to_record

import sys
import matplotlib.pyplot as plt
import time

def apply_pattern(marks):
    maximum = 0.6
    middle = maximum / 2

    high_alpha = [x for x in range(marks)]
    rng = np.random.default_rng(seed=42)
    rng.shuffle(high_alpha)

    mid_alpha = [x for x in range(marks)]
    rng = np.random.default_rng(seed=43)
    rng.shuffle(mid_alpha)

    matrix = np.full((marks, marks), 0.0)

    for item in [[high_alpha, maximum], [mid_alpha, middle]]:
        for i in range(len(item[0])):
            matrix[item[0][i]][item[0][(i+1)%marks]] = item[1]

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap='magma', aspect='auto')
    plt.colorbar(label='alpha value')
    plt.title('150x150 Hawkes Alpha Value Matrix')
    plt.xlabel('alpha from class')
    plt.ylabel('alpha to class')
    plt.tight_layout()
    plt.savefig('hawkes_heatmap.pdf', dpi=300)

    return np.array(matrix, dtype=np.float32)


def main(args: Namespace):
    if args.verbose:
        print(args)

    # Create paths for plots
    data_dir = os.path.expanduser(args.data_dir)
    hawkes_data_dir = os.path.join(data_dir, "hawkes")
    Path(hawkes_data_dir).mkdir(parents=True, exist_ok=True)

    seeds = {
        "train": [args.train_size, args.seed],
        "val": [args.val_size, args.seed + args.train_size],
        "test": [args.test_size, args.seed + args.train_size + args.val_size]}

    for name, [size, seed] in seeds.items():
        range_size = range(size)
        if args.verbose:
            range_size = tqdm(range_size)

        times_marked = [
            generate_points(
                n_processes=args.marks,
                mu=args.mu.astype(np.float64),
                alpha=args.alpha.astype(np.float64),
                decay=args.beta.astype(np.float64),
                window=args.window,
                seed=seed + i
            ) for i in range_size]  # D x M x Li

        records = [hawkes_seq_to_record(r) for r in times_marked]

        with open(os.path.join(hawkes_data_dir, name + ".json"), "w") as h:
            h.write(
                '[' + ',\n'.join(json.dumps(i) for i in records) + ']\n')

    codes_to_names = {str(i): str(i) for i in range(args.marks)}

    args_dict = vars(args)
    args_dict['alpha'] = args_dict['alpha'].tolist()
    args_dict['beta'] = args_dict['beta'].tolist()
    args_dict['mu'] = args_dict['mu'].tolist()
    keys_to_keep = ["seed", "mu", "alpha", "beta", "marks", "hawkes_seed",
                    "window", "train_size", "val_size", "test_size"]
    args_dict = {k: args_dict[k] for k in keys_to_keep}
    with open(os.path.join(hawkes_data_dir, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)
    with open(os.path.join(hawkes_data_dir, 'int_to_codes.json'), 'w') as fp:
        json.dump(codes_to_names, fp)
    with open(os.path.join(hawkes_data_dir, 'codes_to_int.json'), 'w') as fp:
        json.dump(codes_to_names, fp)
    with open(os.path.join(hawkes_data_dir, 'codes_to_names.json'), 'w') as fp:
        json.dump(codes_to_names, fp)
    with open(os.path.join(hawkes_data_dir, 'names_to_codes.json'), 'w') as fp:
        json.dump(codes_to_names, fp)
    print(hawkes_data_dir)


if __name__ == "__main__":

    # inject cli parameters
    sys.argv = sys.argv + [str(x) for x in ['--marks', 150,
                                            '--train-size', 2000,
                                            '--val-size', 0,
                                            '--test-size', 0,
                                            '--window', 65,
                                            # '--alpha',  0.01, 0.05, 0.01,
                                            #             0.05, 0.01, 0.01,
                                            #             0.01, 0.01, 0.05,
                                            # '--beta',   0.1, 0.5, 0.1,
                                            #             0.5, 0.1, 0.1,
                                            #             0.1, 0.1, 0.5,
                                            # '--mu',     0.01, 0.01, 0.01,
    ]]

    parsed_args = parse_args()
    # check_repo(allow_uncommitted=not parsed_args.use_mlflow)
    make_deterministic(seed=parsed_args.seed)

    parsed_args.mu = np.array([0.01 for _ in range(parsed_args.marks)], dtype=np.float32)
    parsed_args.beta = matrix = np.full((parsed_args.marks, parsed_args.marks), 1.0)

    parsed_args.alpha = apply_pattern(parsed_args.marks)

    start = time.time()
    main(args=parsed_args)
    print(time.time()-start)

    data = json.load(open('/home/user/neural-tpps/data/hawkes/train.json'))
    counter = 0
    tscounter = 0
    lens = []
    for el in data:
        lens.append(len(el))
        tscounter += 1
        for item in el:
            counter += 1
    print(counter, 'events')
    print(tscounter, 'timeseries')
    print(np.mean(lens))
