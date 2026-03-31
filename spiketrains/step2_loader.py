import pickle
import pandas as pd
import math

def calculate_class(y):
    reval = math.floor(y * 10) - 1
    if reval == 9:
        return 8
    return reval

for idx in [1,2,3,4,5,6]:
    with open('network'+str(idx)+'_oopsi.pkl', 'rb') as f:
        (S, F, C, network, pos) = pickle.load(f)

        dfs = pd.DataFrame(S)

        #dfs=dfs.head(10000)
        #
        # plt.plot(dfs.index, dfs[0])
        # plt.show()

        with open("full_preproc_" + str(idx) + ".jsonl", 'w', encoding='utf-8') as g:

            for x in dfs.columns:
                new_df = dfs[[x]].rename({x:"n"}, axis=1)

                new_df = new_df[new_df['n'] >= 0.1]
                new_df['n'] = new_df['n'].apply(calculate_class)
                new_df['n'] = new_df['n'].astype(int)

                new_df['seconds'] = new_df.index / 50

                g.write(new_df.T.to_json(orient="values") + '\n')

                #print(new_df)