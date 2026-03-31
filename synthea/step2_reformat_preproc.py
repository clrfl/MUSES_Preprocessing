import json
import pandas as pd

encounter_codes = list(eval("{'SNOMED-CT_439740005', 'SNOMED-CT_453131000124105', 'SNOMED-CT_305408004', 'SNOMED-CT_40274000', 'SNOMED-CT_79094001', 'SNOMED-CT_305411003', 'SNOMED-CT_56876005', 'SNOMED-CT_702927004', 'SNOMED-CT_386395000', 'SNOMED-CT_397821002', 'SNOMED-CT_183460006', 'SNOMED-CT_448337001', 'SNOMED-CT_439708006', 'SNOMED-CT_185316007', 'SNOMED-CT_86013001', 'SNOMED-CT_183452005', 'SNOMED-CT_185317003', 'SNOMED-CT_47505003', 'SNOMED-CT_162673000', 'SNOMED-CT_170838006', 'SNOMED-CT_394701000', 'SNOMED-CT_36228007', 'SNOMED-CT_185345009', 'SNOMED-CT_390906007', 'SNOMED-CT_183495009', 'SNOMED-CT_33879002', 'SNOMED-CT_108219001', 'SNOMED-CT_4525004', 'SNOMED-CT_185389009', 'SNOMED-CT_424441002', 'SNOMED-CT_305432006', 'SNOMED-CT_50849002', 'SNOMED-CT_305351004', 'SNOMED-CT_1505002', 'SNOMED-CT_424619006', 'SNOMED-CT_185347001', 'SNOMED-CT_185349003', 'SNOMED-CT_32485007', 'SNOMED-CT_67799006', 'SNOMED-CT_169762003', 'SNOMED-CT_371883000', 'SNOMED-CT_305336008', 'SNOMED-CT_270427003', 'SNOMED-CT_698314001', 'SNOMED-CT_410620009', 'SNOMED-CT_310061009', 'SNOMED-CT_308335008', 'SNOMED-CT_305342007', 'SNOMED-CT_281036007', 'SNOMED-CT_449214001', 'SNOMED-CT_308251003'}"))
idsdict = {}


for idx, i in enumerate(encounter_codes):
    idsdict[i] = idx

results = []

with open ('./synthea/output/FULL/preproc/train.json') as f:
    for x in f.readlines():
        ids = []
        dates = []

        x = x[x.find('{'):x.rfind('}')+1]
        y = json.loads(x)['events']
        for z in y:
            ids.append(idsdict[z['encounter']['code']])
            dates.append(z['encounter']['start'])

        if len(ids) < 2:
            continue

        df = pd.DataFrame({'type_event': ids, 'datetime': dates})

        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%SZ')

        first_time = df['datetime'][0]

        df['time_since_start'] = (df['datetime'] - first_time).dt.total_seconds() / 3600  # convert to hours
        df['time_since_last_event'] = (df['datetime'].diff().dt.total_seconds().fillna(0)) / 3600

        results.append(
            [df['datetime'].to_list(), len(df), df['time_since_start'].to_list(), df['time_since_last_event'].to_list(), df['type_event'].to_list()])

df = pd.DataFrame(results, columns=['datetime', 'seq_len', 'time_since_start', 'time_since_last_event', 'type_event'])

df['dim_process'] = 51

print(len(df))
train = df.sample(frac=0.8, random_state=42)
train['seq_idx'] = range(len(train))
train.to_parquet('train.parquet', index=False)
remain = df.drop(train.index)
print(len(train), len(remain))
val = remain.sample(frac=0.5, random_state=42)
val['seq_idx'] = range(len(val))
test = remain.drop(val.index)
test['seq_idx'] = range(len(test))
val.to_parquet('validation.parquet', index=False)
test.to_parquet('test.parquet', index=False)
print(len(val), len(test))
