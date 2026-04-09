# MUSES Preprocessing Code

MUSES, a benchmark for **M**arked **U**nevenly **S**paced **E**vent **S**equences, is a collection of unevenly spaced time series datasets from various domains, containing marked events for training and evaluating prediction approaches.

## Datasets
MUSES datasets are available on [Huggingface](https://huggingface.co/datasets/ddrg/MUSES).

## Repository Contents
- This repository contains preprocessing code for MUSES datasets.
- The raw data sources are indicated in the respective **sources.txt** files.
- Multi-step preprocessing pipelines are indicated by the file names.
- For target data format description, data source licensing information and preprocessing literature sources, see the Huggingface repository (as referenced above) 

### Analysis
- The `_analysis` folder contains code for baseline generation, data analysis and plotting.

### Validation
- The `_misc` folder contains a data type validator for parquet files.
- This should be executed before uploading additional datasets to the Huggingface repository. 

### Licensing
- This Repository contains 3rd party code, which is subject to licensing.
- Licenses are indicated in a *LICENSE* or *license.md* file in the corresponding directory.
- Any of our own code, **unless indicated otherwise by a differing license-file or comment**, is distributed under the [MIT license](LICENSE).

## Attribution
If you wish to attribute us when using this code, see the Huggingface repository (as referenced above) for a citation template.  


# How to run the code
## PREREQUISITE
To run any preprocessing, download the raw data as indicated in the corresponding sources.txt file

## _easytpp
- run: `python main.py`
- output: parquet files for each data split of datasets 'taobao_easytpp', 'retweet_easytpp', 'volcano_easytpp', 'amazon_easytpp', 'taxi_easytpp' in subdirectories indicated by dataset name

## 911
- run: `python main.py`
- output: parquet files for each data split of dataset '911'

## crypto_transactions
- run: `python step1_polars.py`
- run: `python step2_read_parquet.py`
- output: parquet files for each data split of dataset 'crypto_transactions'

## earthquake
- run: `python earthquake.py`
- output: parquet files for each data split of dataset 'earthquake'

## hawkes
- run: `python generate_hawkes_data_enguehard.py`
- run: `python generate_hawkes_data_omi.py`
- Adjust paths to username in 'preprocess_hawkes_data.py'
- run: `python preprocess_hawkes_data.py`
- output: parquet files for each data split of datasets 'hawkes_dependent' (enguehard) and 'hawkes_1' (omi)
- NOTE: Used libraries might not yet work on latest python versions. Try running the hawkes generator on Python 3.8

## human_activity
- run: `python main.py`
- output: parquet files for each data split of dataset 'human_activity'

## memetrack
- run: `python memetrack.py`
- output: parquet files for each data split of dataset 'memetrack'
  
## mooc
- run: `python main.py`
- output: parquet files for each data split of dataset 'mooc'

## spiketrains
- run: `python step1_main.py`
- run: `python step2_loader.py`
- run: `python step3.py`
- output: parquet files for each data split of dataset 'spiketrains'
- NOTE: Used libraries might not yet work on latest python versions. Try running 'step1_main.py' on Python 3.7
  
## stackoverflow
- run: `python stackoverflow.py`
- output: parquet files for each data split of dataset 'stackoverflow'

## synthea
- PREREQUISITE: Run synthea simulator as indicated in sources.txt
- run: `python step1_main.py2`
- run: `python step2_reformat_preproc.py`
- output: parquet files for each data split of dataset 'synthea'

## taxi_nyc_neighborhoods
- run multiple: `python step1.py ID` (replace ID with the dataset-id for each in [1,2,6,7,8,9,10,11,12])
- run: `python step2.py`
- output: parquet files for each data split of dataset 'taxi_nyc_neighborhoods'

## wikipedia
- run: `python step1.py`
- run: `python step2.py`
- output: parquet files for each data split of dataset 'wikipedia'
