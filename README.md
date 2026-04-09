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


