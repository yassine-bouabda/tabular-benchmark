# Tabular ML Benchmark Tool
---  
## About The Tool 
The Tabular ML Benchmark is a tool designed to compare the performance of machine learning algorithms on tabular data. It is particularly useful for evaluating new tabular transformers.against traditional ML models. This benchmark supports both classification and regression tasks. 
Check the results at my meduim blog 

## Hypothesis
- The benchmark is based on datasets from OpenML, which have been cleaned and curated by [Inria-soda](https://huggingface.co/datasets/inria-soda/tabular-benchmark).
- To ensure fair comparisons, no additional feature engineering or hyperparameter tuning was performed. All algorithms are used with their default parameters.

## Customization 
- Due to computational and memory constraints, users can select a subset of datasets.
- The maximum dataset length can be limited (e.g., TabPFN recommends restricting the dataset size to 10,000 rows for optimal performance).
- Performance metrics are customizable, allowing flexibility in evaluation criteria.



