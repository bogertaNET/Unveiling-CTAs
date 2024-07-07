# Unveiling Cyber Threat Actors: A Hybrid Deep Learning Approach for Behavior-based Attribution

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Folder Structure](#folder-structure)
4. [Questions](#questions)

## Overview
Authors:
- Emirhan Böge, Sabanci University
- M. Bilgehan Ertan, Vrije Universiteit Amsterdam
- Halit Alptekin, PRODAFT
- Orçun Çetin, Sabanci University

This paper has been accepted to the ACM Journal: Digital Threats: Research and Practice (DTRAP).

## Abstract
```
In this paper, we leverage natural language processing and machine learning algorithms to profile threat actors based on their behavioral signatures to establish identification for soft attribution. Our unique dataset comprises various actors and the commands they have executed, with a significant proportion using the Cobalt Strike framework in August 2020-October 2022. We implemented a hybrid deep learning structure combining transformers and convolutional neural networks to benefit global and local contextual information within the sequence of commands, which provides a detailed view of the behavioral patterns of threat actors. We evaluated our hybrid architecture against pre-trained transformer-based models such as BERT, RoBERTa, SecureBERT, and DarkBERT with our high-count, medium-count, and low-count datasets. Hybrid architecture has achieved F1-score of 95.11% and an accuracy score of 95.13% on the high-count dataset, F1-score of 93.60% and accuracy score of 93.77% on the medium-count dataset, and F1-score of 88.95\% and accuracy score of 89.25% on the low-count dataset. Our approach has the potential to substantially reduce the workload of incident response experts who are processing the collected cybersecurity data to identify patterns.
```

## Installation

To install all the required dependencies, run:

```bash
pip install -r requirements.txt
```
## Folder Structure

- data_prep: Contains code for standardized common language converters and creation of Torch DataLoaders and Hugging Face Datasets. 
- experiments: This folder is dedicated to hyperparameter tuning.
- validate: Contains code for model validation using cross-validation techniques.
- training: Includes the last training run and testing scripts.
- models: Trained hybrid architecture for each dataset is stored here.
- data: All datasets and related supporting files are stored here. **You need to unzip the data.zip file inside this folder to access the all necessary files.**

## Questions

If you have any questions or need further clarification, feel free to reach out to the authors of this paper or create an issue in this repository:

- Emirhan Böge: [emirhanboge@sabanciuniv.edu](mailto:emirhanboge@sabanciuniv.edu)
- M. Bilgehan Ertan: [m.b.ertan@student.vu.nl](mailto:m.b.ertan@student.vu.nl)

```
@article{10.1145/3676284,
author = {B\"{o}ge, Emirhan and Ertan, Murat Bilgehan and Alptekin, Halit and \c{C}etin, Or\c{c}un},
title = {Unveiling Cyber Threat Actors: A Hybrid Deep Learning Approach for Behavior-based Attribution},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3676284},
doi = {10.1145/3676284},
journal = {Digital Threats},
}
```
