# Campus Crowd Dataset and STEN Benchmark Models

This repository provides the publicly accessible portion of the Campus Crowd dataset (no raw videos, only contains crowd counts) and the experimental implementations for the paper 

*Enhancing Data-Driven Predictive Modeling of Pedestrian Crowd Flow with Spatial Priors – Case Studies with Post-Event Crowd Data on a University Campus*

published at *IEEE Big Data 2024* - Author: Vivian W.H. Wong

## Installation

```bash
git clone https://github.com/vivian-wong/Campus-Crowd
cd Campus-Crowd
# create conda virtual environment
conda create --name campus-crowd python=3.10 
conda activate campus-crowd
# install prerequisites
pip install -r requirements.txt
# install pytorch geometric temporal
python setup.py
```

## Usage
Loading datasets
```
from dataloader import DatasetLoaderStatic
dataset = DatasetLoaderStatic('path/to/dataset')
```
Running crowd flow forecasting with STEN models 
See demo notebook on forecasting. 

## Examples
Check the examples/ directory for simplified demo notebooks.

## Reproducing paper experiments 
To run all experiments as detailed in the paper, run 
```
bash reproduce_paper_experiments.sh
```
and generate plots with the jupyter notebook experiments/generate_plots.ipynb

## Contributing
Contributions are welcome! Please read the CONTRIBUTING.md for guidelines.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

