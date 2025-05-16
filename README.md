# Components of DRPAD


# Data
SMD, MSL, SMAP, SMD datasets were acquired at datasets and SWaT, WADI can be requested at Itrust. MBA, UCR, NAB was acquired at TranAD datasets and MSDS can be requested at zenodo. Pruned and remedied {SMD, MSL, SMAP} were acquired at TranAD datasets.

# Data Preparation

All data Preparation are same as AFMF framework \url{https://github.com/OrigamiSL/AFMF?tab=readme-ov-file}. You can run ./DRPAD/data/preprocess.py to preprocess these raw data. 

# Usage
Commands for training and testing models combined with DRPAD of all datasets are in ./scripts/<model>.sh.

# Results
The experiment parameters of certain model under each data set are formated in the <model>.sh files in the directory ./scripts/. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better results.
