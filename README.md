# Components of DRPAD
![main_img](img/main_image.jpg)
## Dynamic Prediction Replacement (DPR)
To mitigate the propagation of anomalies, we introduce a dynamic prediction replacement mechanism. This component dynamically updates identified anomalous points with their predicted values, thereby suppressing their disruptive effects on future predictions and significantly enhancing overall detection performance.
![main_img](img/main_image.jpg)
## Segmentation-Based Normalization via Change Point Detection(SN)
To address distribution shifts, we propose a segmentation-based normalization approach. Specifically, the time series is divided into statistically independent segments using change point detection, and each segment is normalized individually. The segments are then reassembled to restore the original temporal structure. This process ensures a consistent statistical scale across segments, effectively mitigating the impact of distributional heterogeneity
## Mean \& Dimension Dual-Check Strategy(MDDC)
To improve the detection of univariate anomalies, we develop a hybrid thresholding approach based on multidimensional sensitivity. This strategy combines global statistical indicators with per-dimension checks to better capture subtle and localized deviations.





# Data
SMD, MSL, SMAP, SMD datasets were acquired at datasets and SWaT, WADI can be requested at Itrust. MBA, UCR, NAB was acquired at TranAD datasets and MSDS can be requested at zenodo. Pruned and remedied {SMD, MSL, SMAP} were acquired at TranAD datasets.

# Data Preparation

All data Preparation are same as [AFMF](https://github.com/OrigamiSL/AFMF?tab=readme-ov-file) framework. You can run ./DRPAD/data/preprocess.py to preprocess these raw data. 

# Usage
Commands for training and testing models combined with DRPAD of all datasets are in ./scripts/<model>.sh.

# Results
The experiment parameters of certain model under each data set are formated in the <model>.sh files in the directory ./scripts/. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better results.
