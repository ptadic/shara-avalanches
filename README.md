# Predicting the risk of avalanches on the Shara mountains

This repo contains Jupyter Notebooks and a Python script that demonstrate the training and evaluation of several standard machine learning models (logistic regression, random forest, SVM, XGBoost, LightGBM, neural network) for predicting the risk of avalanches based on features such as elevation, slope, land use, snow index etc.

The models rely on geospatial data stored in tif files and historical data about past occurrence of avalanches on the Shara mountain. To obtain these files, contact us at [urosdurlevicuros@gmail.com](mailto:durlevicuros@gmail.com).

Intended use of the files in the repo:

* `utils.py` contains helper functions, mainly for data loading and wrangling
* `eda.ipynb` performs an exploratory analysis of the dataset
* `avalanche_prediction.ipynb` demonstrates the training and evaluation of models for a single train/test split
* `classifier_evaluation.ipynb` performs a more detailed analysis of the trained models
* `double_cross_val.py` repeats training and evaluation of the models for several different train/test splits

Before running the notebooks or the `double_cross_val.py` script, make sure you have all the required data and set the paths where appropriate.
