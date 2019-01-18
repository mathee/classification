# Machine Learning Essentials  
"Machine Learning Essentials" is two things at the same time:
1. A workflow structure for doing machine learning (in this case: building and
evaluating a model)
2. A collection of frequently used functions.
The goal of this project is to provide a solid template that
can - after customization - be used for many of the "classical"
classification and/or regression problems that might occur in a hands-on
business context. No fancy rocket science, but solid machine learning
with a practial approach.
It is continuoulsy developed - so if I come across new and exiting 
things that fit in, I will definitely add them.
## Used Tech
#### General
Python 3.6
Jupyter 
#### Data Handling
numpy
pandas
scipy
#### Models 
Keras
tensorflow
scikit-learn
xgboost
#### Plotting
matplotlib
seaborn
## Files
#### config.py
#### main.py
#### wrangle.py
#### exploration.py
#### explore.ipynb
#### preprocess_train.py
#### feature_engineering.py
#### train_ML.py
#### train_NN.py
#### preprocess_test.py
#### predict.py
## Folder Structure
-/data - *contains all data*<br/>
-/data/raw - *contains raw data, raw files, in various formats, shapes etc.*<br/>
-/data/prepared - *train.csv, test.csv combining all features in single files*<br/>
-/data/preprocessed - *numerical, cleaned data for direct use in ML models*<br/>
-/models - *contains trained models*<br/>
-/models/support - *support models like fitted PCAs, scalers etc that will be applied on test data*<br/>
-/results - *all results*<br/>