# Machine Learning Essentials  
"Machine Learning Essentials" is two things at the same time:
1. A workflow structure for doing machine learning (in this case: building and
evaluating a model)
2. A collection of frequently used functions.
The goal of this project is to provide a solid template that
can - after customization - be used for many of the "classical"
classification and/or regression problems that might occur in a hands-on
business context. No fancy rocket science, but solid machine learning
with a practial approach.<br/>
It is continuoulsy developed - so if I come across new and exiting 
things that fit in, I will definitely add them.
## Used Tech
#### General
Python 3.6
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
## Workflow and used files
#### 1. Wrangling
Obtain raw data from different sources like APIs, websites, databases, csv files etc and
'wrangle' them into single files for model training and test (train.csv / test.csv).
Output is saved in *data/wrangled*.
(**wrangle.py** is the script for that - in this template only a placeholder)
#### 2. Exploration
Reading data from *data/wrangled*, then getting a feel for the data, doing some visualizations, #
trying out different things: Essentially it is about getting to know the data and some ideas
on how it can be used for machine learning.
(**explore.ipynb** for actually doing stuff, **exploration.py**
contains useful and reusable visualization functions etc.)
#### 3. Preprocessing trainingdata / Feature Engineering
Doing all preprocessing steps that are required to read training data into a machine learning
model or neural net, e.g. encoding of categorical features, scaling, imputations etc. In addition
to that, this is also the steo where all the feature engineering happens (**preprocess_train.py** contains
many reusable preprocessing functions and the main preprocessing pipeline, 
**feature_engineering.py** is the place to put more specialized functions 
that are highly individual for every project, thus the file
is more or less a placeholder here)


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
#### evaluate.py
#### preprocess_test.py
#### predict.py
## Folder Structure
-/data - *contains all data*<br/>
-/data/raw - *contains raw data, raw files, in various formats, shapes etc.*<br/>
-/data/wrangled - *train.csv, test.csv combining wrangled features in single files*<br/>
-/data/preprocessed - *numerical, cleaned data for direct use in ML models*<br/>
-/models - *contains trained models*<br/>
-/models/support - *support models like fitted PCAs, scalers etc that will be applied on test data*<br/>
-/results - *all results*<br/>
-/reports - *model information, trainings scores, neural net learning history etc.*<br/>