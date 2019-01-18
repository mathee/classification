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
Python 3.6<br/>
#### Data Handling
numpy<br/>
pandas<br/>
scipy<br/>
#### Models 
Keras<br/>
tensorflow<br/>
scikit-learn<br/>
xgboost<br/>
#### Plotting
matplotlib<br/>
seaborn<br/>
## Main Process
#### 1. Data Wrangling
#### 2. Exploration
#### 3. Preprocessing
#### 4. Training
#### 5. Predicting
## Workflow Structure
![alt text](dataflows.png)
## Files
main.py<br/>
config.py<br/>
wrangle.py<br/>
exploration.py<br/>
explore.ipynb<br/>
preprocess_train.py<br/>
feature_engineering.py<br/>
train_ML.py<br/>
train_NN.py<br/>
evaluate.py<br/>
preprocess_test.py<br/>
predict.py<br/>
## Folder Structure
-/data - *contains all data*<br/>
-/data/raw - *contains raw data, raw files, in various formats, shapes etc.*<br/>
-/data/wrangled - *train.csv, test.csv combining wrangled features in single files*<br/>
-/data/preprocessed - *numerical, cleaned data for direct use in ML models*<br/>
-/models - *contains trained models*<br/>
-/models/support - *support models like fitted PCAs, scalers etc that will be applied on test data*<br/>
-/results - *all results*<br/>
-/reports - *model information, trainings scores, neural net learning history etc.*<br/>