"""This script combines alle major steps of the ML pipeline and serves as the main
controller. Check config.py for settings"""

from predict import ML_predict, NN_predict, combine_submission_chunks
from preprocess_test import main as preprocess_testingdata
from preprocess_train import main as preprocess_trainingdata
from train_ML import main as train_model
from train_NN import perform_training, initialize_model, cv_kfold

###############################################################################
# DATA PREPARATION
"""
def wrangle_data():
    '''combine/wrangle data from different sources into single *.csv files'''
    wrangle_trainingdata()
    wrangle_testdata()
"""

def preprocess_traindata(trainingset_size): # set trainingset size in config
    '''preprocess trainingdata as defined in preprocess_train.py'''
    preprocess_trainingdata(trainingset_size)
    
###############################################################################
# TRAIN ML MODEL
def train_ML_model(modelname):
    train_model(modelname)

###############################################################################
# TRAIN NEURAL NETWORK 
def initialize_nn(modelname):
    initialize_model(modelname)
    
def perform_training_nn(modelname, epochs):
    '''perform initial training / continuation of training using Keras API, for big 
    datasets'''
    perform_training(modelname, epochs)
    
def cv_nn(folds, epochs = 1):
    '''for smaller datasets, create NN, then do k-fold cross-validation on it'''
    cv_kfold(folds, epochs)

###############################################################################
# APPLY MODEL ON UNSEEN DATA (PREDICT ON TEST) 
def preprocess_testdata(chunksize=1000000): # set chunksize < dataset to predict on chunks
    # ! BE SURE PROCESS MATCHES PREPROCESSING OF TRAININGDATA
    preprocess_testingdata(chunksize)    

def predict_chunks_ML(modelname, chunk_first, chunk_last):
    ML_predict(modelname, chunk_first, chunk_last)
    
def predict_chunks_NN(modelname, chunk_first=0, chunk_last=0):
    NN_predict(modelname, chunk_first, chunk_last)
    
##############################################################################
# COMBINE SUBMISSION FILE CHUNKS
def combine_submissions(modelname, chunk_first, chunk_last):
    combine_submission_chunks(modelname, chunk_first, chunk_last)
