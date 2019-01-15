"""TRAIN NEURAL NETS"""

import pandas as pd
from config import (PATH_MODELS, PATH_XTRAIN_PREPROCESSED,
                    PATH_YTRAIN_PREPROCESSED, SEPARATOR)
from evaluate import save_model_summary, save_learning_history
from keras import backend as K
from keras.layers import Dense, Dropout
#from keras.layers import BatchNormalization
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

###############################################################################
# LOADING DATA

def load_preprocessed_Xtrain():
    X = pd.read_csv(PATH_XTRAIN_PREPROCESSED, sep = SEPARATOR)
    print(f"LOADED Xtrain FROM DISC")
    return X
    
def load_preprocessed_ytrain():
    y = pd.read_csv(PATH_YTRAIN_PREPROCESSED,  sep = SEPARATOR)    
    print(f"LOADED ytrain FROM DISC")
    return y

###############################################################################
# INPUT TWEAKING
    
def get_input_shape():
    iter_csv = pd.read_csv(PATH_XTRAIN_PREPROCESSED, iterator=True, chunksize=1,sep = SEPARATOR) #usecols = X_COLUMNS
    Xtrain = next(iter_csv)
    shape = Xtrain.shape
    out = shape[1]
    return out

def tweak_X(Xtrain):
    shape = Xtrain.shape
    shape1 = shape[0]
    shape2 = shape[1]
    Xtrain = Xtrain.values.reshape(shape1,shape2)
    return Xtrain
    
def tweak_y(ytrain):
    '''ytrain comes as dataframe, needs little reshape before it can be used
    in scikit-learn models'''
    return ytrain.values.reshape(-1,)

###############################################################################
# MAIN FUNCTIONS
    
def prepare_data():
    Xtrain = load_preprocessed_Xtrain()
    Xtrain = tweak_X(Xtrain)
    ytrain = load_preprocessed_ytrain()
    ytrain = tweak_y(ytrain)
    return Xtrain, ytrain

def initialize_neural_net():
    input_shape = get_input_shape()
    K.clear_session()
    
    # MODEL DESIGN
    model = Sequential()
    model.add(Dense(60, activation = 'relu', input_shape=(input_shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation = 'relu' ))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation = 'relu' ))
    model.add(Dense(1, activation = 'sigmoid'))
    
#    loss = "categorical_crossentropy"
#    loss = "sparse_categorical_crossentropy"   
    loss = "binary_crossentropy"    
    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def initialize_model(modelname):
    '''initialize model and train one initial epoch'''
    model = initialize_neural_net()
    #history = model.fit(Xtrain, ytrain, epochs = 1, validation_split=0.2, batch_size=100)
    model.save(f"{PATH_MODELS}{modelname}.model")
    save_model_summary(model, modelname)
    #evaluate_nn(history, model, modelname)
    print("MODEL INITIALIZED AND SAVED")

def perform_training(modelname, epochs):
    '''load model and continue training for # of epochs'''
    Xtrain, ytrain = prepare_data()
    model = load_model(f"{PATH_MODELS}{modelname}.model")
    history = model.fit(Xtrain, ytrain, epochs = epochs, validation_split=0.2, batch_size=100)
    model.save(f"{PATH_MODELS}{modelname}.model")
    save_learning_history(history, modelname)
#    evaluate_nn(history, model, modelname)
    print("CONTINUED TRAINING AND SAVED MODEL")

def cv_kfold(folds, epochs):
    '''for small datasets, create NN, then do k-fold crossvalidation on it'''
    Xtrain, ytrain = prepare_data()
  #  ytrain = ytrain.values.reshape(-1,)
#    model = load_model(f"{PATH_MODELS}{modelname}.model")
    
    neural_network = KerasClassifier(build_fn=initialize_neural_net, 
                                 epochs=epochs, 
                                 batch_size=100, 
                                 verbose=1)
    print(cross_val_score(neural_network, Xtrain, ytrain, cv=folds))
    
