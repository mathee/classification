from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.models import load_model
from config import PATH_XTRAIN_PREPROCESSED, PATH_YTRAIN_PREPROCESSED, SEPARATOR, PATH_MODELS
import pandas as pd


###############################################################################
# LOADING DATA

def load_preprocessed_Xtrain():
    X = pd.read_csv(PATH_XTRAIN_PREPROCESSED, sep = SEPARATOR)
    print(f"LOADED Xtest FROM DISC")
    return X
    
def load_preprocessed_ytrain():
    y = pd.read_csv(PATH_YTRAIN_PREPROCESSED,  sep = SEPARATOR)    
    print(f"LOADED ytest FROM DISC")
    return y

###############################################################################
# INPUT TWEAKING
    
def get_input_shape(Xtrain):
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
    input_shape = get_input_shape(Xtrain)
    return Xtrain, ytrain, input_shape

def initialize_neural_net(input_shape):
    K.clear_session()
    
    model = Sequential()
    
    model.add(Dense(40, activation = 'relu', input_shape=(input_shape,)))
#    model.add(Dropout(0.5))
    model.add(Dense(100, activation = 'relu' ))
#    model.add(Dropout(0.5))
#    model.add(Dense(1000, activation = 'relu' ))
    model.add(Dense(1, activation = 'sigmoid'))
    
#    loss = "categorical_crossentropy"
#    loss = "sparse_categorical_crossentropy"   
    loss = "binary_crossentropy"    
    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def initialize_training():
    Xtrain, ytrain, input_shape = prepare_data()
    model = initialize_neural_net(input_shape)
    model.fit(Xtrain, ytrain, epochs = 1, validation_split=0.2, batch_size=400)
    model.save(f"{PATH_MODELS}NEURAL_NET.model")
    info = f"{model.summary()}"
    with open(f"{PATH_MODELS}NEURAL_NET_readme.txt", "w") as text_file:
        text_file.write(info)
    print("MODEL INITIALIZED AND SAVED")

def continue_training(epochs):
    Xtrain, ytrain, input_shape = prepare_data()
    model = load_model(f"{PATH_MODELS}NEURAL_NET.model")
    model.fit(Xtrain, ytrain, epochs = epochs, validation_split=0.2, batch_size=400)
    model.save(f"{PATH_MODELS}NEURAL_NET.model")
    info = f"{model.summary()}"
    with open(f"{PATH_MODELS}NEURAL_NET_readme.txt", "w") as text_file:
        text_file.write(info)
    print("CONTINUED TRAINING AND SAVED MODEL")

