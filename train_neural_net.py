from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from config import PATH_XTRAIN_PREPROCESSED, PATH_YTRAIN_PREPROCESSED, SEPARATOR
import pandas as pd


###############################################################################
# LOADING DATA

def load_preprocessed_Xtrain():
    X = pd.read_csv(PATH_XTRAIN_PREPROCESSED, sep = SEPARATOR)
    print(f"LOADED X FROM DISC")
    return X
    
def load_preprocessed_ytrain():
    y = pd.read_csv(PATH_YTRAIN_PREPROCESSED,  sep = SEPARATOR)    
    print(f"LOADED y FROM DISC")
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
    Xtrain = Xtrain - Xtrain.mean()
    return Xtrain
    
def tweak_y(ytrain):
    '''ytrain comes as dataframe, needs little reshape before it can be used
    in scikit-learn models'''
    return ytrain.values.reshape(-1,)

###############################################################################
# MAIN FUNCTIONS
    

def initialize_neural_net(input_shape):
    K.clear_session()
    model = Sequential()
    model.add(Dense(26, activation = 'sigmoid', input_shape=(input_shape,)))
    model.add(Dense(20, activation = 'sigmoid'))
    model.add(Dense(20, activation = 'sigmoid'))
    model.add(Dense(20, activation = 'sigmoid'))
    model.add(Dense(20, activation = 'sigmoid'))
    model.add(Dense(20, activation = 'sigmoid'))
    model.add(Dense(20, activation = 'sigmoid'))
    model.add(Dense(20, activation = 'sigmoid'))
    model.add(Dense(1, activation = 'sigmoid'))
#    loss = "categorical_crossentropy"
#    loss = "sparse_categorical_crossentropy"   
    loss = "binary_crossentropy"    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def initialize_training(model, Xtrain, ytrain):
    model.fit(Xtrain, ytrain, epochs = 50, validation_split=0.2, batch_size=100)
    

def continue_training():
    return 1
    # load model
    # model fit
    # save model

def main():
    Xtrain = load_preprocessed_Xtrain()
    Xtrain = tweak_X(Xtrain)
    ytrain = load_preprocessed_ytrain()
    ytrain = tweak_y(ytrain)
    input_shape = get_input_shape(Xtrain)
    model = initialize_neural_net(input_shape)
    model = initialize_training(model, Xtrain, ytrain)
    return model
main()