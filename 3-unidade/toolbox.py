import keras
from keras import layers

def MyModel(input_size=200):
    
    x_in = keras.Input(shape=(input_size,))
    
    x = layers.Dense(50, activation="relu")(x_in)
    
    x_out = layers.Dense(1, activation="sigmoid")(x)
    
    return keras.Model(inputs=x_in, outputs=x_out)

