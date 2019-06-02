# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
import generators
import importlib as lb
lb.reload(generators)

class MediumModel:
    
    def build_a_branch(inputs, numclasses =26, kernel_size = (20,11), finalAct="softmax", name = None):
        # CONV => RELU => POOL
        x = Conv2D(filters=32,kernel_size= kernel_size,padding='same')(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(32, kernel_size, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(32, (20, 11), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(numclasses)(x)
        x = Activation(finalAct, name=name)(x)

        return x

    @staticmethod
    def build(width = 200, height = 20, numclasses =26, kernel_size = (20,11), finalAct="softmax"):
        
        inputShape = (height, width, 1)
        inputs = Input(shape=inputShape)
        
        first  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'first')
        second = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'second')
        third  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'third')
        fourth = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'fourth')
        fifth  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'fifth')
        sixth  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'sixth')
        seventh= MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'seventh')
        eighth = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'eighth')
        nineth = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'nineth')
        tenth  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'tenth')
        outputs=[first, second, third, fourth, fifth, sixth, seventh, eighth, nineth, tenth]
        model = Model(inputs=inputs,outputs= outputs, name="ten outputs")
        # return the constructed network architecture
        
        return model
      
class DeepModel:
    
    def build_a_branch(inputs, numclasses =26, kernel_size = (20,11), finalAct="softmax", name = None):
        # CONV => RELU => POOL
        x = Conv2D(filters=32,kernel_size= kernel_size,padding='same')(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(32, kernel_size, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(32, (20, 11), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(64, (5, 5), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(numclasses)(x)
        x = Activation(finalAct, name=name)(x)

        return x

    @staticmethod
    def build(width = 200, height = 20, numclasses =26, finalAct="softmax"):
        
        inputShape = (height, width, 1)
        inputs = Input(shape=inputShape)
        
        first  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'first')
        second = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'second')
        third  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'third')
        fourth = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'fourth')
        fifth  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'fifth')
        sixth  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'sixth')
        seventh= MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'seventh')
        eighth = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'eighth')
        nineth = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'nineth')
        tenth  = MultipleOutputs.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'tenth')
        outputs=[first, second, third, fourth, fifth, sixth, seventh, eighth, nineth, tenth]
        model = Model(inputs=inputs,outputs= outputs, name="ten outputs")
        # return the constructed network architecture
        
        return model
