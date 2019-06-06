# inspired from https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
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
        x = Conv2D(32, kernel_size, padding="same")(x)
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
    # a deeper/longer branch to handle letters in the middle
    def build_a_deep_branch(inputs, numclasses =26, kernel_size = (20,11), kernel_size_deep = (3,3), finalAct="softmax", name = None):
        
        # CONV => RELU => POOL
        x = Conv2D(32, kernel_size =kernel_size, padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        # (CONV => RELU) * 2 => POOL
        x = Conv2D(64, kernel_size =kernel_size, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size =kernel_size, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # (CONV => RELU) * 2 => POOL
        x = Conv2D(128, kernel_size =kernel_size_deep, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, kernel_size =kernel_size_deep, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # define a branch of output layers for the number of different
        # clothing categories (i.e., shirts, jeans, dresses, etc.)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(numclasses)(x)
        x = Activation(finalAct, name=name)(x)

        # return the category prediction sub-network
        return x


    @staticmethod
    def build(width = 200, height = 20, numclasses =26, kernel_size = (20,11), finalAct="softmax"):
        
        inputShape = (height, width, 1)
        inputs = Input(shape=inputShape)
        
        first  = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'first')
        second = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'second')
        third  = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'third')
        fourth = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'fourth')
        fifth  = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'fifth')
        sixth  = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'sixth')
        seventh= MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'seventh')
        eighth = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'eighth')
        nineth = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'nineth')
        tenth  = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'tenth')
        outputs=[first, second, third, fourth, fifth, sixth, seventh, eighth, nineth, tenth]
        model = Model(inputs=inputs,outputs= outputs, name="ten outputs")
        # return the constructed network architecture
        
        return model
     
    @staticmethod
    def build_diff_length(width = 200, height = 20, numclasses =26, kernel_size = (20,11),kernel_size_deep=(3,3), finalAct="softmax"):
        
        inputShape = (height, width, 1)
        inputs = Input(shape=inputShape)
        
        first  = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size,finalAct=finalAct, name = 'first')
        second = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size,finalAct=finalAct, name = 'second')
        third  = MediumModel.build_a_deep_branch(inputs, numclasses = numclasses, kernel_size= kernel_size,\
                kernel_size_deep = kernel_size_deep, finalAct=finalAct, name = 'third')
        fourth = MediumModel.build_a_deep_branch(inputs, numclasses = numclasses, kernel_size= kernel_size,\
                kernel_size_deep = kernel_size_deep, finalAct=finalAct, name = 'fourth')
        fifth  = MediumModel.build_a_deep_branch(inputs, numclasses = numclasses, kernel_size= kernel_size,\
                kernel_size_deep = kernel_size_deep, finalAct=finalAct, name = 'fifth')
        sixth  = MediumModel.build_a_deep_branch(inputs, numclasses = numclasses, kernel_size= kernel_size,\
                kernel_size_deep = kernel_size_deep, finalAct=finalAct, name = 'sixth')
        seventh= MediumModel.build_a_deep_branch(inputs, numclasses = numclasses, kernel_size= kernel_size,\
                kernel_size_deep = kernel_size_deep, finalAct=finalAct, name = 'seventh')
        eighth = MediumModel.build_a_deep_branch(inputs, numclasses = numclasses, kernel_size= kernel_size,\
                kernel_size_deep = kernel_size_deep, finalAct=finalAct, name = 'eighth')
        nineth = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'nineth')
        tenth  = MediumModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'tenth')
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
        x = Conv2D(32, kernel_size, padding="same")(x)
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
    def build(width = 200, height = 20, numclasses =26, kernel_size = (20,11),  finalAct="softmax"):
        
        inputShape = (height, width, 1)
        inputs = Input(shape=inputShape)
        
        first  = DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'first')
        second = DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'second')
        third  = DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'third')
        fourth = DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'fourth')
        fifth  = DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'fifth')
        sixth  = DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'sixth')
        seventh= DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'seventh')
        eighth = DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'eighth')
        nineth = DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'nineth')
        tenth  = DeepModel.build_a_branch(inputs, numclasses = numclasses, kernel_size= kernel_size, finalAct=finalAct, name = 'tenth')
        outputs=[first, second, third, fourth, fifth, sixth, seventh, eighth, nineth, tenth]
        model = Model(inputs=inputs,outputs= outputs, name="ten outputs")
        # return the constructed network architecture
        
        return model
