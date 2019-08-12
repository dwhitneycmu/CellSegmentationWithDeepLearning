# -*- coding: utf-8 -*-
"""
Last updated on 6/26/2019

@author: dwhitney
"""

from logger import logger, updateDictionary
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPool2D, Reshape
from neuralNetworkModel import NeuralNetworkModel

# DEFAULT VALUES
DEFAULT_MODEL_PARAMETERS = {} # DEFAULT PARAMETER FILE FOR MODEL
DEFAULT_MODEL_PARAMETERS.update({'nConvLayers': 3})        # N-Conv layers
DEFAULT_MODEL_PARAMETERS.update({'nConvFilters': 32})      # N-Conv feature maps per convolutional layer
DEFAULT_MODEL_PARAMETERS.update({'convKernelSize': [5,5]}) # Convolutional kernel size
DEFAULT_MODEL_PARAMETERS.update({'convStride': (1,1)})     # Stride for convolutional filter kernel
DEFAULT_MODEL_PARAMETERS.update({'convPadding': 'same'})   # Padding type for convolutional filter kernel
DEFAULT_MODEL_PARAMETERS.update({'maxPoolKernel': [2,2]}) # Max pooling kernel size
DEFAULT_MODEL_PARAMETERS.update({'maxPoolStride': 2})     # Max pooling stride
DEFAULT_MODEL_PARAMETERS.update({'nDenseLayers': 3})   # N-Dense layers
DEFAULT_MODEL_PARAMETERS.update({'nDenseUnits': 1024}) # N-units per dense layer
DEFAULT_MODEL_PARAMETERS.update({'nOutputUnits': 10})  # N-units for output layer
DEFAULT_MODEL_PARAMETERS.update({'kernelInitializer': 'he_normal'}) # Weight initialization function
DEFAULT_MODEL_PARAMETERS.update({'activationFunc': 'relu'}) # Conv/Dense layer activation function
DEFAULT_MODEL_PARAMETERS.update({'outputFunc': 'softmax'})  # Output layer activation function
DEFAULT_MODEL_PARAMETERS.update({'imgSize': (28,28,1)}) # Input image size
DEFAULT_MODEL_PARAMETERS.update({'dropoutProb': 0.0}) # Dropout probability
DEFAULT_MODEL_PARAMETERS.update({'useBatchNormalization': False}) # Boolean flag to include a batch normalization layer at the end of each dense layer
DEFAULT_MODEL_PARAMETERS.update({'optimizer': 'adam'}) # Training optimizer
DEFAULT_MODEL_PARAMETERS.update({'loss': 'sparse_categorical_crossentropy'}) # Loss function to minimize during training
DEFAULT_MODEL_PARAMETERS.update({'metrics': ['accuracy']}) # Metrics to monitor during training

def ConvNetArchitecture(pars=DEFAULT_MODEL_PARAMETERS):
    modelLayers = []
    # Initial Convolutional layers
    modelLayers.append(InputLayer(pars['imgSize']))
    for n in range(pars['nConvLayers']):
        # Define sequential layers in each convolutional layer block
        convLayer = Conv2D(filters=pars['nConvFilters'], kernel_size=pars['convKernelSize'], \
                           strides=pars['convStride'], padding=pars['convPadding'], \
                           activation=pars['activationFunc'], kernel_initializer=pars['kernelInitializer'], \
                           data_format="channels_first")
        dropoutLayer    = Dropout(pars['dropoutProb'])
        maxPoolingLayer = MaxPool2D(pool_size=pars['maxPoolKernel'], strides=pars['maxPoolStride'], padding='valid')
        blockLayers     = [convLayer,dropoutLayer,maxPoolingLayer]
        if(pars['useBatchNormalization']):
            blockLayers.append(BatchNormalization(axis=1))
        modelLayers = modelLayers+blockLayers
        
    # Add Dense layers
    modelLayers.append(Flatten(data_format="channels_first"))
    for n in range(pars['nDenseLayers']):
        # Define sequential layers in each dense layer block
        denseLayer   = Dense(pars['nDenseUnits'], activation=pars['activationFunc'], \
                             kernel_initializer=pars['kernelInitializer'])
        dropoutLayer = Dropout(pars['dropoutProb'])
        blockLayers  = [denseLayer,dropoutLayer]
        if(pars['useBatchNormalization']):
            blockLayers.append(BatchNormalization(axis=1))
        modelLayers = modelLayers+blockLayers
    modelLayers.append(keras.layers.Dense(pars['nOutputUnits'], activation=pars['outputFunc']))
    return modelLayers

class ConvNet(NeuralNetworkModel):
    def __init__(self,modelLayers=[],modelParameters={}):
        self.parameters = updateDictionary(modelParameters,DEFAULT_MODEL_PARAMETERS)
        self.setupModel(modelLayers)
        self.logger = logger()
        
    def __del__(self):
        del self.logger
        
    def setupModel(self,modelLayers=[],modelParameters={}):
        # Setup model 
        pars = updateDictionary(modelParameters,DEFAULT_MODEL_PARAMETERS)
        self.parameters = pars
        if(len(modelLayers)==0):
            modelLayers = ConvNetArchitecture(self.parameters)
        self.model = keras.Sequential(modelLayers)
        self.model.compile(optimizer=self.parameters['optimizer'], loss=self.parameters['loss'], metrics=self.parameters['metrics'])
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Load MNIST FASHION DATASET
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images  = test_images  / 255.0
    
    # Show some example images
    plt.figure(figsize=(15,15))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
    
    # Setup and train model
    model = ConvNet()
    accuracy = model.trainModel(train_images, train_labels, test_images, test_labels)
    model.logger.close()
    
    # Show accuracy model
    fig,ax=plt.subplots()
    ax.plot(accuracy)
    ax.set_ylim([0.5,1])
    ax.set_ylabel('Classification accuracy')
    ax.set_xlabel('Training epoch')
    ax.legend(['Training data','Test data'])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))