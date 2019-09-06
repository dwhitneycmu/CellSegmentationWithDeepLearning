# -*- coding: utf-8 -*-
"""
Last updated on 6/26/2019

@author: dwhitney
"""

from logger import logger, updateDictionary
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Conv2D, Concatenate, Cropping2D, Dropout, Input, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import Activation, LeakyReLU, PReLU, ReLU, Softmax, ThresholdedReLU
import tensorflow.keras.metrics as tf_metrics
from neuralNetworkModel import NeuralNetworkModel
from tensorflow.keras.optimizers import Adam

# DEFAULT VALUES
ENTRY_LAYER = 0
DEFAULT_UNET_PARAMETERS = {} # DEFAULT PARAMETER FILE FOR MODEL
DEFAULT_UNET_PARAMETERS.update({'nConvBlocks': 5})   # N-Conv block layers
DEFAULT_UNET_PARAMETERS.update({'nConvLayers': 2})   # N-Conv layers per block
DEFAULT_UNET_PARAMETERS.update({'nConvFilters': 64})      # N-Conv feature maps per convolutional layer
DEFAULT_UNET_PARAMETERS.update({'convKernelSize': [3,3]}) # Convolutional kernel size
DEFAULT_UNET_PARAMETERS.update({'convStride': (1,1)})     # Stride for convolutional filter kernel
DEFAULT_UNET_PARAMETERS.update({'convPadding': 'same'})   # Padding type for convolutional filter kernel --> 'same' or 'valid'
DEFAULT_UNET_PARAMETERS.update({'maxPoolKernel': [2,2]}) # Max pooling kernel size
DEFAULT_UNET_PARAMETERS.update({'maxPoolStride': 2})     # Max pooling stride
DEFAULT_UNET_PARAMETERS.update({'kernelInitializer': 'he_normal'}) # Weight initialization function
DEFAULT_UNET_PARAMETERS.update({'activateFunc': 'LeakyReLU'}) # Activation function for conv layers. Can be leaky_relu, relu, sigmoid, softmax, tanh
DEFAULT_UNET_PARAMETERS.update({'outputFunc': 'sigmoid'})  # Output layer activation function
DEFAULT_UNET_PARAMETERS.update({'imgSize': (512,512,1)}) # Input image size
DEFAULT_UNET_PARAMETERS.update({'dropoutProb': 0.1}) # Dropout probability
DEFAULT_UNET_PARAMETERS.update({'useBatchNormalization': True}) # Was False. Boolean flag to include a batch normalization layer at the end of each dense layer
DEFAULT_UNET_PARAMETERS.update({'optimizer': 'Adam'}) # Training optimizer. Override with object if specific parameters like learning rate or exponential decay function want to be optimized.
DEFAULT_UNET_PARAMETERS.update({'loss': 'binary_crossentropy'}) # Loss function to minimize during training
DEFAULT_UNET_PARAMETERS.update({'metrics': ['BinaryAccuracy', 'Precision', 'Recall']}) # Metrics to monitor during training

def ActivationLayer(activationFunc,alpha=0.3,theta=1.0):
    "Returns an activation function layer for the convolutional layers"
    if(activationFunc == 'LeakyReLU'):
        layer = LeakyReLU(alpha=alpha)
    elif(activationFunc == 'PReLU'):
        layer = PReLU()
    elif(activationFunc =='ReLU'):
        layer = ReLU()
    elif(activationFunc =='Softmax'):
        layer = Softmax()
    elif(activationFunc=='ThresholdedReLU'):
        layer = ThresholdedReLU(theta=theta)
    else: #Use Tanh
        layer = Activation('tanh')
    return layer

def ConvBlockLayers(filters,pars=DEFAULT_UNET_PARAMETERS):
    "Returns convolutional block layers of the UNet model"
    blockLayers = []
    for convFilter in range(pars['nConvLayers']):
        blockLayers.append(Conv2D(filters=filters, kernel_size=pars['convKernelSize'],
                                  strides=pars['convStride'], padding=pars['convPadding'],
                                  activation=None, kernel_initializer=pars['kernelInitializer']))
        blockLayers.append(ActivationLayer(pars['activateFunc']))
    blockLayers.append(Dropout(pars['dropoutProb']))
    if(pars['useBatchNormalization']):
        blockLayers.append(BatchNormalization(axis=1))
    return blockLayers

def UpConvLayers(filters,pars=DEFAULT_UNET_PARAMETERS):
    "Return upsampling 2d convolution layers for UNet model"
    blockLayers = []
    blockLayers.append(UpSampling2D(size=pars['maxPoolKernel']))
    blockLayers.append(Conv2D(filters=filters, kernel_size=pars['maxPoolKernel'],
                              strides=pars['convStride'], padding='same',
                              activation=None, kernel_initializer=pars['kernelInitializer']))
    blockLayers.append(ActivationLayer(pars['activateFunc']))
    return blockLayers

def UNetArchitecture(pars=DEFAULT_UNET_PARAMETERS):
    # Setup downsampling blocks
    downsampleBlockLayers = []
    for i in range(pars['nConvBlocks']):
        # Downsampling convolutional block
        nFilters        = (2**i)*pars['nConvFilters']
        convBlockLayer  = ConvBlockLayers(nFilters,pars)
        maxPoolingLayer = MaxPool2D(pool_size=pars['maxPoolKernel'], strides=pars['maxPoolStride'], padding='valid')
        downsampleBlockLayers.append(convBlockLayer+[maxPoolingLayer])
    downsampleBlockLayers[-1].pop() # Swap maxpooling layer with an upsampling layer
    downsampleBlockLayers[-1]=downsampleBlockLayers[-1]+UpConvLayers(int(nFilters/2),pars)

    # Setup upsampling blocks
    nUpsamplingBlocks   = pars['nConvBlocks']-1
    upsampleBlockLayers = []
    for i in range(nUpsamplingBlocks):
        nFilters         = (2**(pars['nConvBlocks']-2-i))*pars['nConvFilters']
        convBlockLayer   = ConvBlockLayers(nFilters,pars)
        upsamplingLayers = UpConvLayers(int(nFilters/2),pars)
        upsampleBlockLayers.append(convBlockLayer+upsamplingLayers)
    upsampleBlockLayers[-1].pop() # Swap upsampling layers with final Conv2d layer
    upsampleBlockLayers[-1].pop()
    upsampleBlockLayers[-1][-1] = Conv2D(filters=1, kernel_size=(1,1), activation=pars['outputFunc'],
                                         strides=pars['convStride'], padding=pars['convPadding'], kernel_initializer=pars['kernelInitializer'])

    # Connect all downsampling and upsampling block layers using the functional API approach
    modelLayers = [[Input(pars['imgSize'])]]
    inputLayer  = modelLayers[0][0]
    for block in downsampleBlockLayers: # Connecting downsampling layers is just sequential connections
        blockLayer = []
        for layer in block:
            layer = layer(inputLayer)
            inputLayer = layer
            blockLayer.append(layer)
        modelLayers.append(blockLayer)
    for (i,block) in enumerate(upsampleBlockLayers):
        blockLayer = []
        for (j,layer) in enumerate(block):
            if(j==ENTRY_LAYER):
                # Get skip connection layer
                skipConnectionIndex = pars['nConvBlocks']-1-i
                skipConnectionLayer = modelLayers[skipConnectionIndex][-2] # Get the last block layer element before max pooling

                # Crop skip connection layer to match input layer
                skipSize   = np.array([x.value for x in skipConnectionLayer.shape])
                inputSize  = np.array([x.value for x in inputLayer.shape])
                cropSize   = (skipSize[1:3]-inputSize[1:3])/2.
                cropMatrix = [tuple((i+np.mod(i,2)*np.array([1,-1])).astype('int')) for i in cropSize] # Complicated cropping matrix to account for situations where we don't get whole numbers for the cropping
                croppedConnectionLayer = Cropping2D(cropping=cropMatrix)(skipConnectionLayer)

                # Concatenate input and skip connection layers and connect to current upsampling layer
                inputLayer = Concatenate(axis=3)([inputLayer,croppedConnectionLayer])
            layer = layer(inputLayer)
            inputLayer = layer
            blockLayer.append(layer)
        modelLayers.append(blockLayer)
    modelLayers = [layer for blockLayer in modelLayers for layer in blockLayer]
    return modelLayers

class UNet(NeuralNetworkModel):
    def __init__(self,modelLayers=[],modelParameters={}):
        self.parameters = updateDictionary(modelParameters,DEFAULT_UNET_PARAMETERS)
        self.setupModel(modelLayers)
        self.compileModel()
        self.logger = logger()

    def setupModel(self,modelLayers=[]):
        # Setup UNet model
        if(len(modelLayers)==0):
            modelLayers = UNetArchitecture(self.parameters)
        self.model = keras.Model(inputs=modelLayers[0], outputs=modelLayers[-1])

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
    model = UNet()
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