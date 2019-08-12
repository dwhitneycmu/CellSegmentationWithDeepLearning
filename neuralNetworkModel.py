# -*- coding: utf-8 -*-
"""
Last updated on 6/26/2019

@author: dwhitney
"""

from logger import logger, updateDictionary
import matplotlib.pyplot as plt
import numpy as np
import os
from PCA import PCA
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# DEFAULT VALUES
DEFAULT_MODEL_PATH = os.path.join(os.getcwd(),'temp_model.h5')
DEFAULT_MODEL_PARAMETERS = {} # DEFAULT PARAMETER FILE FOR MODEL
DEFAULT_MODEL_PARAMETERS.update({'nDenseLayers': 3})  # N-Dense layers
DEFAULT_MODEL_PARAMETERS.update({'nDenseUnits': 128}) # N-units per dense layer
DEFAULT_MODEL_PARAMETERS.update({'nOutputUnits': 10}) # N-units for output layer
DEFAULT_MODEL_PARAMETERS.update({'activationFunc': tf.nn.relu}) # Dense layer activation function
DEFAULT_MODEL_PARAMETERS.update({'outputFunc': tf.nn.softmax}) # Output layer activation function
DEFAULT_MODEL_PARAMETERS.update({'imgSize': (28,28,1)}) # Input image size
DEFAULT_MODEL_PARAMETERS.update({'dropoutProb': 0.0}) # Dropout probability
DEFAULT_MODEL_PARAMETERS.update({'useBatchNormalization': False}) # Boolean flag to include a batch normalization layer at the end of each dense layer
DEFAULT_MODEL_PARAMETERS.update({'optimizer': 'adam'}) # Training optimizer. Override with object if specific parameters like learning rate or exponential decay function want to be optimized.
DEFAULT_MODEL_PARAMETERS.update({'loss': 'sparse_categorical_crossentropy'}) # Loss function to minimize during training
DEFAULT_MODEL_PARAMETERS.update({'metrics': ['accuracy']}) # Metrics to monitor during training
DEFAULT_TRAINING_OPTIONS = {} # DEFAULT TRAINING OPTIONS. 
DEFAULT_TRAINING_OPTIONS.update({'epochs': 5})    # Training epochs
DEFAULT_TRAINING_OPTIONS.update({'batchSize': 0.2}) # Proportion of training set used per epoch (0 to 1)
DEFAULT_TRAINING_OPTIONS.update({'seed': 1})        # RNG seed
DEFAULT_TRAINING_OPTIONS.update({'showTrainingData': True}) # Show a subset of training data
DEFAULT_TRAINING_OPTIONS.update({'subplotDims': [1,1]}) # [5,5]: If showTrainingData==True, then the number of subplots shown are a multiple of (nRows,nCols)==>nSubplots=(nCols)x(nRows)
DEFAULT_TRAINING_OPTIONS.update({'augmentData': True}) # Boolean flag to augment imaging data
DEFAULT_TRAINING_OPTIONS.update({'augment_angleRange':    0}) # Range of rotation angles 
DEFAULT_TRAINING_OPTIONS.update({'augment_shiftRange':    0.1}) # Fraction of total img width/height for (X,Y) shifts
DEFAULT_TRAINING_OPTIONS.update({'augment_shearRange':    0.1}) # Shearing range
DEFAULT_TRAINING_OPTIONS.update({'augment_zoomRange':     [0.8,1.25]}) # Zoom range: [lower, upper]
DEFAULT_TRAINING_OPTIONS.update({'augment_brightRange':   [0.1,1]}) # Brightness range: [lower, upper]. Range for picking the new brightness (relative to the original image), with 1.0 being the original's brightness. Lower must not be smaller than 0.
DEFAULT_TRAINING_OPTIONS.update({'augment_flipHoriz':     False}) # Boolean flag for horizontal flip
DEFAULT_TRAINING_OPTIONS.update({'augment_flipVert':      False}) # Boolean flag for vertical flip
DEFAULT_TRAINING_OPTIONS.update({'augment_fillMode':      'constant'}) # Points outside image boundary are filled with: {"constant", "nearest", "reflect" or "wrap"}
DEFAULT_TRAINING_OPTIONS.update({'augment_fillVal':       0}) # If augment_fillMode=='constant', points outside image boundary are filled with fillVal
DEFAULT_TRAINING_OPTIONS.update({'augment_zca_whitening': False}) # Apply ZCA whitening   

# Define the default model: a fully, connected dense layer model
def defaultModel(parameters=DEFAULT_MODEL_PARAMETERS):
    modelLayers = []
    modelLayers.append(keras.layers.Flatten(input_shape=parameters['imgSize']))
    for n in range(parameters['nDenseLayers']):
        modelLayers.append(keras.layers.Dense(parameters['nDenseUnits'], activation=parameters['activationFunc']))
        modelLayers.append(keras.layers.Dropout(parameters['dropoutProb']))
        if(parameters['useBatchNormalization']):
            modelLayers.append(keras.layers.BatchNormalization(axis=1))
    modelLayers.append(keras.layers.Dense(parameters['nOutputUnits'], activation=parameters['outputFunc']))
    return modelLayers

# Helper functions to visualize/modify model layers
def showAndModifyNetworkLayers(func):
    def loopThroughModelLayer(model,inputDict={}):
        messages = []
        messages.append("Model layers: ")
        for index,layer in enumerate(model.layers):
            messages.append(func(layer,index,inputDict)) # Function must return a string value. Necessary to get data saved to a logger function without refactoring for multiple use cases
    return loopThroughModelLayer

@showAndModifyNetworkLayers
def getLayerStats(layer,index,inputDict):
    return("Layer {} ({}): Size: {}".format(index,layer.name,layer.output_shape))

@showAndModifyNetworkLayers
def getLayerDimensionality(layer,index,inputDict):
    if layer.name.startswith(inputDict['ValidDimLayers']): # Must be either a single string or tuple of strings
        pcaObj = PCA(matrix=layer.get_weights()[0])
        message = "Layer {}: Dimensionality: {:4.2f}".format(index,pcaObj.dimensionality())
    else:
        message = "Layer {}: Dimensonality is N/A".format(index)
    return message        

@showAndModifyNetworkLayers
def reduceLayerDimensionality(layer,index,inputDict):
    if layer.name.startswith(inputDict['ValidDimLayers']): # Must be either a single string or tuple of strings
        weights = layer.get_weights()
        pcaObj  = PCA(matrix=weights[0])
        weights[0] = pcaObj.filterMatrix(n=pcaObj.computeTargetPCs(targetRatio=inputDict['targetRatio']))
        layer.set_weights(weights)   
        message = "Layer {}: Reduced layer dimensionality".format(index)
    else:
        message = "Layer {}: Dimensionality unchanged".format(index)
    return(message)

class NeuralNetworkModel:
    def __init__(self,modelLayers=[],modelParameters={}):
        self.parameters = updateDictionary(modelParameters,DEFAULT_MODEL_PARAMETERS)
        self.setupModel(modelLayers)
        self.logger = logger()       
#    def __del__(self):
#        del self.logger
    
    def updateParameters(self,modelParameters=DEFAULT_MODEL_PARAMETERS):
        # Update model parameters, or reinitializes to default values
        self.parameters = updateDictionary(modelParameters,self.parameters)
    
    def setupModel(self,modelLayers=[],modelParameters={}):
        # Setup model 
        self.parameters = updateDictionary(modelParameters,DEFAULT_MODEL_PARAMETERS)
        if(len(modelLayers)==0):
            modelLayers = defaultModel(parameters=self.parameters)
        self.model = keras.Sequential(modelLayers)
        self.model.compile(optimizer=self.parameters['optimizer'], loss=self.parameters['loss'], metrics=self.parameters['metrics'])
        
    def trainModel(self, train_data, train_labels, test_data, test_labels, options=DEFAULT_TRAINING_OPTIONS):
        "Training the model on data"

        # Ensure that the data shape is rank 4: (Images)x(X)x(Y)x(NChannels)
        if(len(train_data.shape)==3): # If data is not RGB (rank 4), then make rank 4 with last dimension of size 1
            train_data = train_data.reshape((train_data.shape+(1,)))
            test_data  = test_data.reshape((test_data.shape+(1,)))

        # Update options dictionary with input from user, else use DEFAULT_TRAINING_OPTIONS
        options = updateDictionary(options,DEFAULT_TRAINING_OPTIONS)
        self.trainingOptions = options

        # Setup training parameters        
        nAllSamples   = train_labels.shape[0]
        nTrainSamples = np.ceil(options['batchSize']*nAllSamples).astype('int')
        np.random.seed(options['seed'])
        
        # Data augmentation of imaging data (optional)
        if(options['augmentData']):
            dataGen = ImageDataGenerator(rotation_range     = options['augment_angleRange'], \
                                         width_shift_range  = options['augment_shiftRange'], \
                                         height_shift_range = options['augment_shiftRange'], \
                                         shear_range        = options['augment_shearRange'], \
                                         zoom_range         = options['augment_zoomRange'],  \
                                         brightness_range   = options['augment_brightRange'],\
                                         horizontal_flip    = options['augment_flipHoriz'],  \
                                         vertical_flip      = options['augment_flipVert'],   \
                                         fill_mode          = options['augment_fillMode'],   \
                                         cval               = options['augment_fillVal'],    \
                                         zca_whitening      = options['augment_zca_whitening'])
            if(options['augment_zca_whitening']):
                dataGen.fit(train_data)
                
        # Visualize sample training data during each training epoch (optional)
        if(options['showTrainingData']):
            fig,axes=plt.subplots(nrows=options['subplotDims'][0],ncols=options['subplotDims'][1],figsize=(15,15))
            if isinstance(axes,np.ndarray):
                axes=axes.flatten()
            else:
                axes=[axes]
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
        
        self.logger.write("******************")
        self.logger.write("Training model: {}".format(type(self)))
        self.logger.write("Options: {}".format(options))
        self.logger.write("******************")
        t0 = time.time()
        modelAccuracy = np.zeros((options['epochs'],2))
        for training_epoch in range(options['epochs']):
            self.logger.write("Training epoch: {}/{}".format(training_epoch+1,options['epochs']))
    
            # A subset of training imaging samples, an amount specified by nTrainSamples,
            # are selected from the full training dataset and then optionally augmented
            trainingSamples = np.random.permutation(nAllSamples)[:nTrainSamples]
            trainingData  = train_data[trainingSamples,:,:,:]
            trainLabels   = train_labels[trainingSamples,:,:,:]
            testingData   = test_data
            testingLabels = test_labels
            if(options['augmentData']):
                seed = training_epoch;
                trainingData  = dataGen.flow(trainingData , batch_size=trainingData.shape[0],seed=seed).next()
                trainLabels   = dataGen.flow(trainLabels  , batch_size=trainingData.shape[0],seed=seed).next()/255.
                testingData   = dataGen.flow(testingData  , batch_size=testingData.shape[0],seed=seed+1).next()
                testingLabels = dataGen.flow(testingLabels, batch_size=testingData.shape[0],seed=seed+1).next()/255.

            # Train model using the minibatch approach (with a size equal to nTrainSamples)
            self.model.fit(trainingData, trainLabels, batch_size=2, epochs=10)
    
            # Show some example images (optional)
            if(options['showTrainingData']):
                for i,ax in enumerate(axes):
                    ax.imshow(trainLabels[i,:,:,0], cmap='gray') #plt.cm.binary
                plt.show()
                plt.draw() 
                plt.pause(0.5)
    
            # Evaluate model accuracy
            train_accuracy = self.model.history.history['acc'][0]
            (_, test_accuracy)  = self.model.evaluate(testingData, testingLabels)
            modelAccuracy[training_epoch,:]=[train_accuracy,test_accuracy]
            self.logger.write("Model accuracy: {} (training) / {} (test)\n".format(train_accuracy,test_accuracy))
        elapsedTime = time.time()-t0
        self.logger.write('Total elapsed training time: {:4.2f}s (or {:4.2f}s per epoch)'.format(elapsedTime,elapsedTime/options['epochs']))
        return(modelAccuracy)

    def loadModel(self,file_path=DEFAULT_MODEL_PATH):
        # Load a keras model file (*.h5 format) to the specified filePath
        self.model = keras.models.load_model(file_path) 
        
    def saveModel(self,file_path=DEFAULT_MODEL_PATH):
        # Save a keras model file (*.h5 format) to the specified file_path
        self.model.save(file_path)
        self.file_path = file_path
        pass
    
    def showLayerStats(self):
        # Show layer types and sizes
        messages = getLayerStats(self.model)
        for message in messages:
            self.logger.write(message)
        
    def showLayerDimensionality(self,validDimLayers=('dense')):
        # Show the dimensionality of each layer's weights
        inputDict = {'validDimLayers':validDimLayers}
        messages = getLayerDimensionality(self.model,inputDict)
        for message in messages:
            self.logger.write(message)
             
    def reduceLayerDimensionality(self,targetRatio=0.5,validDimLayers=('dense')):
        # Reduce the dimensionality of each layer's weights
        inputDict = {'validDimLayers':validDimLayers,'targetRatio':targetRatio} 
        messages = reduceLayerDimensionality(self.model,inputDict)  
        for message in messages:
            self.logger.write(message)
        
if __name__ == "__main__":
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
    model = NeuralNetworkModel()
    accuracy = model.trainModel(train_images, train_labels, test_images, test_labels)
    model.logger.close()
    
    # Show accuracy model
    fig,ax=plt.subplots()
    ax.plot(accuracy)
    ax.set_ylim([0.0,1])
    ax.set_ylabel('Classification accuracy')
    ax.set_xlabel('Training epoch')
    ax.legend(['Training data','Test data'])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))