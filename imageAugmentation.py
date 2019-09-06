# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 07:58:43 2019

@author: dwhitney
"""

from skimage.filters import gaussian
from logger import updateDictionary
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

DEFAULT_GENERATOR_OPTIONS = {} # DEFAULT TRAINING OPTIONS. 
DEFAULT_GENERATOR_OPTIONS.update({'seed': 1})        # RNG seed
DEFAULT_GENERATOR_OPTIONS.update({'augment_rescaling':     1}) # Rescale responses
DEFAULT_GENERATOR_OPTIONS.update({'augment_angleRange':    0}) # Range of rotation angles 
DEFAULT_GENERATOR_OPTIONS.update({'augment_shiftRange':    0.1}) # Fraction of total img width/height for (X,Y) shifts
DEFAULT_GENERATOR_OPTIONS.update({'augment_shearRange':    0.1}) # Shearing range
DEFAULT_GENERATOR_OPTIONS.update({'augment_zoomRange':     [0.8,1.25]}) # Zoom range: [lower, upper]
DEFAULT_GENERATOR_OPTIONS.update({'augment_brightRange':   [0.1,1]}) # Brightness range: [lower, upper]. Range for picking the new brightness (relative to the original image), with 1.0 being the original's brightness. Lower must not be smaller than 0.
DEFAULT_GENERATOR_OPTIONS.update({'augment_flipHoriz':     False}) # Boolean flag for horizontal flip
DEFAULT_GENERATOR_OPTIONS.update({'augment_flipVert':      False}) # Boolean flag for vertical flip
DEFAULT_GENERATOR_OPTIONS.update({'augment_fillMode':      'reflect'}) # Points outside image boundary are filled with: {"constant", "nearest", "reflect" or "wrap"}
DEFAULT_GENERATOR_OPTIONS.update({'augment_fillVal':       0}) # If augment_fillMode=='constant', points outside image boundary are filled with fillVal
DEFAULT_GENERATOR_OPTIONS.update({'augment_gaussianNoiseLevel': 0.0}) # Add gaussian noise with a strength factor from 0 to 1
DEFAULT_GENERATOR_OPTIONS.update({'augment_highpassCutOff': 100}) # High pass cutoff value. If zero, then ignored
DEFAULT_GENERATOR_OPTIONS.update({'augment_lowpassCutOff': 0}) # Low pass cutoff value. If zero, then ignored

def addGaussianNoise(img,noiseFactor):
    noiseImg = np.random.randn(img.shape[0],img.shape[1])
    return noiseFactor*(img.std()*noiseImg+img.mean())

class preProcessFunction():
    def __init__(self,options={}):
        self.options = options
        self.setup()
            
    def setup(self,options={}):
        """Specify the preprocessing options to use with the Image Augmentation class""" 
        self.options = updateDictionary(self.options,DEFAULT_GENERATOR_OPTIONS)
        
    def transform(self,imgs):
        for index in range(imgs.shape[0]):
            img = imgs[index,:,:,:]
            """Transform each image"""
            if self.options['augment_highpassCutOff']>0:
                img = img-gaussian(img,sigma=self.options['augment_highpassCutOff'])
            if self.options['augment_lowpassCutOff']>0:
                img = gaussian(img,sigma=self.options['augment_lowpassCutOff'])
            if self.options['augment_gaussianNoiseLevel']>0:    
                img = img+addGaussianNoise(img,self.options['augment_gaussianNoise'])
            imgs[index,:,:,:] = img
            return imgs

class ImageAugmentation():
    """Superclass for setting up data augmentation of imaging data for training/testing with TensorFlow.
    See ImageAugmentationGenerator for use with a Keras Generator and ImageAugmentationSequence for use
    with a Keras Sequence"""
    def __init__(self, options={}):
        self.options = options
        self.setup()

    def setup(self, options={}):
        """Initialize a data augmentation generator through Keras"""

        # Update options dictionary with input from user, else use DEFAULT_TRAINING_OPTIONS
        self.options = updateDictionary(self.options, DEFAULT_GENERATOR_OPTIONS)

        # Initialize generator
        self.extraProcessing = preProcessFunction(
            self.options)  # Adds spatial filtering and gaussian noise, but these image operations should not be performed on the mask
        self.dataGenerator = ImageDataGenerator(rescale=self.options['augment_rescaling'],
                                                rotation_range=self.options['augment_angleRange'],
                                                width_shift_range=self.options['augment_shiftRange'],
                                                height_shift_range=self.options['augment_shiftRange'],
                                                shear_range=self.options['augment_shearRange'],
                                                zoom_range=self.options['augment_zoomRange'],
                                                brightness_range=self.options['augment_brightRange'],
                                                horizontal_flip=self.options['augment_flipHoriz'],
                                                vertical_flip=self.options['augment_flipVert'],
                                                fill_mode=self.options['augment_fillMode'],
                                                cval=self.options['augment_fillVal'])

class ImageAugmentationGenerator(ImageAugmentation):
    "Class for setting up data augmentation using a Keras generator"
    def __init__(self,options={}):
        self.options = options
        self.setup()

    def get(self,image_data,image_masks,batch_size,seed):
        """Returns a generator that applies to the image and mask data """

        imageDataGenerator  = self.dataGenerator.flow(image_data ,batch_size=batch_size,seed=seed)
        imageMasksGenerator = self.dataGenerator.flow(image_masks,batch_size=batch_size,seed=seed)
        while True:
            imgs  = imageDataGenerator.next()
            imgs  = self.extraProcessing.transform(imgs)
            masks = imageMasksGenerator.next()>0
            yield (imgs,masks)

class ImageAugmentationSequence(ImageAugmentation,Sequence):
    "Class for setting up data augmentation using a Keras sequence"
    def __init__(self,options={}):
        self.options = options
        self.setup()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x,batch_y

    def generate(self,image_data,image_masks,batch_size,seed):
        """Generates an internal set of augmented images and mask data """
        nImgs = image_data.shape[0]
        self.batch_size = batch_size
        augmentedData   = self.dataGenerator.flow(image_data , batch_size=nImgs, seed=seed).next()
        augmentedMasks  = self.dataGenerator.flow(image_masks, batch_size=nImgs, seed=seed).next()
        self.x = self.extraProcessing.transform(augmentedData)
        self.y = augmentedMasks>0