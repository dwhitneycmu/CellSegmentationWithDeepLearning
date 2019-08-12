# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:37:28 2019

@author: dwhitney
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import imagingDataset
    from scipy import ndimage
    from UNet import UNet
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Load mouse imaging data for training neural network
    plt.close('all')
    reloadData= False
    dataFolder  = 'D:/Code/ROI Segmentation/DataSets/Neurofinder/Training Data/saved_data_expt'
    imageSets = []
    nExpts    = 5
    if reloadData:
        # Load imaging data
        for expt in range(nExpts):
            baseFolder    = 'D:/Code/ROI Segmentation/DataSets/Neurofinder/Training Data/neurofinder.0{0}.00/neurofinder.0{0}.00/'.format(expt,expt)
            imageDataPath = baseFolder+'images/'
            ROIPath       = baseFolder+'regions/regions.json'
        
            print('Loading Mouse Dataset #{0}:'.format(expt))
            imageSet=imagingDataset.imagingDataset()
            imageSet.getImagingData(imageDataPath,returnOnlyAverage=True)
            imageSet.getJSONROIs(ROIPath)
            imageSet.showROIs()
            imageSet.saveToHDF5('{0}{1}.h5'.format(dataFolder,expt))
            imageSets.append(imageSet)
    else:
        # Load imaging data
        for expt in range(nExpts):
            imageSet=imagingDataset.imagingDataset()
            imageSet.loadFromHDF5('{0}{1}.h5'.format(dataFolder,expt))
            imageSet.showROIs()
            imageSets.append(imageSet)

    # Setup training dataset
    trainExpts = [0,1,2,4]
    testExpts = [0,1,2,4]
    train_images = np.stack([imageSets[i].getReferenceImg() for i in trainExpts])
    train_labels = np.stack([imageSets[i].getLabeledROIMask() for i in trainExpts])>0
    test_images  = np.stack([imageSets[i].getReferenceImg() for i in testExpts])
    test_labels  = np.stack([imageSets[i].getLabeledROIMask() for i in testExpts])>0
    train_images = train_images.reshape(train_images.shape+(1,))
    train_labels = train_labels.reshape(train_labels.shape+(1,))
    test_images  = test_images.reshape(test_images.shape+(1,))
    test_labels  = test_labels.reshape(test_labels.shape+(1,))

    # Setup and train model
    trainModel = False;
    options=dict();
    options.update({'showTrainingData':False})
    options.update({'epochs': 100})    # Training epochs
    options.update({'batchSize': 1}) # Proportion of training set used per epoch (0 to 1)
    options.update({'augmentData': True}) # Boolean flag to augment imaging data
    options.update({'augment_angleRange':    180}) # Range of rotation angles 
    options.update({'augment_shiftRange':    0.25}) # Fraction of total img width/height for (X,Y) shifts
    options.update({'augment_shearRange':    0.25}) # Shearing range
    options.update({'augment_zoomRange':     [0.75,1.5]}) # Zoom range: [lower, upper]
    options.update({'augment_brightRange':   [0.75,1.25]}) # Brightness range: [lower, upper]. Range for picking the new brightness (relative to the original image), with 1.0 being the original's brightness. Lower must not be smaller than 0.
    options.update({'augment_flipHoriz':     True}) # Boolean flag for horizontal flip
    options.update({'augment_flipVert':      True}) # Boolean flag for vertical flip
    options.update({'augment_fillMode':      'reflect'}) # Points outside image boundary are filled with: {"constant", "nearest", "reflect" or "wrap"}
    options.update({'augment_fillVal':       0}) # If augment_fillMode=='constant', points outside image boundary are filled with fillVal
    options.update({'augment_zca_whitening': False}) # Apply ZCA whitening   
    model = UNet()
    if trainModel:
        accuracy = model.trainModel(train_images, train_labels, test_images, test_labels,options)
        model.saveModel(r'D:\Code\ROI Segmentation\Code\Dave\Unet_model2.h5')
        
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
    else:
        model.loadModel(r'D:\Code\ROI Segmentation\Code\Dave\Unet_model2.h5')
    
    # Get validation experience (ferret image and ROIs)
    from PIL import Image
    filePath=r'D:\Data Science\2018 MPFI Imaging Course\RAW Data\rawData_200xDownsampled.tif'
    img = np.zeros((512,512))
    with Image.open(filePath) as file:
        for i in range(file.n_frames):
            # Get imaging frame
            file.seek(i)
            img = img+ np.array(file)
    img = (img-np.nanmean(img.flatten()))/np.nanstd(img.flatten())
    from readImageJROIs import readImageJROIsFromZip
    ROIPath = r'D:\Data Science\2018 MPFI Imaging Course\analyzedData\RoiSet.zip'
    ROIObject = imagingDataset.imagingDataset()
    ROIObject.ROIs = readImageJROIsFromZip(ROIPath)
    ROIObject.imgAverage = img
    ROIObject.showROIs()
    ROIObject.fillROIs()
    ROIImg = ROIObject.getLabeledROIMask()
    ROIObject.showROIs()
    
    # Show prediction images
    #prediction_images = test_images;
    prediction_images = img.reshape((1,512,512,1));
    n_images=prediction_images.shape[0]
    if(options['augmentData']):
        options.update({'augment_angleRange':    0}) # Range of rotation angles 
        options.update({'augment_shiftRange':    0}) # Fraction of total img width/height for (X,Y) shifts
        options.update({'augment_shearRange':    0}) # Shearing range
        options.update({'augment_zoomRange':     [0.99,1.01]}) # Zoom range: [lower, upper]
        options.update({'augment_brightRange':   [0.95,1]}) # Brightness range: [lower, upper]. Range for picking the new brightness (relative to the original image), with 1.0 being the original's brightness. Lower must not be smaller than 0.
        options.update({'augment_flipHoriz':     False}) # Boolean flag for horizontal flip
        options.update({'augment_flipVert':      False}) # Boolean flag for vertical flip
        options.update({'augment_fillMode':      'reflect'}) # Points outside image boundary are filled with: {"constant", "nearest", "reflect" or "wrap"}
        options.update({'augment_fillVal':       0}) # If augment_fillMode=='constant', points outside image boundary are filled with fillVal
        options.update({'augment_zca_whitening': False}) # Apply ZCA whitening 
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
        predictionImages=dataGen.flow(prediction_images, batch_size=test_images.shape[0]).next()
    else:
        predictionImages=prediction_images
    predictions=model.model.predict(predictionImages)
    for index in range(n_images):
        rawImg = predictionImages[index,:,:,0]
        predictionImg  = predictions[index,:,:,0]
        filteredImg    = ndimage.median_filter(predictionImg,7)
        thresholdedImg = filteredImg>0.5
        
        fig,ax=plt.subplots(ncols=2,nrows=2);
        ax=ax.flatten()
        ax[0].imshow(rawImg,cmap='gray')
        ax[0].set_title('Raw image')
        ax[1].imshow(predictionImg,cmap='hot',vmin=0,vmax=1)
        ax[1].set_title('Mask prediction')
        ax[2].imshow(filteredImg,cmap='hot',vmin=0,vmax=1)
        ax[2].set_title('Mask prediction (Filtered)')
        ax[3].imshow(thresholdedImg,cmap='gray',vmin=0,vmax=1)
        ax[3].set_title('Thresholded mask prediction')