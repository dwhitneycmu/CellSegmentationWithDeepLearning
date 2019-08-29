# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:37:28 2019

@author: dwhitney
"""

if __name__ == "__main__":
    from imageAugmentation import ImageAugmentation
    import imagingDataset
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from scipy import ndimage
    from UNet import UNet
    from readImageJROIs import readImageJROIsFromZip
    from PIL import Image
    
    def getImagingAndROIData(imgPath,ROIPath,imgSize=(512,512)):
        # Load an unlabeled, z-projection image (*.TIF file) and z-score
        img = np.zeros(imgSize)
        with Image.open(imgPath) as file:
            for i in range(file.n_frames):
                file.seek(i)
                frame = np.array(file)
                (x,y)=np.meshgrid(range(frame.shape[1]),range(frame.shape[0]))
                img[y,x] = img[y,x] + frame/file.n_frames
        img = (img-np.nanmean(img.flatten()))/np.nanstd(img.flatten())
        
        # Load ROIs and add z-projection image
        ROIObject = imagingDataset.imagingDataset()
        ROIObject.ROIs = readImageJROIsFromZip(ROIPath)
        ROIObject.fillROIs()
        ROIObject.erodeROIs(useXOR=False,useErosion=True,kernel=np.ones((3,3)))
        ROIObject.imgAverage = img
        return ROIObject
    
    reloadData=True
    if(reloadData):
        # Load mouse imaging data for training neural network
        plt.close('all')
        dataFolder  = r'D:\Code\ROI Segmentation\DataSets\Neurofinder\Revised Data'
        imgDataSets = [os.path.join(dataFolder,f) for f in os.listdir(dataFolder) if f.endswith('.tif')];
        ROIDataSets = [os.path.join(dataFolder,f) for f in os.listdir(dataFolder) if f.endswith('.zip')];
        imageSets = []
        for (imgPath,ROIPath) in zip(imgDataSets,ROIDataSets):
            imageSet = getImagingAndROIData(imgPath,ROIPath)
            imageSets.append(imageSet)
        nMouseFoVs = len(imageSets)
            
        # Load ferret data for training neural network
        dataFolder  = r'D:\Code\ROI Segmentation\DataSets\FerretData_DaveAndJeremy'
        imgDataSets = [os.path.join(path,filename) for (path, names, filenames) in os.walk(dataFolder) for filename in filenames if filename.endswith('.tif')]
        ROIDataSets = [os.path.join(path,filename) for (path, names, filenames) in os.walk(dataFolder) for filename in filenames if filename.endswith('.zip')]
        for (imgPath,ROIPath) in zip(imgDataSets,ROIDataSets):
            imageSet = getImagingAndROIData(imgPath,ROIPath)
            imageSets.append(imageSet)
        nFerretFoVs = len(imageSets)-nMouseFoVs;
            
        # Get validation experiment (ferret image and ROIs)
        imgDataPath = r'D:\Data Science\2018 MPFI Imaging Course\RAW Data\rawData_200xDownsampled.tif'
        ROIPath     = r'D:\Data Science\2018 MPFI Imaging Course\analyzedData\RoiSet.zip'
        validationSet = getImagingAndROIData(imgDataPath,ROIPath)
    nImageSets = len(imageSets)
    nCellROIs  = sum([len(imageSet.ROIs) for imageSet in imageSets]) 
    print('Loaded {} FoVs and {} ROIs'.format(nImageSets,nCellROIs))

    # Setup training dataset
    seed  = 1;
    percentOfDataToTrain = 0.8
    randomFerretFoVs = np.random.permutation(nFerretFoVs)+nMouseFoVs
    dividingExpt     = int(np.round(percentOfDataToTrain*len(randomFerretFoVs)))
    trainExpts = [5,6,7,9,8]+randomFerretFoVs[:dividingExpt].tolist()
    testExpts  = [0,1,2,4,3]+randomFerretFoVs[dividingExpt:].tolist()
    train_images = np.stack([imageSets[i].getReferenceImg() for i in trainExpts])
    train_labels = np.stack([imageSets[i].getLabeledROIMask() for i in trainExpts])>0
    test_images  = np.stack([imageSets[i].getReferenceImg() for i in testExpts])
    test_labels  = np.stack([imageSets[i].getLabeledROIMask() for i in testExpts])>0
    train_images = train_images.reshape(train_images.shape+(1,))
    train_labels = train_labels.reshape(train_labels.shape+(1,))
    test_images  = test_images.reshape(test_images.shape+(1,))
    test_labels  = test_labels.reshape(test_labels.shape+(1,))

    # Setup and train model
    trainModel = True;
    options=dict();
    options.update({'showTrainingData':False})
    options.update({'blocks': 1})    # 10, Training blocks (Each is a full number of training epochs)
    options.update({'epochs': 50})   # 100, Training epochs
    options.update({'batchSize': 2}) # Number of images trained per batch
    options.update({'augmentData': True}) # Boolean flag to augment imaging data
    options.update({'augment_angleRange':  180}) # Range of rotation angles 
    options.update({'augment_shiftRange':  0.25}) # Fraction of total img width/height for (X,Y) shifts
    options.update({'augment_shearRange':  0.25}) # Shearing range
    options.update({'augment_zoomRange':   [0.25,2]}) # Zoom range: [lower, upper]
    options.update({'augment_brightRange': [0.25,2]}) # Brightness range: [lower, upper]. Range for picking the new brightness (relative to the original image), with 1.0 being the original's brightness. Lower must not be smaller than 0.
    options.update({'augment_flipHoriz':   True}) # Boolean flag for horizontal flip
    options.update({'augment_flipVert':    True}) # Boolean flag for vertical flip
    options.update({'augment_fillMode':    'reflect'}) # Points outside image boundary are filled with: {"constant", "nearest", "reflect" or "wrap"}
    options.update({'augment_fillVal':     0}) # If augment_fillMode=='constant', points outside image boundary are filled with fillVal
    model = UNet()
    modelFilePath = r'D:\Code\ROI Segmentation\Code\Dave\Unet_model_FerretAndMouseROIs_Ver2.h5'
    if trainModel:
        (accuracy,loss) = model.trainModel(train_images, train_labels, test_images, test_labels,options)
        model.saveModel(modelFilePath)
        
        # Show training/test accuracy and loss for model
        for (metric,name) in zip([accuracy,loss],['Accuracy','Loss']):
            fig,ax=plt.subplots()
            ax.plot(metric.T)
            if(name.startswith('Accuracy')):
                ax.set_ylim([0.5,1])
            elif(name.startswith('Loss')):
                ax.set_ylim([0,1.25*np.percentile(metric.flatten(),95)])
            ax.set_ylabel(name)
            ax.set_xlabel('Training epoch')
            ax.legend(['Training data','Test data'])
            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
            ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    else:
        model.loadModel(modelFilePath)
    
    
    # Show prediction images
    seed=1
    ferretImg = (validationSet.imgAverage).reshape((1,512,512,1))
    ferretROI = (validationSet.getLabeledROIMask()>0).reshape((1,512,512,1))
    prediction_images = test_images
    prediction_images = np.concatenate((prediction_images,ferretImg),axis=0)
    prediction_labels = test_labels
    prediction_labels = np.concatenate((prediction_labels,ferretROI),axis=0)
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
        dataGenerator = ImageAugmentation(options) 
        for (imgs,masks) in dataGenerator.get(prediction_images,prediction_labels,n_images,seed):
            predictionImages = imgs
            predictionLabels = masks
            break
    else:
        predictionImages=prediction_images
        predictionLabels=prediction_labels
    #predictions=model.model.predict(predictionImages)
    
    for index in range(n_images):
        rawImg = predictionImages[index,:,:,0]
        predictionImg  = model.model.predict(rawImg.reshape([1,rawImg.shape[0],rawImg.shape[1],1])).reshape(rawImg.shape)
        filteredImg    = ndimage.median_filter(predictionImg,7)
        thresholdedImg = predictionImg>0.3
        ROI = predictionLabels[index,:,:,0];
        meanImg = np.nanmean(rawImg)
        stdImg  = np.nanstd(rawImg)
        clippingValue = 4
        clipVals = [meanImg-clippingValue*stdImg,meanImg+clippingValue*stdImg]
        
        fig,ax=plt.subplots(ncols=4,nrows=2);
        ax=ax.flatten()
        ax[0].imshow(rawImg,cmap='gray',vmin=clipVals[0],vmax=clipVals[1])
        ax[0].set_title('Raw image')
        ax[1].imshow(ROI,cmap='Blues')
        ax[1].set_title('Mask')
        ax[2].imshow(rawImg,cmap='gray',vmin=clipVals[0],vmax=clipVals[1])
        ax[2].imshow(ROI,cmap='Blues',alpha=0.25)
        ax[2].set_title('Mask overlay')
        
        ax[4].imshow(predictionImg,cmap='Reds',vmin=0,vmax=1)
        ax[4].set_title('Mask prediction')
        ax[5].imshow(predictionImg,cmap='Reds',vmin=0,vmax=1)
        ax[5].imshow(ROI,cmap='Blues',alpha=0.5)
        ax[5].set_title('Mask/prediction overlay')
        ax[6].imshow(rawImg,cmap='gray',vmin=clipVals[0],vmax=clipVals[1])
        ax[6].imshow(predictionImg,cmap='Reds',vmin=0,vmax=1,alpha=0.25)
        ax[6].set_title('Prediction overlay')
        ax[7].imshow(thresholdedImg,cmap='Reds',vmin=0,vmax=1)
        ax[7].imshow(ROI,cmap='Blues',alpha=0.5)
        ax[7].set_title('Mask/thresholded overlay')
