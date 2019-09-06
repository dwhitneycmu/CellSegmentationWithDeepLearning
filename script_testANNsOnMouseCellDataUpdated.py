# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:37:28 2019

@author: dwhitney
"""

# import os
# os.chdir(r'D:\Code\ROI Segmentation\Code\Dave')
# exec(open('D:\Code\ROI Segmentation\Code\Dave\script_testANNsOnMouseCellDataUpdated.py').read())

if __name__ == "__main__":
    from imageAugmentation import ImageAugmentationSequence
    import imagingDataset
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from scipy import ndimage
    from UNet import UNet
    from readImageJROIs import readImageJROIsFromZip
    from PIL import Image
    matplotlib.use('Qt5Agg')
    matplotlib.matplotlib_fname()
    
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
    
    reloadData = True
    plt.close('all')
    np.random.seed(1)
    baseFolder = r'D:\Code\ROI Segmentation'
    if(reloadData):
        useMouseData  = True
        useFerretData = True
        imageSets  = []
        trainExpts = []
        testExpts  = []

        # Load mouse imaging data for training neural network
        if useMouseData:
            # Find and load mouse experiments
            dataFolder  = baseFolder+r'\DataSets\MouseData_Neurofinder\Revised Data'
            imgDataSets = [os.path.join(dataFolder,f) for f in os.listdir(dataFolder) if f.endswith('.tif')]
            ROIDataSets = [os.path.join(dataFolder,f) for f in os.listdir(dataFolder) if f.endswith('.zip')]
            nMouseFoVs  = len(imgDataSets)
            for index,(imgPath,ROIPath) in enumerate(zip(imgDataSets,ROIDataSets)):
                print('Loading mouse dataset {} (of {}): {}'.format(index+1,nMouseFoVs,imgPath))
                imageSet = getImagingAndROIData(imgPath,ROIPath)
                imageSets.append(imageSet)

            # Use all mouse data except hippocampal datasets that are dense image
            trainExpts = trainExpts + [5, 6, 7, 9]
            testExpts  = testExpts  + [0, 1, 2, 4]
        else:
            nMouseFoVs = 0

        # Load ferret data for training neural network
        if useFerretData:
            # Find and load ferret experiments
            dataFolder  = baseFolder+r'\DataSets\FerretData_DaveAndJeremy'
            imgDataSets = [os.path.join(path,filename) for (path, names, filenames) in os.walk(dataFolder) for filename in filenames if filename.endswith('.tif')]
            ROIDataSets = [os.path.join(path,filename) for (path, names, filenames) in os.walk(dataFolder) for filename in filenames if filename.endswith('.zip')]
            nFerretFoVs = len(imgDataSets)
            for index,(imgPath,ROIPath) in enumerate(zip(imgDataSets,ROIDataSets)):
                print('Loading ferret dataset {} (of {}): {}'.format(index+1,nFerretFoVs,imgPath))
                imageSet = getImagingAndROIData(imgPath,ROIPath)
                imageSets.append(imageSet)

            # Split ferret experiments into training and testing data
            percentOfDataToTrain = 0.8
            randomFerretFoVs = np.random.permutation(nFerretFoVs) + useMouseData*nMouseFoVs
            dividingExpt = int(np.round(percentOfDataToTrain * len(randomFerretFoVs)))
            trainExpts = trainExpts + randomFerretFoVs[:dividingExpt].tolist()
            testExpts  = testExpts  + randomFerretFoVs[dividingExpt:].tolist()
        else:
            nFerretFoVs = 0

        # Get extra validation experiment (ferret image and ROIs)
        imgDataPath = baseFolder+r'\DataSets\AVG_rawData_200xDownsampled.tif'
        ROIPath     = baseFolder+r'\DataSets\RoiSet.zip'
        validationSet = getImagingAndROIData(imgDataPath,ROIPath)
    nImageSets = len(imageSets)
    nCellROIs  = sum([len(imageSet.ROIs) for imageSet in imageSets]) 
    print('Loaded {} FoVs and {} ROIs'.format(nImageSets,nCellROIs))

    # Setup training dataset
    train_images = np.stack([imageSets[i].getReferenceImg() for i in trainExpts])
    train_labels = np.stack([imageSets[i].getLabeledROIMask() for i in trainExpts])>0
    test_images  = np.stack([imageSets[i].getReferenceImg() for i in testExpts])
    test_labels  = np.stack([imageSets[i].getLabeledROIMask() for i in testExpts])>0
    train_images = train_images.reshape(train_images.shape+(1,))
    train_labels = train_labels.reshape(train_labels.shape+(1,))
    test_images  = test_images.reshape(test_images.shape+(1,))
    test_labels  = test_labels.reshape(test_labels.shape+(1,))

    # Setup and train model
    reuseModel = True  # Reuse an existing model, else initializes a new model with default weights
    trainModel = False  # Training model
    options=dict()
    options.update({'showTrainingData': False})
    options.update({'blocks': 30})    # 5, Training blocks (Each is a full number of training epochs)
    options.update({'epochs': 5})   # 100, Training epochs
    options.update({'batchSize': 1}) # Number of images trained per batch
    options.update({'augmentData': True}) # Boolean flag to augment imaging data
    options.update({'augment_angleRange':  180}) # 180, Range of rotation angles
    options.update({'augment_shiftRange':  0.1}) # 0.1, Fraction of total img width/height for (X,Y) shifts
    options.update({'augment_shearRange':  0.1}) # 0.25, Shearing range
    options.update({'augment_zoomRange':   [0.25,2]}) # [0.25,2], Zoom range: [lower, upper]
    options.update({'augment_brightRange': [0.75,1.25]}) # [0.5,2], Brightness range: [lower, upper]. Range for picking the new brightness (relative to the original image), with 1.0 being the original's brightness. Lower must not be smaller than 0.
    options.update({'augment_flipHoriz':   True}) # True, Boolean flag for horizontal flip
    options.update({'augment_flipVert':    True}) # True, Boolean flag for vertical flip
    options.update({'augment_fillMode':    'reflect'}) # Reflect, Points outside image boundary are filled with: {"constant", "nearest", "reflect" or "wrap"}
    options.update({'augment_fillVal':     0}) # 0, If augment_fillMode=='constant', points outside image boundary are filled with fillVal
    model = UNet()
    modelType = 'Unet' #Unet
    modelName = 'FerretAndMouseROIs_Relu_BatchNorm_Dropout{}_{}'.format(options['blocks'],options['epochs']) #_Relu_5x100_BatchNorm
    filePath_model   = r'{}\Code\Dave\{}_model_{}.h5'.format(   baseFolder,modelType,modelName)
    filePath_history = r'{}\Code\Dave\{}_history_{}.hdf5'.format(baseFolder,modelType,modelName)
    filePath_params  = r'{}\Code\Dave\{}_params_{}.txt'.format(  baseFolder,modelType,modelName)
    if reuseModel:
        model.loadModel(filePath_model)
        model.loadModelHistory(filePath_history)
        model.loadModelParameters(filePath_params)
    if trainModel:
        model.trainModel(train_images, train_labels, test_images, test_labels,options)
        model.saveModel(filePath_model)
        model.saveModelHistory(filePath_history)
        model.saveModelParameters(filePath_params)

    # Show training/test accuracy and loss for model
    try:
        modelHistory = model.modelHistory
        metrics = modelHistory.keys()
        for metric_name in metrics:
            metric = modelHistory[metric_name]
            fig, ax = plt.subplots()
            ax.plot(metric.T)
            if (metric_name.startswith('accuracy')):
                ax.set_ylim([0.5, 1])
            elif (metric_name.startswith('loss')):
                ax.set_ylim([0, 1.25 * np.percentile(metric.flatten(), 95)])
            else:
                ax.set_ylim([metric.min(), metric.max()])
            ax.set_ylabel(metric_name)
            ax.set_xlabel('Training epoch')
            ax.legend(['Training data', 'Test data'])
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    except:
        print('Warning: No model training history found. Skipping visualization of performance metrics...\n')

    # Quantify network performance
    if 0:
        validationImg = validationSet.getReferenceImg()
        validationImg = validationImg.reshape((1,)+validationImg.shape+(1,))
        validationMask = validationSet.getLabeledROIMask()>0
        validationMask = validationMask.reshape(validationImg.shape)
        validationImgs  = np.concatenate((validationImg,test_images[0:4,:,:,:]),axis=0)
        validationMasks = np.concatenate((validationMask,test_labels[0:4, :, :, :]), axis=0)
        options.update({'augment_angleRange': 0})  # Range of rotation angles
        options.update({'augment_shiftRange': 0})  # Fraction of total img width/height for (X,Y) shifts
        options.update({'augment_shearRange': 0})  # Shearing range
        options.update({'augment_zoomRange': [0.99, 1.01]})  # Zoom range: [lower, upper]
        options.update({'augment_brightRange': [0.95,1]})  # Brightness range: [lower, upper]. Range for picking the new brightness (relative to the original image), with 1.0 being the original's brightness. Lower must not be smaller than 0.
        options.update({'augment_flipHoriz': False})  # Boolean flag for horizontal flip
        options.update({'augment_flipVert': False})  # Boolean flag for vertical flip
        options.update({'augment_fillMode': 'reflect'})  # Points outside image boundary are filled with: {"constant", "nearest", "reflect" or "wrap"}
        options.update({'augment_fillVal': 0})  # If augment_fillMode=='constant', points outside image boundary are filled with fillVal
        dataGenerator = ImageAugmentationSequence(options)
        dataTested = [(validationImg, validationMask),(validationImgs, validationMasks)]#,
                      #(train_images, train_labels),(test_images, test_labels)]
        for threshold in [0.25,0.5,0.75]:
            performanceMetricsSets = []
            for index,(imgs,labels) in enumerate(dataTested):
                nImgs = imgs.shape[0]
                dataGenerator.generate(imgs,labels,nImgs,seed=1)
                augmentedImgs   = dataGenerator.x
                augmentedLabels = dataGenerator.y
                performanceMetrics = dict()
                performanceMetrics['accuracy'] = []
                performanceMetrics['precision'] = []
                performanceMetrics['recall'] = []
                for index in range(nImgs):
                    img   = augmentedImgs[index,:,:,:]
                    actualLabel    = augmentedLabels[index,:,:,:]==1
                    predictedLabel = model.model.predict(img.reshape((1,)+img.shape)).reshape(actualLabel.shape)>=threshold

                    accuracy = np.mean(actualLabel==predictedLabel)
                    true_positives  = np.nansum((actualLabel==1)&(predictedLabel==1))
                    false_positives = np.nansum((actualLabel==0)&(predictedLabel==1))
                    true_negative   = np.nansum((actualLabel == 0) & (predictedLabel == 0))
                    false_negative  = np.nansum((actualLabel == 1) & (predictedLabel == 0))
                    precision       = true_positives / (true_positives+false_positives)
                    recall          = true_positives / (true_positives+false_negative)
                    performanceMetrics['accuracy'].append(accuracy)
                    performanceMetrics['precision'].append(precision)
                    performanceMetrics['recall'].append(recall)
                performanceMetricsSets.append(performanceMetrics)
            print('Threshold = {}'.format(threshold))
            for metric in ['accuracy','precision','recall']:
                print('Mean {}: {:4.2f}% (Validation Img), {:4.2f}% (Validation Imgs), {:4.2f}% (Training), {:4.2f}% (Test)'.format(metric,
                      100*np.nanmean(performanceMetricsSets[0][metric]),100*np.nanmean(performanceMetricsSets[1][metric]),
                      100*np.nanmean(performanceMetricsSets[2][metric]),100*np.nanmean(performanceMetricsSets[3][metric])))

    # Show prediction images
    seed = 1  # RNG Seed
    ferretImg = (validationSet.imgAverage).reshape((1,512,512,1))
    ferretROI = (validationSet.getLabeledROIMask()>0).reshape((1,512,512,1))
    prediction_images = test_images[:10] #train_images, test_images
    prediction_images = np.concatenate((prediction_images,ferretImg),axis=0)
    prediction_labels = test_labels[:10] #train_labels, test_labels
    prediction_labels = np.concatenate((prediction_labels,ferretROI),axis=0)
    n_images=prediction_images.shape[0]
    if(options['augmentData']):
        dataGenerator.generate(prediction_images,prediction_labels,n_images,seed)
        predictionImages = dataGenerator.x
        predictionLabels = dataGenerator.y
    else:
        predictionImages=prediction_images
        predictionLabels=prediction_labels
    predictions=model.model.predict(predictionImages)
    
    for index in range(n_images): #n_images
        rawImg = predictionImages[index,:,:,0]
        predictionImg  = model.model.predict(rawImg.reshape([1,rawImg.shape[0],rawImg.shape[1],1])).reshape(rawImg.shape)
        filteredImg    = ndimage.median_filter(predictionImg,7)
        thresholdedImg = predictionImg>0.5
        ROI = predictionLabels[index,:,:,0]
        meanImg = np.nanmean(rawImg)
        stdImg  = np.nanstd(rawImg)
        clippingValue = 4
        clipVals = [meanImg-clippingValue*stdImg,meanImg+clippingValue*stdImg]
        
        fig,ax=plt.subplots(ncols=4,nrows=2)
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

    plt.show()