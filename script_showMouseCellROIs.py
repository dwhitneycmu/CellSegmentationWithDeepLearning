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
        # Load an unlabeled, z-projection image (*.TIF file)
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

    # Load mouse imaging data for training neural network
    #plt.close('all')
    dataFolder = r'D:\Code\ROI Segmentation\DataSets\Neurofinder\Revised Data'
    imageFileTypes = '.tif';
    imgDataSets    = [os.path.join(dataFolder,f) for f in os.listdir(dataFolder) if f.endswith('.tif')];
    ROIDataSets    = [os.path.join(dataFolder,f) for f in os.listdir(dataFolder) if f.endswith('.zip')];
    imageSets = []
    for (imgPath,ROIPath) in zip(imgDataSets,ROIDataSets):
        imageSet = getImagingAndROIData(imgPath,ROIPath)
        imageSet.showROIs()
        imageSets.append(imageSet)
        
    trainExpts  = [0,1,2,4,5,6,7,9]
    train_images = np.stack([imageSets[i].getReferenceImg() for i in trainExpts])
    train_labels = np.stack([imageSets[i].getLabeledROIMask() for i in trainExpts])>0
    train_images = train_images.reshape(train_images.shape+(1,))
    train_labels = train_labels.reshape(train_labels.shape+(1,))
        
    # Get validation experience (ferret image and ROIs)
    filePath=r'D:\Data Science\2018 MPFI Imaging Course\RAW Data\rawData_200xDownsampled.tif'
    ROIPath = r'D:\Data Science\2018 MPFI Imaging Course\analyzedData\RoiSet.zip'
    validationSet = getImagingAndROIData(filePath,ROIPath)
    
    # Show image augmentation techniques
    dataGenerator = ImageAugmentation() 
    for (imgs,masks) in dataGenerator.get(train_images,train_labels,train_images.shape[0],seed=1):
        for index in range(imgs.shape[0]):
            fig,axes=plt.subplots(nrows=1,ncols=3)
            clippingValue = 2
            img  = imgs[index,:,:,0]
            mask = masks[index,:,:,0]
            meanImg = np.nanmean(img)
            stdImg  = np.nanstd(img)
            for index in [0,2]:
                axes[index].imshow(img,cmap='gray',vmin=meanImg-clippingValue*stdImg,vmax=meanImg+clippingValue*stdImg)
                axes[index].set_title('Average z-projection image')
            for (index,alphaVal) in zip([1,2],[1,0.25]):
                axes[index].imshow(mask,cmap='nipy_spectral',alpha=alphaVal)
                axes[index].set_title('Cell masks')
            axes[-1].set_title('Overlay')
        break