# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 22:33:02 2019

@author: dwhitney
"""

import h5py
import json
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from PIL import Image
import sys
from scipy.spatial import Delaunay
from scipy import ndimage
import time

DEFAULT_SAVE_PATH  = os.path.join(os.getcwd(),'temp_imaging_data.h5')

def excludeOverlappedRegions(inputArgs):
    ROI = inputArgs[0]
    sharedRegions = inputArgs[1]
    return np.any(np.prod(ROI==sharedRegions,axis=2),axis=0)            
                
def progressBar(value, endvalue, elapsedTime, bar_length=20):
    "Show a progress bar that updates as more imaging data is read"
    # Based on: https://stackoverflow.com/questions/6169217/replace-console-output-in-python
    try:
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        estimatedTime = (endvalue-value)*(elapsedTime/value)
        sys.stdout.write("\rLoading data: [{0}] {1}% done (Elapsed time: {2:4.1f} seconds. Estimated time left: {3:4.1f} seconds".format(arrow + spaces, int(round(percent * 100)), elapsedTime, estimatedTime))
        sys.stdout.flush()
    except:
        pass
    
def ROIMask(coordinates,imsize):
    """Creates a boolean mask with image dimensions specified by imsize using (X,Y)
    locations specified by coordinates"""
    mask  = np.zeros((imsize),dtype='bool')
    #index = coordinates[:,0]+coordinates[:,1]*imsize[0]
    mask[coordinates[:,0],coordinates[:,1]]=True
    return mask

class imagingDataset():
    def __init__(self,imageDataPath='',ROIPath=''):
        self.imageDataPath = imageDataPath;
        self.ROIPath = ROIPath;
        return;
    
    def getImagingData(self,imageDataPath='', returnOnlyAverage=False, imageFileTypes=('tiff','tif'),isVerbose=True):
        "Load all imaging data from imageDataPath. Can optionally return only average z-projection image to save on memory-overhead"
        if(len(imageDataPath)==0):
            imageDataPath = self.imageDataPath;
        else:
            self.imageDataPath = imageDataPath;
        imageFiles = [os.path.join(imageDataPath,f) for f in os.listdir(imageDataPath) if f.endswith(imageFileTypes)];
        
        # Get image dimensions and number of frames
        self.imgAverage = [];
        self.imgStack   = [];        
        if len(imageFiles)==0:
            print('No image files found in {}'.format(imageDataPath))
            return (self.imgAverage,self.imgStack);
        else:
            with Image.open(imageFiles[0]) as img:            
                (h,w,nFramesPerFile) = (img.height,img.width,img.n_frames)
            if(nFramesPerFile>1): #isMultipageTIFF
                nFrames = sum([Image.open(f).n_frames for f in imageFiles])
            else:
                nFrames = len(imageFiles)
            self.imgAverage = np.zeros((h,w))
        
        # Loop through and generate an average projection image 
        # (and optionally return entire imaging stack)
        t0 = time.time();
        imgsProcessed  = 0.;  
        updateInterval = 0.05; # Updates progress bar every X%
        nextUpdate     = updateInterval; 
        for file in imageFiles:
            with Image.open(file) as file:
                for i in range(file.n_frames):
                    # Get imaging frame
                    file.seek(i)
                    frame = np.array(file)
                    self.imgAverage = self.imgAverage+(frame/nFrames)
                    if not returnOnlyAverage:                    
                        self.imgStack.append(frame)
                    
                    # Update progress bar
                    imgsProcessed += 1.    
                    if(isVerbose&(imgsProcessed/nFrames>=nextUpdate)):
                        progressBar(100*imgsProcessed/nFrames,100.,time.time()-t0)
                        nextUpdate += (updateInterval-np.finfo(float).eps) # Eps added to ensure that the update frequency is always correct
                        time.sleep(0.1)
        self.imgStack = np.array(self.imgStack) # Convert imaging stack to a numpy array
        return (self.imgAverage,self.imgStack)
    
    def getJSONROIs(self,ROIPath=''):
        "load cell ROIs based on the input ROIPath"
        if(len(ROIPath)==0):
            ROIPath = self.ROIPath
        else:
            self.ROIPath = ROIPath
        try:
            with open(self.ROIPath) as f:
                ROIData = json.load(f);
                self.ROIs = [np.array(ROI['coordinates']) for ROI in ROIData]
            return self.ROIs
        except:
            print('\nNo ROIs found in {}'.format(ROIPath))
            self.ROIs = []
        return self.ROIs;
    
    def erodeROIs(self,useXOR=True,useErosion=True,kernel=np.ones((3,3)),useMultiProcessing=True):
        """Performs XOR on ROIs and erodes filled ROIs by a specified kernel"""
                
        # Find and eliminate shared ROI regions
        ROIs = self.ROIs
        if useXOR:
            # Finds pixels that are shared between ROIs
            (uniqueInstances,nInstances) = np.unique(np.concatenate(ROIs,axis=0),axis=0,return_counts=True)
            sharedRegions = uniqueInstances[nInstances>1]
            sharedRegions = np.swapaxes(sharedRegions[:,:,None],1,2) # Swap axes here, so that we can efficiently compared the shared regions with each ROI array
            uniqueROIs = list()
            
            # Identify shared pixels for each ROI
            if useMultiProcessing:
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    overlappedRegions = pool.map(excludeOverlappedRegions, [(ROI,sharedRegions) for ROI in ROIs])
            else: #This step is very slow without parallelization of some kind
                 overlappedRegions = [np.any(np.prod(ROI==sharedRegions,axis=2),axis=0) for ROI in ROIs]
            for (ROI,sharedRegion) in zip(ROIs,overlappedRegions):
                uniqueROI    = ROI[~sharedRegion]
                if(sum(uniqueROI.flatten())>1): # Only include ROIs that have a value
                    uniqueROIs.append(uniqueROI)  
        else:
            uniqueROIs = ROIs
        
        # Erode ROIs
        if useErosion:
            erodedROIs = list()
            for ROI in uniqueROIs:
                (up,left)    = ROI.min(axis=0)
                (down,right) = ROI.max(axis=0)
                ROIImage = np.zeros((down-up+1,right-left+1),'bool')
                ROIImage[ROI[:,0]-up,ROI[:,1]-left]=True
                ROIImage = ndimage.binary_erosion(ROIImage,structure=kernel)
                boundaryPts = np.array([(i+up,j+left) for i in range(ROIImage.shape[0]) for j in range(ROIImage.shape[1]) if ROIImage[i,j]==1])
                erodedROIs.append(boundaryPts)  
        else:
            erodedROIs = uniqueROIs
            
        self.ROIs = erodedROIs
        return erodedROIs;
            
    def fillROIs(self):
        """Ensures that all cell ROIs are filled (useful if using ImageJ ROIs that just label boundary)"""
        
        newROIs = list()
        for ROI in self.ROIs:
            # Define (X,Y) cell boundary
            (up,left)    = ROI.min(axis=0)
            (down,right) = ROI.max(axis=0)
            (y,x) = np.meshgrid(range(up,down),range(left,right))
            (y,x) = (y.flatten(), x.flatten())
            grid = np.vstack((y,x)).T
        
            # Define a new ROI mask including all points within the ROI path
            ROIPath = matplotlib.path.Path(ROI)
            ROIMask = ROIPath.contains_points(grid)
            newROI  = np.array([(y[i],x[i]) for i in range(len(ROIMask)) if ROIMask[i]])
            newROI  = np.unique(np.concatenate((ROI,newROI)),axis=0) # Add previous ROI boundary points along with new points
            newROIs.append(newROI)
        self.ROIs = newROIs
        return newROIs;
    
    def openROIs(self,useImageDilation=True):
        """ Opens each cell ROI, so that edge points are only labeled"""
        
        newROIs = list()
        for ROI in self.ROIs:
            if(useImageDilation):
                (up,left)    = ROI.min(axis=0)
                (down,right) = ROI.max(axis=0)
                ROIImage = np.zeros((down-up+1,right-left+1),'bool')
                ROIImage[ROI[:,0]-up,ROI[:,1]-left]=True
                ROIImage = np.bitwise_xor(ROIImage,ndimage.binary_erosion(ROIImage,ndimage.generate_binary_structure(2,1)))
                boundaryPts = np.array([(i+up,j+left) for i in range(ROIImage.shape[0]) for j in range(ROIImage.shape[1]) if ROIImage[i,j]==1])
            else:
                # Compute boundary nodes
                nodes  = Delaunay(ROI).simplices
                adjacentNodes = np.zeros((len(nodes)),'int') 
                for (i,node) in enumerate(nodes):
                    sharedVertices   = np.array([(6-len(np.unique(np.concatenate((node,adjNode))))) for adjNode in nodes]) #Computes the number of shared nodes
                    adjacentNodes[i] = np.sum(sharedVertices==2); # An adjacent node has a shared edge (i.e. 2 vertices)
                boundaryNodes = nodes[adjacentNodes<3,:]
                
                # Compute boundary vertices
                localEdges = [[0,1],[0,2],[1,2]]
                allEdges      = np.sort(np.array([node[edge] for node in nodes for edge in localEdges]),axis=1)
                boundaryEdges = np.sort(np.array([node[edge] for node in boundaryNodes for edge in localEdges]),axis=1)
                uniqueEdges   = [edge for edge in boundaryEdges if (edge==allEdges).min(axis=1).sum()==1]
                boundaryPts   = ROI[np.unique(uniqueEdges),:]                
            newROIs.append(boundaryPts)   
        self.ROIs = newROIs
        return newROIs;
                
    def saveToHDF5(self,filePath=DEFAULT_SAVE_PATH):
        "saves data to a hd5f file"
        with h5py.File(filePath, 'w') as file:
            file.create_dataset(name='imgStack'  , data=getattr(self,'imgStack'  ))
            file.create_dataset(name='imgAverage', data=getattr(self,'imgAverage'))
            ROIGroup = file.create_group('ROIs')
            for (index,ROI) in enumerate(getattr(self,'ROIs')):
                ROIGroup.create_dataset(name='ROI{:05d}'.format(index), data=ROI)
            file.close()
        
    def loadFromHDF5(self,filePath=DEFAULT_SAVE_PATH):
        "loads data from a hd5f file"
        with h5py.File(filePath, 'r') as file:
            ROIs = []
            for ROI in file['ROIs'].keys():
                ROIs.append(file['/ROIs/{}'.format(ROI)].value)
            setattr(self,'imgStack'  ,file['imgStack'].value)
            setattr(self,'imgAverage',file['imgAverage'].value)
            setattr(self,'ROIs'      ,ROIs)
            file.close()
            
    def getLabeledROIMask(self):
        "Returns labeled ROI mask"
        mask = np.max(np.array([(1+index)*ROIMask(ROI,(512,512)).astype('float64') for (index,ROI) in enumerate(self.ROIs)]),axis=0); # Setup a maximum intensity projection of the cellular masks
        return mask
    
    def getReferenceImg(self,zScore=True):
        img=self.imgAverage
        if zScore:
            img=(img-np.nanmean(img.flatten()))/np.nanstd(img.flatten())
        return img
            
    def showROIs(self,clippingValue=3):
        "Show show cellular ROIs with average z-projection"
        fig,axes=plt.subplots(nrows=1,ncols=3)
        try:
            meanImg = np.nanmean(self.imgAverage)
            stdImg  = np.nanstd(self.imgAverage)
            for index in [0,2]:
                axes[index].imshow(self.imgAverage,cmap='gray',vmin=meanImg-clippingValue*stdImg,vmax=meanImg+clippingValue*stdImg)
                axes[index].set_title('Average z-projection image')
        except:
            print('Error: imgAverage not found\n')
        
        try:
            mask = self.getLabeledROIMask() # Setup a maximum intensity projection of the cellular masks
            for (index,alphaVal) in zip([1,2],[1,0.25]):
                axes[index].imshow(mask,cmap='nipy_spectral',alpha=alphaVal)
                axes[index].set_title('Cell masks')
        except:
            print('Error: ROIs not found\n')
        axes[-1].set_title('Overlay')
            
if __name__ == "__main__":
    baseFolder    = r'D:/Code/ROI Segmentation/DataSets/Neurofinder/neurofinder.00.00/neurofinder.00.00/'
    imageDataPath = baseFolder+'images/'
    ROIPath       = baseFolder+'regions/regions.json'
    
    imageSet=imagingDataset()
    imageSet.getImagingData(imageDataPath)
    imageSet.getJSONROIs(ROIPath)
    imageSet.showROIs()