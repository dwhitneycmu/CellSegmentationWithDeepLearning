# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:56:29 2019

@author: dwhitney
"""

'''  ImageJ/NIH Image 64 byte ROI outline header
    2 byte numbers are big-endian signed shorts
    (Source: https://imagej.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html)
        
    0-3     "Iout"
    4-5     version (>=217)
    6-7     roi type (encoded as one byte)
    8-9     top
    10-11   left
    12-13   bottom
    14-15   right
    16-17   NCoordinates
    18-33   x1,y1,x2,y2 (straight line) | x,y,width,height (double rect) | size (npoints)
    34-35   stroke width (v1.43i or later)
    36-39   ShapeRoi size (type must be 1 if this value>0)
    40-43   stroke color (v1.43i or later)
    44-47   fill color (v1.43i or later)
    48-49   subtype (v1.43k or later)
    50-51   options (v1.43k or later)
    52-52   arrow style or aspect ratio (v1.43p or later)
    53-53   arrow head size (v1.43p or later)
    54-55   rounded rect arc size (v1.43p or later)
    56-59   position
    60-63   header2 offset
    64-       x-coordinates (short), followed by y-coordinates 
    
    # ImageJ Byte Offsets
    VERSION = 4;
    TYPE = 6;
    TOP = 8;
    LEFT = 10;
    BOTTOM = 12;
    RIGHT = 14;
    N_COORDINATES = 16;
    X1 = 18;
    Y1 = 22;
    X2 = 26;
    Y2 = 30;
    XD = 18;
    YD = 22;
    WIDTHD = 26;
    HEIGHTD = 30;
    SIZE = 18;
    STROKE_WIDTH = 34;
    SHAPE_ROI_SIZE = 36;
    STROKE_COLOR = 40;
    FILL_COLOR = 44;
    SUBTYPE = 48;
    OPTIONS = 50;
    ARROW_STYLE = 52;
    FLOAT_PARAM = 52; //ellipse ratio or rotated rect width
    POINT_TYPE= 52;
    ARROW_HEAD_SIZE = 53;
    ROUNDED_RECT_ARC_SIZE = 54;
    POSITION = 56;
    HEADER2_OFFSET = 60;
    COORDINATES = 64;
        // header2 offsets
    C_POSITION = 4;
    Z_POSITION = 8;
    T_POSITION = 12;
    NAME_OFFSET = 16;
    NAME_LENGTH = 20;
    OVERLAY_LABEL_COLOR = 24;
    OVERLAY_FONT_SIZE = 28; //short
    AVAILABLE_BYTE1 = 30;  //byte
    IMAGE_OPACITY = 31;  //byte
    IMAGE_SIZE = 32;  //int
    FLOAT_STROKE_WIDTH = 36;  //float
    ROI_PROPS_OFFSET = 40;
    ROI_PROPS_LENGTH = 44;
    COUNTERS_OFFSET = 48;   
    
    // subtypes
    TEXT = 1;
    ARROW = 2;
    ELLIPSE = 3;
    IMAGE = 4;
    ROTATED_RECT = 5;
        
    // options
    SPLINE_FIT = 1;
    DOUBLE_HEADED = 2;
    OUTLINE = 4;
    OVERLAY_LABELS = 8;
    OVERLAY_NAMES = 16;
    OVERLAY_BACKGROUNDS = 32;
    OVERLAY_BOLD = 64;
    SUB_PIXEL_RESOLUTION = 128;
    DRAW_OFFSET = 256;
    ZERO_TRANSPARENT = 512;
    SHOW_LABELS = 1024;
    SCALE_LABELS = 2048;
    PROMPT_BEFORE_DELETING = 4096; //points
        
    // types
    polygon=0, rect=1, oval=2, line=3, freeline=4, polyline=5, noRoi=6,
            freehand=7, traced=8, angle=9, point=10;
    '''

import numpy as np
from zipfile import ZipFile

IMAGEJ_HEADER_DEFAULTS = {'Iout':1232041332,'VERSION':223,'TYPE':7,'TYPE_EXTRABYTE':0,'TOP':0,'LEFT':0,'BOTTOM':0,'RIGHT':0,'N_COORDINATES':0, \
                          'XD':0,'YD':0,'WIDTHD':0,'HEIGHTD':0,'STROKE_WIDTH':0,'SHAPE_ROI_SIZE':0,'STROKE_COLOR':0, \
                          'FILL_COLOR':0,'SUBTYPE':0,'OPTIONS':0,'ARROW_STYLE':0,'ARROW_HEAD_SIZE':0, \
                          'ROUNDED_RECT_ARC_SIZE':0,'POSITION':0,'HEADER2_OFFSET':0}
IMAGEJ_HEADER_BYTESIZE = {'Iout':4,'VERSION':2,'TYPE':1,'TYPE_EXTRABYTE':1,'TOP':2,'LEFT':2,'BOTTOM':2,'RIGHT':2,'N_COORDINATES':2, \
                          'XD':4,'YD':4,'WIDTHD':4,'HEIGHTD':4,'STROKE_WIDTH':2,'SHAPE_ROI_SIZE':4,'STROKE_COLOR':4, \
                          'FILL_COLOR':4,'SUBTYPE':2,'OPTIONS':2,'ARROW_STYLE':1,'ARROW_HEAD_SIZE':1, \
                          'ROUNDED_RECT_ARC_SIZE':2,'POSITION':4,'HEADER2_OFFSET':4}
COORDINATE_BYTES = 2 # Is a short (16-bit unsigned integer)

def readNextBytes(fileData,byteOffset=0,byteSize=2):
    "Reads byte data from a binary list"
    bytesRead  = fileData[byteOffset:(byteOffset+byteSize)]
    value      = int.from_bytes(bytesRead, byteorder='big')
    byteOffset += byteSize
    return (value,byteOffset)

def writeNextBytes(data,byteSize=2,isNumpy=False):
    "Convert data to bytes and return"
    if(isNumpy):
        byteData = data.byteswap().tobytes() #Byte swap ensures data is formatted as big-endian
    else:
        byteData = data.to_bytes(byteSize,byteorder='big')
        print('{} (bytesize={}): {}'.format(data,byteSize,byteData))
    return(byteData)

def readImageJROI(filePath='',isDataStream=False):
    "Extracts ROI information from an ImageJ .ROI file"
    
    # Read file data from ImageJ file
    if(isDataStream): # Here we override the function to allow a stream of byte data as input (helpful for feeding *.zip data into function for decoding)
        fileData = filePath
    else:
        with open(filePath,'rb') as file:
            fileData = file.readlines()
        fileData = [byte for line in fileData for byte in line]
    
    # Read ImageJ Header
    byteOffset = 0
    ImageJDict = dict()
    for key in IMAGEJ_HEADER_BYTESIZE.keys():
        (value,byteOffset) = readNextBytes(fileData,byteOffset,IMAGEJ_HEADER_BYTESIZE[key])
        ImageJDict.update({key:value})
        
    # Read ImageJ ROI Coordinates
    XYOffsets = [ImageJDict['LEFT'],ImageJDict['TOP']]
    ROICoordinates = np.zeros((ImageJDict['N_COORDINATES'],2),dtype='int32')
    for XYIndex in range(2): # Loops through x-coordinates and then y-coordinates
        for CoordinateIndex in range(ImageJDict['N_COORDINATES']):
            (value,byteOffset) = readNextBytes(fileData,byteOffset,COORDINATE_BYTES)
            ROICoordinates[CoordinateIndex,1-XYIndex]=value+XYOffsets[XYIndex]-1 # Flip the XYIndex for variable ROICoordinates, so that y-coordinates are first. We subtract "1" because ImageJ indexes start at 1
        
    return(ROICoordinates,ImageJDict)  

def writeImageJROI(filePath='',ROICoordinates=None,streamData=False):
    "Writes ROI information to an ImageJ .ROI file"
    
    # Setup and format ImageJ Header
    minVals = np.min(ROICoordinates,axis=0)
    maxVals = np.max(ROICoordinates,axis=0)
    nROIs   = ROICoordinates.shape[0]
    ImageJDict = IMAGEJ_HEADER_DEFAULTS
    ImageJDict['N_COORDINATES'] = nROIs
    ImageJDict['TOP']    =  int(minVals[0])
    ImageJDict['LEFT']   =  int(minVals[1])
    ImageJDict['BOTTOM'] =  int(maxVals[0])
    ImageJDict['RIGHT']  =  int(maxVals[1])

    # Convert header information and ImageJ ROI Coordinates into byte data and concatenate into a list
    fileData = list()
    XYOffsets = [ImageJDict['LEFT'],ImageJDict['TOP']]
    for key in IMAGEJ_HEADER_BYTESIZE.keys():
        print("{} : {}".format(key,ImageJDict[key]))
        byteData = writeNextBytes(ImageJDict[key],IMAGEJ_HEADER_BYTESIZE[key])
        fileData.append(byteData)
    for XYIndex in range(2): # Loops through x-coordinates and then y-coordinates
        for CoordinateIndex in range(ImageJDict['N_COORDINATES']):
            coordinate = np.int16(1+ROICoordinates[CoordinateIndex,1-XYIndex]-XYOffsets[XYIndex]); # Note: We had flipped the XYIndex for ROICoordinates, so that y-coordinates were first. We also add "1" because ImageJ indexes start at 1
            byteData = writeNextBytes(coordinate,isNumpy=True)
            fileData.append(byteData)

    # Write ROI coordinates to ImageJ .ROI file
    if(streamData):
        return(fileData)
    else:
        with open(filePath,'wb') as file:
            for byte in fileData:
                file.write(byte)
                
def readImageJROIsFromZip(filePath):
    "Read ImageJ ROIs from a *.zip file"
    
    ROIs = list()
    with ZipFile(filePath,mode='r') as file:
        for ROIFile in file.namelist():
            (ROICoordinates,ImageJDict) = readImageJROI(file.read(ROIFile),isDataStream=True)
            ROIs.append(ROICoordinates)
    return ROIs

def writeImageJROIsFromZip(filePath,ROIs,useCenterOfMass=True):
    "Writes ImageJ ROIs to a *.zip file"
    
    with ZipFile(filePath,mode='w') as file:
        for (index,ROI) in enumerate(ROIs):
            # Define ImageJ ROI Name
            if useCenterOfMass:
                centerOfMass = np.median(ROI,axis=0)
                ROIName = "{:04d}-{:04d}.roi".format(int(centerOfMass[0]),int(centerOfMass[1]))
            else:
                ROIName = "ROI{:04d}".format(index)
            
            # Write byte data to an ImageJ .ROI file
            byteData = writeImageJROI(ROICoordinates=ROI,streamData=True)
            file.writestr(ROIName,b" ".join(byteData))