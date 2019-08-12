# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:11:52 2019

@author: dwhitney
"""
from importlib import reload # Necessary for reloadding the logging handler
import logging
import os
import sys

DEFAULT_LOGGING_LOCATION = os.path.join(os.getcwd(),'loggingFile.log')
LOGGING_OPTIONS = {} # DEFAULT LOGGER OPTIONS DICTIONARY
LOGGING_OPTIONS.update({'writeToStdOut': True})  # Displays logs on console
LOGGING_OPTIONS.update({'writeToLogFile': True}) # Writes logs to a file
LOGGING_OPTIONS.update({'interceptStdOut': False})  # Will take std. out "print" commands as input
LOGGING_OPTIONS.update({'loggingFilePath': DEFAULT_LOGGING_LOCATION}) # Logging location
LOGGING_OPTIONS.update({'stringFormat':'%(asctime)s %(message)s'}) #'%(asctime)s%(levelname)s%(message)s'

def updateDictionary(srcDict,refDict):
    updatedDict = dict(refDict)
    for key in refDict.keys():
        if key in srcDict.keys():
            updatedDict[key]=srcDict[key]
    return updatedDict 

class logger():
    def __init__(self,options=LOGGING_OPTIONS):
        self.setup(options)
        
    def __del__(self):
        self.close()
        
    def setup(self,options=LOGGING_OPTIONS):
        "Setup event handlers. Can ouput logs to StdOut and/or write a log file"
        
        # Setup logger handlers
        handlers = []
        if(options['writeToStdOut']):
            handlers.append(logging.StreamHandler(sys.stdout))
        if(options['writeToLogFile']):
            handlers.append(logging.FileHandler(options['loggingFilePath'])) 
        
        # Setup log formating
        formatter = logging.Formatter(options['stringFormat'],"%Y-%m-%d %H:%M:")
        for handler in handlers:
            handler.setFormatter(formatter)
            
        # Initialize logger
        reload(logging)
        logging.basicConfig(level=logging.INFO, handlers=handlers)
        self.logger  = logging.getLogger()
        self.options = options
        
        # Replace StdOut with logger object
        if(options['interceptStdOut']):
            self.stdout = sys.stdout
            sys.stdout = self
        
    def write(self,message):
        "Write text message to logging handlers"
        self.logger.info(message)
        
    def close(self):
        "Close existing handlers and replace logger with original StdOut (optional)"
        handlers = self.logger.handlers[:]
        for handler in handlers:
        	self.logger.removeHandler(handler)
        	handler.flush()
        	handler.close()
        if(self.options['interceptStdOut']):
            sys.stdout = self.stdout
        
if __name__ == "__main__":
    obj = logger()
    obj.logger.info('Test message')