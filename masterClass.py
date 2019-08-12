from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

DEFAULT_CNN_OPTIONS = {'nConvLayers':     3,      \
                       'nDenseLayers':    1,      \
                       'kernelSize':      [5,5],  \
                       'filters':         32,     \
                       'padding':         'same', \
                       'max_pool_kernel': [2,2],  \
                       'max_pool_stride': 2,      \
                       'nDenseNeurons':   1024,   \
                       'dropoutRate':     0.5,    \
                       'activationFunc':  tf.nn.relu};
DEFAULT_UNET_OPTIONS = {'nConvLayers':    3,   \
                        'dropoutRate':    0.5, \
                        'activationFunc': tf.nn.relu};   

def updateDictionary(srcDict,refDict):
    updatedDict = dict(refDict);
    for key in refDict.keys():
        if key in srcDict.keys():
            updatedDict[key]=srcDict[key];
    return updatedDict;                    
                       
nConvLayers    = 3
layerName      = ['ConvLayer{}'.format(layer) for layer in range(nConvLayers)]
filters        = [2**(index+6) for index in range(nConvLayers)]
kernelSize     = 3; # (kernelSize)x(kernelSize) kernel
stride         = [1,1,1,1];
activationFunc = tf.nn.relu;
dropoutRate    = 0.5;
paddingType    = 'same';

def setup_CNN_Model(inputArray, labels, inputOptions={}):
  """Model function for CNN."""
  # Store model layers
  modelLayers = [];
  options     = updateDictionary(inputOptions,DEFAULT_CNN_OPTIONS);
  
  # Sequentially initialize convolutional and pooling layers
  for layer in range(options['nConvLayers']):
      layerName = 'convLayer{}'.format(layer+1);
      if(layer==0):
          inputTensor = tf.reshape(inputArray, [-1, imgDims[0], imgDims[1], 1]);
      else:
          inputTensor = pool;
      conv = tf.layers.conv2d(inputs=inputTensor, filters=options['filters']*(layer+1), kernel_size=options['kernelSize'], padding=options['padding'], activation=options['activationFunc'])
      pool = tf.layers.max_pooling2d(inputs=conv, pool_size=options['max_pool_kernel'], strides=options['max_pool_stride'])
      modelLayers.append({'name':layerName,'conv': conv,'pool':pool});

  # Sequentially initialize dense layers (with dropout)
  for layer in range(options['nDenseLayers']):
      layerName = 'denseLayer{}'.format(layer+1);
      if(layer==0):
          inputTensor = tf.reshape(pool, [-1, imgDims[0]*imgDims[0] * options.filters*(layer+1)]);
      else:
          inputTensor = tf.reshape(dropout, [-1, options['nDenseNeurons']]);
      dense     = tf.layers.dense(inputs=inputTensor, units=options['nDenseNeurons'], activation=options['activationFunc'])
      dropout   = tf.layers.dropout(inputs=dense, rate=options['dropout'], training=mode == tf.estimator.ModeKeys.TRAIN)
      modelLayers.append({'name':layerName,'dense': dense,'dropout':dropout});

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)
  modelLayers.append({'name':layerName,'logits': logits});
  return layers
  

def runModel(model,mode):
    
    logits = model[-1]['logits'];
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
    
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Our application logic will be added here
if __name__ == "__main__":
    tf.app.run()