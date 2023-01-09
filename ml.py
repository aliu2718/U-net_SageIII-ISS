"""Module containing useful functions for the U-net projects

This module supplements both U-net projects (transmission profiles and PCA principle axes). It contains
many useful functions for processing data, creating different masks, generating labels, initializing the
U-net, and visualizing the results.

Andrew Liu
Last Modified: August 12, 2022
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
 
import re

import tensorflow as tf


def getClassBounds(arr):
    """Returns a list of all float boundaries for ranges specified in a list as strings

    Examples: 
        For ['12.5', '13.0-15.5'], the function returns [[12.5], [13.0, 15.5]]
        For ['13.0'], the function returns [[13.0]]
        For ['24.2-25.2'], the function returns [[24.2, 25.2]]

    Parameters
    ----------
    arr : list
        List of strings specifying single float values or a range of floats (all floats must have a decimal point)

    Returns
    -------
    list
        List of lists specifying float boundaries 
    """
    bounds = []
    for str in arr:
        try:
            bounds.append([float(str)])
        except:
            for a, b in re.findall(r'(\d+\.\d+)-?(\d+\.\d*)', str):
                bounds.append([float(a), float(b)])

    return bounds

def getClassification(identifier, events, type='clouds'):
    """Returns a list of all boundaries for the specified atmospheric layer type of the event matching the specified time or event id

    Parameters
    ----------
    identifier : str
        time or event id of the event
    events : dict
        All events contained in the event JSON file
    type : str, optional
        type of layer to get ranges for (either 'clouds' or 'aerosols'), by default 'clouds'

    Returns
    -------
    list
        List of lists specifying boundaries for the specified atmospheric layer type of the specified event
    """
    index = next((i for i, x in enumerate(events['training']) if x['time'] == identifier or x['event_id'] == identifier), None)
    return getClassBounds(events['training'][index][type])

def getDataAsMatrix(t, sage):
    """Returns the transmission profile at the specified time as a 2d matrix

    Parameters
    ----------
    t : str
        time of event
    sage : xarray.Dataset
        SAGE dataset

    Returns
    -------
    2d matrix of floats
        transmission profile at the specified time
    """
    return sage.Transmission.sel(time=t).values

def deNaN(matrix, rand=True, isTransmission=True):
    """Removes all NaN values in the matrix

    Parameters
    ----------
    matrix : 2d array 
        matrix to remove NaN values from
    rand : bool, optional
        whether or not to replace NaN values with random floats between 0 and 1 (if false, NaN values will 
        be replaced with 2 if the matrix is a transmission profile and with 1 otherwise), by default True
    isTransmission : bool, optional
        whether or not the matrix is a transmission profile, by default True
    """
    if rand: 
        rand_matrix = np.random.rand(matrix.shape[0], matrix.shape[1])
        matrix[np.isnan(matrix)] = rand_matrix[np.isnan(matrix)]
    else:
        matrix[np.isnan(matrix)] = 2 if isTransmission else 1

def padMatrix(matrix, newshape):
    """Returns the matrix padded with zeroes

    With an odd number of padding, the top and left will be padded with one less zero. Otherwise, the
    padding is even vertically and horizontally.

    Parameters
    ----------
    matrix : 2d array
        matrix to pad
    newshape : 2-tuple of ints
        new shape of matrix, each dimension must not be less than the corresponding dimension of the original matrix

    Returns
    -------
    2d array
        matrix padded with zeroes
    """
    vert_dif = newshape[0] - matrix.shape[0]
    hor_dif = newshape[1] - matrix.shape[1]
    
    top_pad = int(np.floor(vert_dif / 2))
    left_pad = int(np.floor(hor_dif / 2))

    padded_matrix = np.pad(matrix, ((top_pad, vert_dif - top_pad), (left_pad, hor_dif - left_pad)), 'constant')

    return padded_matrix

def padAll(m_list, newshape):
    """Returns a list of matrices all padded with zeroes

    Parameters
    ----------
    m_list : list of 2d arrays
        list of matrices that must all be padded
    newshape : 2-tuple of ints
        new shape of all matrices, each dimension must not be less than the corresponding dimension of any matrix in the list

    Returns
    -------
    list of 2d arrays
        list of matrices, all padded with zeroes
    """
    pad_list = []
    for m in m_list:
        pad_list.append(padMatrix(m, newshape))
    
    return pad_list

def truncMatrix(matrix, newshape):
    """Returns a matrix truncated to the specified shape

    With an odd number of truncation, the top and left will be truncated by one less. Otherwise, the
    truncation is even vertically and horizontally. This is an inverse function to padMatrix.

    Parameters
    ----------
    matrix : 2d array
        matrix to truncate
    newshape : 2-tuple of ints
        new shape of matrix, each dimension must not be greater than the corresponding dimension of the original matrix

    Returns
    -------
    2d array
        truncated matrix
    """
    vert_dif = matrix.shape[0] - newshape[0]
    hor_dif = matrix.shape[1] - newshape[1]
    
    top_trunc = int(np.floor(vert_dif / 2))
    left_trunc = int(np.floor(hor_dif / 2))

    trunc_matrix = matrix[top_trunc:matrix.shape[0] - (vert_dif - top_trunc), left_trunc:matrix.shape[1] - (hor_dif - left_trunc)]

    return trunc_matrix

def createNaNBitmask(matrix, shape):
    """Returns a mask for the NaN values in the matrix

    For all indices where a NaN value exists in the matrix, ones are set in the corresponding indices in the NaN mask.
    All other indices are zero, which indicate indices that are not NaN in the matrix.

    Parameters
    ----------
    matrix : array or multidimensional array
        matrix to create NaN mask for
    shape : tuple of ints
        shape of matrix

    Returns
    -------
    array or multidimensional array
        NaN mask for matrix
    """
    bitmask = np.zeros(shape)
    bitmask[np.isnan(matrix)] = matrix[np.isnan(matrix)]
    bitmask[np.isnan(bitmask)] = 1

    return bitmask

def trop_altGenerator(event, shape, sage):
    """Returns a mask representing the absolute difference between each altitude and the tropopause altitude

    Parameters
    ----------
    event : dict
        event information from event JSON file
    shape : 2-tuple of ints
        unpadded shape of input
    sage : xarray.Dataset
        SAGE dataset

    Returns
    -------
    2d array
        tropopause altitude difference mask
    """
    trop_alt_doubled = float(sage.trop_alt.sel(time=event['time'])) * 2
    map = np.zeros(shape)
    norm_scale = np.abs(trop_alt_doubled - shape[0])
    for i in range(shape[0]):
        map[i] = [np.abs(trop_alt_doubled - i - 1) / norm_scale] * shape[1]

    return map

def uncGenerator(event, sage):
    """Returns an absolute uncertainty mask for the specified event with NaN values removed

    All NaN values are replaced with ones.

    Parameters
    ----------
    event : dict
        event information from event JSON file
    sage : xarray.Dataset
        SAGE dataset

    Returns
    -------
    array
        absolute uncertainty mask with NaN values removed
    """
    map = np.array(sage.Transmission_unc.sel(time=event['time'])).astype('float32')

    map = deNaN(map, False, False)
    return map

def labelGenerator(event, matrix, shape, num_class=2):
    """Returns a label for the specified event and matrix

    The number of unique labels depends on the number of classes. With two classes, there is only a
    binary classification of cloud and not cloud. With three classes, the labels become cloud, 
    NaN values, and other. With four classes, there is an addition of an aerosol label. 
    
    The label is specific to each first dimension across the entire second dimension (in the context
    of transmission data, the label is assigned to the altitude and is uniform across all wavelength
    channels).

    Parameters
    ----------
    event : dict
        event information from event JSON file
    matrix : 2d array
        matrix of the specified shape corresponding to the event, specifically with NaN values to label
    shape : 2-tuple of ints
        shape
    num_class : int, optional
        number of label classes, by default 2

    Returns
    -------
    2d array
        label for the event and matrix
    """
    map = np.ones(shape)

    # Label aerosol layers
    if num_class >= 4:
        aers = getClassBounds(event['aerosols'])
        for aer in aers:
            if len(aer) == 1:
                map[int(aer[0] * 2 - 1)] = [3] * shape[1]
            else:
                incr = int(aer[0] * 2 - 1)
                while incr <= (aer[-1] * 2 - 1):
                    map[incr] = [3] * shape[1]
                    incr += 1

    # Label cloud layers
    clouds = getClassBounds(event['clouds'])
    for alt in clouds:
        if len(alt) == 1:
            map[int(alt[0] * 2 - 1)] = [0] * shape[1]
        else:
            incr = int(alt[0] * 2 - 1)
            while incr <= (alt[-1] * 2 - 1):
                map[incr] = [0] * shape[1]
                incr += 1
    
    # Label NaN values
    if num_class >= 3:
        map[np.isnan(matrix)] = 2
    
    return map

def create_unet(shape, num_class=2):
    """Returns a U-net model

    The U-net model is a slightly modified version of the model implemented in:
    https://keras.io/examples/vision/oxford_pets_image_segmentation/

    Parameters
    ----------
    shape : 3-tuple of ints
        total shape of input
    num_class : int, optional
        number of label classes, by default 2

    Returns
    -------
    tf.keras.Model
        a U-net model
    """
    inputs = tf.keras.Input(shape=(shape[0], shape[1], shape[-1]))

    # Downsampling Inputs

    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x

    for filters in [64, 128, 256]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)

        x = tf.keras.layers.add([x, residual])
        previous_block_activation = x

    # Upsampling Inputs

    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.UpSampling2D(2)(x)

        residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.add([x, residual])
        previous_block_activation = x

    outputs = tf.keras.layers.Conv2D(num_class, 3, activation="softmax", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    tf.keras.backend.clear_session()

    return model

def plot_perf(history):
    """Creates an epoch vs. accuracy plot for the model's training and validation accuracies 

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Callback object with recorded events of model.fit()
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

def multiclassifier(output_tensor, num_class=2):
    """Returns a list of all visually transformed labels

    The maximum probability is found for each pixel of each prediction, and the pixel value is set accordingly based on the 
    value associated with the label with highest probability. Here, a value of 0 corresponds to cloud, a value of 1 corresponds
    to other, a value of 2 corresponds to NaN (for the transmission project, otherwise aerosol for the PCA principle axes project), 
    and a value of 3 corresponds to aerosol for the transmission project. 

    Parameters
    ----------
    output_tensor : numpy array
        outputted predictions of model.predict()
    num_class : int, optional
        number of label classes, by default 2

    Returns
    -------
    list
        list of visually transformed labels corresponding to the model predictions
    """
    classified_outputs = []
    shape = output_tensor.shape[1:-1]
    for data in output_tensor:
        map = np.empty(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                max_val = max(data[i][j])
                max_found = False
                for k in range(num_class - 2):
                    if max_val == data[i][j][k+2]:
                        map[i][j] = data[i][j][k+2] + k + 1
                        max_found = True
                        break
                
                if not max_found: map[i][j] = data[i][j][1]
        
        classified_outputs.append(map)
    
    return classified_outputs