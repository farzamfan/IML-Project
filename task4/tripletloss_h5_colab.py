# from https://github.com/noelcodella/tripletloss-keras-tensorflow
# Noel C. F. Codella
# Example Triplet Loss Code for Keras / TensorFlow

# Implementing Improved Triplet Loss from:
# Zhang et al. "Tracking Persons-of-Interest via Adaptive Discriminative Features" ECCV 2016

# Got help from multiple web sources, including:
# 1) https://stackoverflow.com/questions/47727679/triplet-model-for-image-retrieval-from-the-keras-pretrained-network
# 2) https://ksaluja15.github.io/Learning-Rate-Multipliers-in-Keras/
# 3) https://keras.io/preprocessing/image/
# 4) https://github.com/keras-team/keras/issues/3386
# 5) https://github.com/keras-team/keras/issues/8130


# GLOBAL DEFINES
T_G_WIDTH = 224 
T_G_HEIGHT = 224 
T_G_NUMCHANNELS = 3
T_G_SEED = 1337

# Misc. Necessities
import sys
import ssl # these two lines solved issues loading pretrained model
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
import h5py
import random
np.random.seed(T_G_SEED)
random.seed(T_G_SEED)

# TensorFlow Includes
import tensorflow as tf
#from tensorflow.contrib.losses import metric_learning
tf.random.set_seed(T_G_SEED)

# Keras Imports & Defines 
import keras
import keras.applications
from keras import backend as K
from keras.models import Model
from keras import optimizers
import keras.layers as kl

# Local Imports
from LR_SGD import LR_SGD


def createModel(emb_size):

    #MobileNetV2 as base with pretrained weights, don't train the MobileNet weights
    mobilenet_input = kl.Input(shape=(T_G_HEIGHT,T_G_WIDTH,T_G_NUMCHANNELS))
    mobilenet_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=mobilenet_input)
    mobilenet_model.trainable = False 

    net = mobilenet_model.output 
    net = kl.GlobalAveragePooling2D(name='gap')(net)
    #net = kl.Dense(1000,activation='relu',name='dense_after_globalavg')(net) #I think that adds 30m parameters...
    net = kl.Dropout(0.4)(net)
    net = kl.Dense(emb_size,activation='relu',name='t_emb_1')(net)
    net = kl.Lambda(lambda  x: K.l2_normalize(x,axis=1), name='t_emb_1_l2norm')(net)

    # model creation
    base_model = Model(mobilenet_input, net, name="base_model")
    #base_model = Model(mynet_input, mynet_model, name="base_model")

    # triplet framework, shared weights
    input_shape=(T_G_HEIGHT,T_G_WIDTH,T_G_NUMCHANNELS)
    input_anchor = kl.Input(shape=input_shape, name='input_anchor')
    input_positive = kl.Input(shape=input_shape, name='input_pos')
    input_negative = kl.Input(shape=input_shape, name='input_neg')

    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)

    # The Lamda layer produces output using given function. Here its Euclidean distance.
    positive_dist = kl.Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = kl.Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])
    tertiary_dist = kl.Lambda(euclidean_distance, name='ter_dist')([net_positive, net_negative])

    # This lambda layer simply stacks outputs so all distances are available to the objective
    stacked_dists = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])

    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.00001), loss=triplet_loss, metrics=[accuracy])

    return model


def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    #return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - 0.5*(K.square(y_pred[:,1,0])+K.square(y_pred[:,2,0])) + margin))
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin)) #we don't need that positives and negatives are far apart

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# loads a set of images from a hdf5 file using a list of indices  
def t_read_image_list(img_indices, hdf5_file, start, length):

    datalen = length
    if (datalen < 0):
        datalen = len(img_indices)

    if (start + datalen > len(img_indices)):
        datalen = len(img_indices) - start

    img_indices_to_extract = img_indices[start:(start+datalen)]
    f = h5py.File(hdf5_file,mode='r')

    args_sorted = np.argsort(img_indices_to_extract)
    img_indices_to_extract_sorted = img_indices_to_extract[args_sorted]
    img_indices_to_extract_unique_sorted = np.unique(img_indices_to_extract_sorted)

    #h5py can only extract sorted and unique lists
    imgset_unique_sorted = f['imgs'][img_indices_to_extract_unique_sorted,:,:,:]

    imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, T_G_NUMCHANNELS))

    # "unsort and unpack duplicates" 
    for i_sort, i_unsort in enumerate(args_sorted):

        i_unique_sorted = np.where(img_indices_to_extract_unique_sorted==img_indices_to_extract_sorted[i_sort])
        imgset[i_unsort,:,:,:] = imgset_unique_sorted[i_unique_sorted,:,:,:]

    f.close()

    return imgset


def extract(argv):

    if len(argv) < 4:
        print('Usage: \n\t <Model Prefix> <Input Image List (list)> <Output File (list)> <hdf5_file> \n\t\tExtracts triplet-loss model')
        return

    modelpref = argv[0]
    imglist = argv[1]
    outfile = argv[2]
    hdf5_file = argv[3]

    with open(modelpref + '.json', "r") as json_file:
        model_json = json_file.read()

    loaded_model = keras.models.model_from_json(model_json)
    loaded_model.load_weights(modelpref + '.h5')

    base_model = loaded_model.get_layer('base_model')

    # create a new single input
    input_shape=(T_G_HEIGHT,T_G_WIDTH,T_G_NUMCHANNELS)
    input_single = kl.Input(shape=input_shape, name='input_single')
    
    # create a new model without the triple loss
    net_single = base_model(input_single)
    model = Model(input_single, net_single, name='embedding_net')

    chunksize = 1000
    total_img = len(imglist)
    total_img_ch = int(np.ceil(total_img / float(chunksize)))

    with open(outfile, 'w') as f_handle:

        for i in range(0, total_img_ch):
            imgs = t_read_image_list(imglist, hdf5_file, i*chunksize, chunksize)

            #preprocess with mobilenet
            imgs = keras.applications.mobilenet_v2.preprocess_input(imgs)

            vals = model.predict(imgs)
    
            np.savetxt(f_handle, vals)


    return



def learn(argv):
    
    if len(argv) < 11:
        print('Usage: \n\t <Train Anchors (list)> <Train Positives (list)> <Train Negatives (list)> <Val Anchors (list)> <Val Positives (list)> <Val Negatives (list)> <embedding size> <batch size> <num epochs> <output model> <hdf5_file> \n\t\tLearns triplet-loss model')
        return
    print('new')
    in_t_a = argv[0]
    in_t_b = argv[1]
    in_t_c = argv[2]

    in_v_a = argv[3]
    in_v_b = argv[4]
    in_v_c = argv[5]

    emb_size = int(argv[6])
    batch = int(argv[7])
    numepochs = int(argv[8])
    outpath = argv[9] 
    hdf5_file = argv[10]

    # chunksize is the number of images we load from disk at a time
    chunksize = batch*10 #changed from *100 to *10 because colab cannot hadnle more RAM-wise
    total_t = len(in_t_a)
    total_v = len(in_v_b)
    total_t_ch = int(np.ceil(total_t / float(chunksize)))
    total_v_ch = int(np.ceil(total_v / float(chunksize)))

    print('Dataset has ' + str(total_t) + ' training triplets, and ' + str(total_v) + ' validation triplets.')

    print('Creating a model ...')
    model = createModel(emb_size)

    print('Training loop ...')
    
    # manual loop over epochs to support very large sets of triplets
    for e in range(0, numepochs):

        for t in range(0, total_t_ch):

            print('Epoch ' + str(e) + ': train chunk ' + str(t+1) + '/ ' + str(total_t_ch) + ' ...')

            print('Reading image lists ...')
            #preprocess with mobilenet
            anchors_t = keras.applications.mobilenet_v2.preprocess_input(t_read_image_list(in_t_a, hdf5_file, t*chunksize, chunksize))
            positives_t = keras.applications.mobilenet_v2.preprocess_input(t_read_image_list(in_t_b, hdf5_file, t*chunksize, chunksize))
            negatives_t = keras.applications.mobilenet_v2.preprocess_input(t_read_image_list(in_t_c, hdf5_file, t*chunksize, chunksize))
            Y_train = np.random.randint(2, size=(1,2,anchors_t.shape[0])).T #random because not needed to calculate loss

            print('Starting to fit ...')
            
            model.fit([anchors_t, positives_t, negatives_t], Y_train, epochs=1,  batch_size=batch) #epochs 1 because running epochs manually
        
        # In case the validation images don't fit in memory, we load chunks from disk again. 
        val_res = [0.0, 0.0]
        total_w = 0.0
        for v in range(0, total_v_ch):

            print('Loading validation image lists ...')
            print('Epoch ' + str(e) + ': val chunk ' + str(v+1) + '/ ' + str(total_v_ch) + ' ...')
            #preprocess with mobilenet
            anchors_v = keras.applications.mobilenet_v2.preprocess_input(t_read_image_list(in_v_a, hdf5_file, v*chunksize, chunksize))
            positives_v = keras.applications.mobilenet_v2.preprocess_input(t_read_image_list(in_v_b, hdf5_file, v*chunksize, chunksize))
            negatives_v = keras.applications.mobilenet_v2.preprocess_input(t_read_image_list(in_v_c, hdf5_file, v*chunksize, chunksize))
            Y_val = np.random.randint(2, size=(1,2,anchors_v.shape[0])).T #random because not needed to calculate loss

            # Weight of current validation measurement. 
            # if loaded expected number of items, this will be 1.0, otherwise < 1.0, and > 0.0.
            w = float(anchors_v.shape[0]) / float(chunksize)
            total_w = total_w + w

            curval = model.evaluate([anchors_v, positives_v, negatives_v], Y_val, batch_size=batch)
            val_res[0] = val_res[0] + w*curval[0]
            val_res[1] = val_res[1] + w*curval[1]

        val_res = [x / total_w for x in val_res] #first entry is validation loss, second validation accuracy

        print('Validation Results: ' + str(val_res))

        # shuffle data after each epoch...
        temp = list(zip(in_t_a, in_t_b, in_t_c))
        random.shuffle(temp)
        in_t_a, in_t_b, in_t_c = zip(*temp)
        in_t_a = np.asarray(in_t_a)
        in_t_b = np.asarray(in_t_b)
        in_t_c = np.asarray(in_t_c)

    print('Saving model ...')

    # Save the model and weights
    model.save(outpath + '.h5')

    # Due to some remaining Keras bugs around loading custom optimizers
    # and objectives, we save the model architecture as well
    model_json = model.to_json()
    with open(outpath + '.json', "w") as json_file:
        json_file.write(model_json)

    return


