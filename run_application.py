import numpy as np
import pandas as pd
import os
#attempt to silence TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import subprocess
import argparse
import tensorflow.keras as k
import json
import time
from tensorflow.keras.layers import Dense 
# try to silence all of the potential python warnings
import warnings
warnings.filterwarnings('ignore')
#import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)

def createSimpleNN(dim, layerSize=100):
    '''
    Creates a simple neural net composed of two FC layers with ReLU activation functions and then a sigmoid
    '''
    def build():
        model = k.models.Sequential()
        model.add(Dense(layerSize, activation='relu', input_shape=(dim,)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return model
    
    model = build()
    return model

def createSimplestNN(dim):
    '''
    Creates a simple neural net composed of two FC layers with a ReLU activation function and then a sigmoid
    '''
    def build():
        model = k.models.Sequential()
        model.add(Dense(100, activation='relu', input_shape=(dim,)))
        model.add(Dense(1, activation='sigmoid'))

        return model
    
    model = build()
    return model

create_models = { 
        'SimpleNN100' : lambda dim: createSimpleNN(dim, 100),
        'SimplestNN' : lambda dim: createSimplestNN(dim)
        }

def runSecureNN(modelName, files, partyId, port, epochs=30):
    t = int(time.time())
    outputFile = os.path.abspath("tmp/{}_{}_{}.txt".format(modelName, partyId, t))
    files = [os.path.abspath(f) for f in files]

    ops = []
    keyFiles = { 0 : './files/keyA ./files/keyAB', 1 : './files/keyB ./files/keyAB', 2 : './files/keyC ./files/keyC' }
    trainFiles = files
    ops.append('cd securenn-public && ./BMRPassive.out {} {} 3PC {} ./files/parties {} {} {} {}'.format(modelName, epochs, partyId, keyFiles[partyId], ' '.join(trainFiles), outputFile, port))

    procs = []
    for op in ops:
        print('Running: ', op)
        proc = subprocess.Popen(op, shell=True)
        procs.append(proc)

    comms = []
    for proc in procs:
        proc.wait()
        assert(proc.returncode == 0)

    return outputFile

def convertSecureNNModel(modelName, model_file, output_file):
    with open(model_file) as f:
        arrays = json.load(f)
        weights = [np.array(a, dtype=np.float32) for a in arrays]

    model = create_models[modelName](weights[0].shape[0])

    model.set_weights(weights)
    k.models.save_model(model, output_file)

def main():
    parser = argparse.ArgumentParser(description='Run secure MPC training')
    parser.add_argument('--model', default='SimpleNN100', help='The model to train under MPC')
    parser.add_argument('--epochs', default=30, type=int, help='The number of epochs to train the model for')
    parser.add_argument('--output-file', dest='output_file', default='Model.hdf5', help='Filename to output the model weights to')
    parser.add_argument('--base-port', dest='basePort', default=32000, type=int, help='The base port to use for all parties. each party will listen on the base + some offset')
    parser.add_argument('partyId', type=int, help='Party id for this server')
    parser.add_argument('dataFile', help='The file that contains shares of data')
    parser.add_argument('labelsFile', help='The file that contains shares of labels')

    args = parser.parse_args()

    assert(args.partyId > 0 and args.partyId < 4)
    shareFileNames = [args.dataFile, args.labelsFile]
    print('Running MPC')
    tmpModelFile = runSecureNN(args.model, shareFileNames, args.partyId - 1, args.basePort, args.epochs)

    output_file = args.output_file
    if(args.partyId == 1):
        print('Converting model to keras model, saved at ', output_file)
        convertSecureNNModel(args.model, tmpModelFile, output_file)


if __name__ == "__main__":
    main()

