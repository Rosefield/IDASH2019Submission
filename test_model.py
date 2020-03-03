import numpy as np
import pandas as pd
import os
import argparse
import json
import tensorflow.keras as k

def readData(tumorFileName, normalFileName):
    x_true = pd.read_csv(tumorFileName, sep='\t', header=0, index_col=0).T
    x_false = pd.read_csv(normalFileName, sep='\t', header=0, index_col=0).T

    # if this data set has some nulls fill it so it does not crash the program
    x = pd.concat([x_true, x_false]).fillna(0)
    y = np.zeros(x.shape[0])
    y[:x_true.shape[0]] = 1

    return x,y

def readHardcodedFeatures():
    with open('gse_hardcoded_1000.txt') as f:
        data = json.load(f)
        return { int(c): np.array(idx) for c,idx in data.items() }

def hardcodeFeatureSelection(X,y, count):
    counts_features = readHardcodedFeatures()

    return X.values[:, counts_features[count]]

def predictWithWeights(Xte, yte, modelFileName):

    model = k.models.load_model(modelFileName)

    pred = np.round(model.predict(Xte)).reshape(Xte.shape[0])
    print('Predicted ', pred)
    print('Real ', yte)
    n = (pred == yte).sum()
    acc = n/Xte.shape[0]
    print('Total correct {} out of {}: {}%'.format(n, Xte.shape[0], acc))

def main():
    parser = argparse.ArgumentParser(description='Run secure MPC training')
    parser.add_argument('outputFile', default='Model.hdf5', help='Filename to that contains the model weights')
    parser.add_argument('positiveClassFile', help='The file that contains positive samples')
    parser.add_argument('negativeClassFile', help='The file that contains negative samples')

    args = parser.parse_args()
    
    print('Reading data from {} {}'.format(args.positiveClassFile, args.negativeClassFile))
    X,y = readData(args.positiveClassFile, args.negativeClassFile)
    X = hardcodeFeatureSelection(X, y, 1000)

    print('Running prediction with pretrained model')
    output_file = args.outputFile
    p = predictWithWeights(X, y, output_file)

if __name__ == "__main__":
    main()
