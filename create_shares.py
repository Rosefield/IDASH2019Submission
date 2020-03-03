import numpy as np
import pandas as pd
import argparse
import os
import time
import json
import socket

def readData(tumorFileName, normalFileName):
    x_true = pd.read_csv(tumorFileName, sep='\t', header=0, index_col=0).T
    x_false = pd.read_csv(normalFileName, sep='\t', header=0, index_col=0).T

    # if this data set has some nulls fill it so it does not crash the program
    x = pd.concat([x_true, x_false]).fillna(0)
    y = np.zeros(x.shape[0])
    y[:x_true.shape[0]] = 1

    p = np.random.permutation(x.shape[0])

    return x.iloc[p], y[p]

def readHardcodedFeatures():
    with open('gse_hardcoded_1000.txt') as f:
        data = json.load(f)
        return { int(c): np.array(idx) for c,idx in data.items() }

def hardcodeFeatureSelection(X,y, count):
    counts_features = readHardcodedFeatures()

    return X.values[:, counts_features[count]]

def encodeVals(vals):
    # this needs to match what is in securenn unless that is changed to take in the share values directly instead of reading floats
    precision = 20
    return (vals * (1 << precision)).astype(np.uint64)

def createSharesOf(vals):
    rands = np.random.randint(0, 18446744073709551615, size=vals.shape, dtype=np.uint64)
    s = encodeVals(vals) - rands
    return pd.DataFrame(rands), pd.DataFrame(s)

def xy_to_net(X, y):
    Xr = X.to_records(index=False)
    yr = y.to_records(index=False)
    tX = Xr.dtype
    ty = yr.dtype
    dX = Xr.tobytes()
    dy = yr.tobytes()

    lX = len(dX).to_bytes(4, byteorder='big')
    ly = len(dy).to_bytes(4, byteorder='big')
    
    data = lX + ly + dX + dy

    return data, tX, ty

def net_to_xy(data, tX, ty):
    lX = int.from_bytes(data[:4], byteorder='big')
    ly = int.from_bytes(data[4:8], byteorder='big')
    
    X = pd.DataFrame(np.frombuffer(data[8:8+lX], dtype=tX))
    y = pd.DataFrame(np.frombuffer(data[8+lX:8+lX+ly], dtype=ty))
    
    return X,y
    

def sendAndReceiveShares(party_id, partyIPFile, port, X1, X2, y1, y2, zx, zy):
    ips = open(partyIPFile).read().splitlines()

    data = ((X1, y1),(X2, y2), (zx, zy))

    
    data_strings = [xy_to_net(X,y) for X,y in data]

    data_array = [(),(),()]
    data_array[party_id] = data[party_id]

    for id in range(3):
        if id == party_id:
            print("sending", id)
            prev_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            next_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            m1 = (party_id - 1) % 3
            p1 = (party_id + 1) % 3
            try:
                prev_socket.bind((ips[id], port + 3*party_id + m1))
                prev_socket.listen()
                clientsocket, addr = prev_socket.accept()
                clientsocket.sendall(data_strings[m1][0])
                clientsocket.close()

                next_socket.bind((ips[id], port + 3*party_id + p1))
                next_socket.listen()
                clientsocket, addr = next_socket.accept()
                clientsocket.sendall(data_strings[p1][0])
                clientsocket.close()
            except socket.error as error:
                print("Error:", error)
            prev_socket.close()
            next_socket.close()
                    
        else:
            count = 0
            while count < 100:
                try:
                    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    recv_socket.connect((ips[id], port + 3*id + party_id))
                    break
                except socket.error as error:
                    time.sleep(1)
                    #print("Error:", error)
                    count = count + 1
            recvd_data = b''
            while True:
                data_chunk = recv_socket.recv(8192)
                if not data_chunk:
                    break
                recvd_data += data_chunk
            other_data = net_to_xy(recvd_data, data_strings[id][1], data_strings[id][2])
            recv_socket.close()
            data_array[id] = other_data

    return data_array

def concatenateDataframes(pair1, pair2, pair3):
    val1_1 = pair1[0]
    val1_2 = pair1[1]
    val2_1 = pair2[0]
    val2_2 = pair2[1]
    val3_1 = pair3[0]
    val3_2 = pair3[1]
    val1_1.append(val2_1)
    val1_1.append(val3_1)
    val1_2.append(val2_2)
    val1_2.append(val3_2)
    return (val1_1, val1_2)

def produceConcatenatedShares(data_array):
    tup1 = data_array[0]
    tup2 = data_array[1]
    tup3 = data_array[2]
    X, y = concatenateDataframes((tup1[0], tup1[1]), (tup2[0], tup2[1]), (tup3[0], tup3[1]))
    return X,y

def createShareFiles(party_id, partyIPFile, port, X, y, directory='tmp/'):
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    data_filename = os.path.abspath(os.path.join(directory, 'data_shares_' + str(party_id + 1)))
    label_filename = os.path.abspath(os.path.join(directory, 'label_shares_' + str(party_id + 1)))

    X1, X2 = createSharesOf(X)
    y1, y2 = createSharesOf(y)
    zx = pd.DataFrame(np.zeros_like(X, dtype=int), index=X.index, columns=X.columns)
    zy = pd.DataFrame(np.zeros_like(y, dtype=int), index=y.index, columns=y.columns)
    
    # get shares from other two parties
    data_array = sendAndReceiveShares(party_id, partyIPFile, port, X1, X2, y1, y2, zx, zy)

    X, y = produceConcatenatedShares(data_array)
    X.to_csv(data_filename, sep='\t', index=False, header=False)
    y.to_csv(label_filename, sep='\t', index=False, header=False)

    return [data_filename, label_filename]

def main():
    parser = argparse.ArgumentParser(description='Create share files')
    parser.add_argument('--base-port', dest='basePort', default=32000, type=int, help='The base port to use for all parties. each party will listen on the base + some offset')
    parser.add_argument('partyID', type=int, help='The id of this party')
    parser.add_argument('partyIPFile', help='A file containing the IP addresses of each party')
    parser.add_argument('positiveClassFile', help='The file that contains positive samples')
    parser.add_argument('negativeClassFile', help='The file that contains negative samples')
    args = parser.parse_args()

    print('Reading data from {} {}'.format(args.positiveClassFile, args.negativeClassFile))
    X,y = readData(args.positiveClassFile, args.negativeClassFile)
    X = hardcodeFeatureSelection(X, y, 1000)

    assert(args.partyID > 0 and args.partyID < 4)
    print('Creating share files for party ', args.partyID)
    shareFileNames = createShareFiles(args.partyID - 1, args.partyIPFile, args.basePort, X, y)
    print('Files saved at ', shareFileNames)

    return 0

if __name__ == "__main__":
    main()
