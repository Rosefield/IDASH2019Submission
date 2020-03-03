# General
This is a submission to IDASH 2019 (Track IV: Secure Collaborative Training of Machine Learning Model) by Schuyler Rosefield (Northeastern University), Aikaterini Sotiraki (MIT), Noah Luther (MIT Lincoln Laboratory), Emily Shen (MIT Lincoln Laboratory), and Vinod Vaikuntanathan (MIT). 

This repository contains code for secure 3-party computation to train a neural network to 1) determine if samples are cancerous or not, 2) determine if cancerous samples are recurrent tumors or not. This neural network is trained using genomic data. Below, the requirements and the instructions for running the application will be presented.

This repository contains code modified from SecureNN which was developed by Sameer Wagh, Divya Gupta, and Nishanth Chandran. The SecureNN code was modified heavily and unneeded features were removed. Citations to the SecureNN paper and the code repository are provided below.

# Requirements
The python scripts expect a working installation of tensorflow/keras and the supporting ecosystem (numpy, pandas, etc).
The MPC code expects a CPU with support for SSE4.1 and AES-NI instructions, and a compiler capable of compiling C++14.

# Running the application

Running the application is broken up into three parts: creating share files, running the MPC code, and calculating model accuracy. 
The expectation is that the train and test data files are in the same general format as was provided in the competition task, namely a separate file for "positive" and "negative" cases.

Scripts for each of the three phases are described below. The run_demo.sh script demonstrates an execution of all three phases
for sample data. It is not intended to be used for evaluation purposes.

The run_all.sh script can be used to run the full pipeline for each individual party with the following syntax

```
sh ./run_all.sh partyId path/to/tumor_train path/to/normal_train path/to/tumor_test path/to/normal_test
```

e.g.
```
sh ./run_all.sh 1 GSE2034-Tumor-train.data GSE2034-Normal-train.data GSE2034-Tumor-test.data GSE2034-Normal-test.data
```
Once the script is started on each of the servers for parties 1,2, and 3 the shares will be distributed and the training will run. Once the training is completed the model will be evaluated against the provided test data on the server run as party #1.

Each individual component and how they function are detailed below.

## Creating shares

Each party takes their local input data, creates secret shares, and distributes the shares to the other parties. The parties then concatenate all of the shares that they have receivedto run the MPC.

In order to create the files run the following command, substituting the paths to the data files that contain training data.

```
python3 create_shares.py partyId path/to/tumor_samples path/to/normal_samples
```
e.g.
```
python3 create_shares.py 1 GSE2034-Tumor-train.data GSE2034-Normal-train.data
```

This will output 2 files, of the form `{data,label}_shares_{partyId}`.


## Running the MPC

Once the share files are distributed to the servers the following command should be run on each server to start the MPC.

```
python3 run_application.py {partyId} path/to/data_shares_{partyId} path/to/label_shares_{partyId}
```
Where the partyId should match the number on the share files.
e.g.
```
python3 run_application.py 1 data_shares_1 label_shares_1
```

This will then output a keras model to `Model.hdf5`. To change where the model is saved, you can specify `--output-file {fileName}` as a parameter.

You can specify the number of epochs `--epochs {numOfEpochs}` as a parameter. The default value is 30.

By default the application trains a network with three fully connected layers. The first two layers have output dimension 100 and are followed by a ReLU, and the third one has output dimension 1 and is followed by a sigmoid function. 

We have also implemented a network with two fully connected layers that performs equally well on easier datasets (e.g., BC-TCGA) and requires less running time. You can train the two-layer model by adding `--model SimplestNN`.

```
python3 run_application.py {partyId} path/to/data_shares_{partyId} path/to/label_shares_{partyId} --model SimplestNN
```


## Calculating accuracy

Lastly, once the model has been created, run the following command.

```
python3 test_model.py {modelFile} path/to/test_data_positive /path/to/test_data_negative
```
e.g.

```
python3 test_model.py Model.hdf5 GSE2034-Tumor-test.data GSE2034-Normal-test.data
```

This will print the predictions on the test data and calculate the accuracy compared to expected.


## Configuration

The file `securenn-public/files/parties` specifies the IP addresses of the different parties with the first line being party 1, second party 2, etc. 
This file should match between all servers and the IP addresses should match with which servers are which parties. 
E.g., server with party id 2 should be the 2nd line in the file.
The application will attempt to use ports 31000-31009 for distributing the shares and ports 32000-32009 for the training to communicate between parties. The base port can be configured by specifying the `--base-port` option for the create_shares and run_application scripts.
 
#

## SecureNN Citation
You can cite the paper using the following bibtex entry:
```
@article{wagh2019securenn,
  title={{S}ecure{NN}: 3-{P}arty {S}ecure {C}omputation for {N}eural {N}etwork {T}raining},
  author={Wagh, Sameer and Gupta, Divya and Chandran, Nishanth},
  journal={Proceedings on Privacy Enhancing Technologies},
  year={2019}
}
```
The code repository can be found [here](https://github.com/snwagh/securenn-public). The repository includes a copy of the whitepaper.

## Distribution
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
 
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2019 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
