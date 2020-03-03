set -e

python3 create_shares.py --base-port 31000 1 securenn-public/files/parties data/GSE2034-Tumor-train.data data/GSE2034-Normal-train.data &
python3 create_shares.py --base-port 31000 2 securenn-public/files/parties data/GSE2034-Empty.data data/GSE2034-Empty.data &
python3 create_shares.py --base-port 31000 3 securenn-public/files/parties data/GSE2034-Empty.data data/GSE2034-Empty.data 
python3 run_application.py --base-port 32000 3 tmp/data_shares_3 tmp/label_shares_3 2>/dev/null &
sleep 1s
python3 run_application.py --base-port 32000 2 tmp/data_shares_2 tmp/label_shares_2 2>/dev/null &
sleep 1s
python3 run_application.py --base-port 32000 1 tmp/data_shares_1 tmp/label_shares_1
python3 test_model.py Model.hdf5 data/GSE2034-Tumor-test.data data/GSE2034-Normal-test.data
