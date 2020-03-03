set -e

# expected run is sh ./run_all.sh partyId trainPositiveFile trainNegativeFile testPositiveFile testNegativeFile

python3 create_shares.py --base-port 31000 $1 securenn-public/files/parties $2 $3
python3 run_application.py --base-port 32000 $1 tmp/data_shares_$1 tmp/label_shares_$1
if [ $1 == "1" ]; then python3 test_model.py Model.hdf5 $4 $5; fi
