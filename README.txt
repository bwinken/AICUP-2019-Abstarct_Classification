How to reproduce train result?

1. pip install -r requirements.txt
2. bash train.sh <path-to-train-data>

How to reproduce test result?
bash test.sh <path-to-test-data> <path-to-private-test-data> <path-to-submit-file>

You can reproduce test result without retrain, the code will fetch the best model store by me in dropbox.
If you wanna to test the model you retrain, you only need to rerun the train bash file. Then run bash test.sh <test_file> <submit_file> , it will fetch the best model automatically.