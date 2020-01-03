# AICUP-2019-Abstract_Classification

## Competition Description
 從arXiv的電腦科學相關論文摘要，預測出摘要所屬的類別(Theoretical Paper, Engineering Paper, Empirical Paper, Others)。
 需注意的是摘要可以有多個分  類，例如: 摘要可以同時是Theoretical Paper和Engineering Paper。

## Data and File Decription
  Training data: 
  - data/task2_trainset.csv
  
  Testing data: 
  - data/task2_public_testset.csv 
  - data/task2_private_testset.csv
  
  Train File
  - train.py
  
  Test File
  - test.py

## How to reproduce train result?
1. pip3 install -r requirements.txt
2. bash train.sh \<path-to-train-data>
3. bash test.sh \<path-to-test-data> \<path-to-private-test-data> \<path-to-submit-file>
