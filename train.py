import os
import wget, tarfile
import torch
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from pytorch_transformers import BertTokenizer
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertModel
from torch.utils.data import DataLoader
from tqdm import trange
import json
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

def _run_epoch(epoch, training,fine_tune):
    model.train(training)
    if training:
        description = 'Train'
        dataset = trainData
        shuffle = True
    else:
        description = 'Valid'
        dataset = validData
        shuffle = False
        
    if fine_tune:
        model.unfreeze()
    else:
        model.freeze()
    dataloader = DataLoader(dataset=dataset,
                            batch_size=12,
                            shuffle=shuffle,
                            collate_fn=dataset.collate_fn,
                            num_workers=4)

    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
    loss = 0
    f1_score = F1()

    
    for i, (x, y) in trange:
        batch_loss,o_labels = _run_iter(x,y)
        if training:
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

        loss += batch_loss.item()
        f1_score.update(o_labels.cpu(), y)


        trange.set_postfix(
            loss=loss / (i + 1), f1=f1_score.print_score())
        
    if training:
        history['train'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})
    else:
        history['valid'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})
        

def _run_iter(x,y):
    abstract = x.to(device)
    labels = y.to(device)
    #l_loss = model(abstract,labels=labels)#modified
    l_loss, o_labels = model(abstract,labels=labels)
    #print(output)
    #print(o_labels)
    #print(logits)

    return l_loss, o_labels

def save(epoch):
    if not os.path.exists('model'):
        osx.to(device).makedirs('model')
    torch.save(model.state_dict(), 'model/model.pkl.'+str(epoch))
    with open('model/history.json', 'w') as f:
        json.dump(history, f, indent=4)

class F1():
    def __init__(self,th=0):
        self.threshold = th
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.name = 'F1'

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0

    def update(self, predicts, groundTruth):
        predicts = predicts > self.threshold
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        self.n_corrects += torch.sum(groundTruth.type(torch.bool) * predicts).data.item()

    def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20)
        return 2 * (recall * precision) / (recall + precision + 1e-20)

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.weight = torch.tensor([0.1,0.1,0.2,0.6])
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()#BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return logits
        
    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True
            
class AbstractDataset(Dataset):
    def __init__(self, data,max_len = 256):
        self.data = data
        self.max_len = max_len
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        
    def collate_fn(self, datas):
        # get max length in this batch
        batch_abstract = []
        batch_label = []
        for data in datas:
            batch_abstract.append(data['Abstract'].ravel())
            # gather labels
            if 'Label' in data:
                batch_label.append(data['Label'])
        #print(batch_label)
            
        return torch.LongTensor(batch_abstract),torch.FloatTensor(batch_label)
        
def label_to_onehot(labels):
    """ Convert label to onehot .
        Args:
            labels (string): sentence's labels.
        Return:
            outputs (onehot list): sentence's onehot label.
    """
    label_dict = {'THEORETICAL': 0, 'ENGINEERING':1, 'EMPIRICAL':2, 'OTHERS':3}
    onehot = [0,0,0,0]
    for l in labels.split():
        onehot[label_dict[l]] = 1
    return onehot
        
def sentence_to_indices(sentence, word_tokenize):
    """ Convert sentence to its word indices.
    Args:
        sentence (str): One string.
    Return:
        indices (list of int): List of word indices.
    """
    tokens = word_tokenize.tokenize(sentence)
    tokens.append('[SEP]')
    indices = word_tokenize.convert_tokens_to_ids(tokens)
    return indices
    
def get_dataset(data_path, word_tokenizer, n_workers=4):
    """ Load data and return dataset for training and validating.

    Args:
        data_path (str): Path to the data.
    """
    dataset = pd.read_csv(data_path, dtype=str)

    results = [None] * n_workers
    with Pool(processes=n_workers) as pool:
        for i in range(n_workers):
            batch_start = (len(dataset) // n_workers) * i
            if i == n_workers - 1:
                batch_end = len(dataset)
            else:
                batch_end = (len(dataset) // n_workers) * (i + 1)
            
            batch = dataset[batch_start: batch_end]
            results[i] = pool.apply_async(preprocess_samples, args=(batch,word_tokenizer))

        pool.close()
        pool.join()

    processed = []
    for result in results:
        processed += result.get()
    return processed

def preprocess_samples(dataset, word_tokenize):
    """ Worker function.

    Args:
        dataset (list of dict)
    Returns:
        list of processed dict.
    """
    processed = []
    for sample in tqdm(dataset.iterrows(), total=len(dataset)):
        processed.append(preprocess_sample(sample[1], word_tokenize))
    
    return processed

def preprocess_sample(data, word_tokenize):
    """
    Args:
        data (dict)
    Returns:
        dict
    """
    processed = {}
    sent_ind = word_tokenize.convert_tokens_to_ids(['[CLS]'])
    for sent in data['Abstract'].split('$$$'):
        sent_ind += sentence_to_indices(sent, word_tokenize)
    sent_ind.pop()
    sent_ind += word_tokenize.convert_tokens_to_ids(['[END]'])
    input_ids = pad_sequences([sent_ind], maxlen=256, dtype="long", truncating="post", padding="post")
    processed['Abstract'] = input_ids
    if 'Task 2' in data:
        processed['Label'] = label_to_onehot(data['Task 2'])

    
    return processed

def get_scibert():
    DATA_URL= 'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar'

    out_fname = 'scibert_scivocab_uncased.tar'

    wget.download(DATA_URL, out=out_fname)
    # 提取压缩包
    tar = tarfile.open(out_fname)
    tar.extractall()
    tar.close

    os.remove(out_fname)
    
if __name__ == "__main__":
    train_set_csv = sys.argv[1]
    torch.manual_seed(42)    
    
    if not os.path.exists('scibert_scivocab_uncased'):
        get_scibert()
    
    # Preprocess Train data
    dataset = pd.read_csv(train_set_csv, dtype=str)
    dataset.drop('Title',axis=1,inplace=True)
    dataset.drop('Categories',axis=1,inplace=True)
    dataset.drop('Created Date',axis=1, inplace=True)
    dataset.drop('Authors',axis=1,inplace=True)    
    
    trainset, validset = train_test_split(dataset, test_size=0.1, random_state=42)
    trainset.to_csv('trainset.csv', index=False)
    validset.to_csv('validset.csv', index=False)
    
    # Tokenize Data
    word_tokenizer = BertTokenizer.from_pretrained('scibert_scivocab_uncased/vocab.txt', do_lower_case=True)
    print('[INFO] Start processing trainset...')
    train = get_dataset('trainset.csv',word_tokenizer, n_workers=4)
    print('[INFO] Start processing validset...')
    valid = get_dataset('validset.csv',word_tokenizer,  n_workers=4)
    
    trainData = AbstractDataset(train)
    validData = AbstractDataset(valid)
    
    # Train Model    
    lr = 3e-6
    max_grad_norm = 1.0
    num_total_steps = 1000
    num_warmup_steps = 100
    warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1

    model = BertForMultiLabelSequenceClassification.from_pretrained(
            "./scibert_scivocab_uncased/weights.tar.gz",
            num_labels = 4)
    opt = BertAdam(model.parameters(), lr=lr, schedule='warmup_linear', warmup=warmup_proportion, t_total=num_total_steps)
    model.to(device)
    fine_tune_epoch = 3
    max_epoch = 10
    history = {'train':[],'valid':[]}

    for epoch in range(fine_tune_epoch):
        print('Epoch: {}'.format(epoch))
        _run_epoch(epoch, True,fine_tune=True)
        _run_epoch(epoch, False,fine_tune=True)
        save(epoch)        

    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        _run_epoch(epoch, True,fine_tune=False)
        _run_epoch(epoch, False,fine_tune=False)
        save(fine_tune_epoch+epoch)