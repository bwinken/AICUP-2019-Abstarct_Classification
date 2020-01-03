import wget, tarfile
import pandas as pd
from pytorch_transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertModel,BertConfig
import os
from multiprocessing import Pool
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

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

def SubmitGenerator(prediction, sampleFile, public=True, filename='prediction.csv'):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
        public (boolean)
        filename (str)
    """
    sample = pd.read_csv(sampleFile)
    submit = {}
    submit['order_id'] = list(sample.order_id.values)
    redundant = len(sample) - prediction.shape[0]
    if public:
        submit['THEORETICAL'] = list(prediction[:,0]) + [0]*redundant
        submit['ENGINEERING'] = list(prediction[:,1]) + [0]*redundant
        submit['EMPIRICAL'] = list(prediction[:,2]) + [0]*redundant
        submit['OTHERS'] = list(prediction[:,3]) + [0]*redundant
    else:
        submit['THEORETICAL'] = [0]*redundant + list(prediction[:,0])
        submit['ENGINEERING'] = [0]*redundant + list(prediction[:,1])
        submit['EMPIRICAL'] = [0]*redundant + list(prediction[:,2])
        submit['OTHERS'] = [0]*redundant + list(prediction[:,3])
    df = pd.DataFrame.from_dict(submit) 
    df.to_csv(filename,index=False)

def get_model():
    DATA_URL= 'https://www.dropbox.com/s/cjiqol20mu6jcr8/final_best.pkl.12?dl=1'

    out_fname = 'best_model.pkl'

    wget.download(DATA_URL, out=out_fname)

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
    test_set_csv = sys.argv[1]
    private_set_csv = sys.argv[2]
    submit_csv = sys.argv[3]

    torch.manual_seed(42)
    retrain = True
    if not os.path.exists('scibert_scivocab_uncased'):
        print("Getting SciBERT...")
        get_scibert()

    if not os.path.exists('model/model.pkl.12'):
        print("Getting Model...")
        get_model()
        retrain = False

    #Preprocess Test data
    dataset = pd.read_csv(test_set_csv, dtype=str)
    dataset.drop('Title',axis=1,inplace=True)
    dataset.drop('Categories',axis=1,inplace=True)
    dataset.drop('Created Date',axis=1, inplace=True)
    dataset.drop('Authors',axis=1,inplace=True)

    private = pd.read_csv(private_set_csv, dtype=str)
    private.drop('Title',axis=1,inplace=True)
    private.drop('Categories',axis=1,inplace=True)
    private.drop('Created Date',axis=1, inplace=True)
    private.drop('Authors',axis=1,inplace=True)

    combined = pd.concat([dataset,private])
    combined.to_csv('testset.csv',index=False)

    word_tokenizer = BertTokenizer.from_pretrained('scibert_scivocab_uncased/vocab.txt', do_lower_case=True)

    print('[INFO] Start processing testset...')
    test = get_dataset('testset.csv',word_tokenizer,  n_workers=4)
    testData = AbstractDataset(test)

    model = BertForMultiLabelSequenceClassification.from_pretrained(
        "./scibert_scivocab_uncased/weights.tar.gz",
        num_labels = 4)

    if retrain:
        model.load_state_dict(torch.load('model/model.pkl.{}'.format(12))) 
    else:
        model.load_state_dict(torch.load('best_model.pkl'))
    model.train(False)
    model.to(device)
    #_run_epoch(1, False)
    dataloader = DataLoader(dataset=testData,
                                batch_size=16,
                                shuffle=False,
                                collate_fn=testData.collate_fn,
                                num_workers=4)
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    prediction = []
    for i, (x,y) in trange:
        o_labels = model(x.to(device),token_type_ids=None, attention_mask=None, labels=None)
        o_labels = o_labels>-0.8
        prediction.append(o_labels.to('cpu'))

    prediction = torch.cat(prediction).detach().numpy().astype(int)

    SubmitGenerator(prediction, 
                    'score/task2_submission.csv',
                    True, 
                    submit_csv)