import json, glob, os, random
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import re, emoji


logger = logging.getLogger(__name__)
model_dict = { 'indobertweet': 'indolem/indobertweet-base-uncased',
               'indobert': 'indolem/indobert-base-uncased'}


def find_url(string):
    # with valid conditions for urls in string 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)
    return [x[0] for x in url]

def preprocess_tweet(tweet):
    tweet = emoji.demojize(tweet).lower()
    new_tweet = []
    for word in tweet.split():
        if word[0] == '@' or word == '[username]':
            new_tweet.append('@USER')
        elif find_url(word) != []:
            new_tweet.append('HTTPURL')
        elif word == 'httpurl' or word == '[url]':
            new_tweet.append('HTTPURL')
        else:
            new_tweet.append(word)
    return ' '.join(new_tweet)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class BertData():
    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained(model_dict[args.bert_model], do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.MAX_TOKEN = args.max_token

    def preprocess_one(self, src_txt, label):
        src_txt = preprocess_tweet(src_txt)
        src_subtokens = [self.cls_token] + self.tokenizer.tokenize(src_txt) + [self.sep_token]        
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        
        if len(src_subtoken_idxs) > self.MAX_TOKEN:
            src_subtoken_idxs = src_subtoken_idxs[:self.MAX_TOKEN]
            src_subtoken_idxs[-1] = self.sep_vid
        else:
            src_subtoken_idxs += [self.pad_vid] * (self.MAX_TOKEN-len(src_subtoken_idxs))
        segments_ids = [0] * len(src_subtoken_idxs)
        assert len(src_subtoken_idxs) == len(segments_ids)
        return src_subtoken_idxs, segments_ids, label
    
    def preprocess(self, src_txts, labels):
        assert len(src_txts) == len(labels)
        output = []
        for idx in range(len(src_txts)):
            output.append(self.preprocess_one(src_txts[idx], labels[idx]))
        return output


class Batch():
    def __init__(self, data, idx, batch_size, device):
        cur_batch = data[idx:idx+batch_size]
        src = torch.tensor([x[0] for x in cur_batch])
        seg = torch.tensor([x[1] for x in cur_batch])
        label = torch.tensor([x[2] for x in cur_batch])
        mask_src = 0 + (src != 0)
        
        self.src = src.to(device)
        self.seg= seg.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)

    def get(self):
        return self.src, self.seg, self.label, self.mask_src


class Model(nn.Module):
    def __init__(self, args, device):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_dict[args.bert_model], do_lower_case=True)
        self.linear = nn.Linear(self.bert.config.hidden_size, args.vocab_label_size)
        self.dropout = nn.Dropout(0.2)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=args.vocab_label_size, reduction='sum')

    def forward(self, src, seg, mask_src):
        top_vec, _ = self.bert(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        top_vec = self.dropout(top_vec)
        top_vec *= mask_src.unsqueeze(dim=-1).float()
        top_vec = torch.sum(top_vec, dim=1) / mask_src.sum(dim=-1).float().unsqueeze(-1)
        conclusion = self.linear(top_vec).squeeze()
        return conclusion
    
    def get_loss(self, src, seg, label, mask_src):
        output = self.forward(src, seg, mask_src)
        return self.loss(output.view(-1,self.args.vocab_label_size), label.view(-1))

    def predict(self, src, seg, mask_src):
        output = self.forward(src, seg, mask_src)
        batch_size = output.shape[0]
        prediction = torch.argmax(output, dim=-1).data.cpu().numpy().tolist()
        return prediction


def prediction(dataset, model, args):
    preds = []
    golds = []
    model.eval()
    for j in range(0, len(dataset), args.batch_size):
        src, seg, label, mask_src = Batch(dataset, j, args.batch_size, args.device).get()
        preds += model.predict(src, seg, mask_src)
        golds += label.cpu().data.numpy().tolist()
    return f1_score(golds, preds, average='macro'), preds

def create_vocab(labels):
    unique = np.unique(labels)
    label2id = {}
    id2label = {}
    counter = 0
    for word in unique:
        label2id[word] = counter
        id2label[counter] = word
        counter += 1
    return label2id, id2label

def convert_label2id(label2id, labels):
    return [label2id[x] for x in labels]

def save_df(pred, id2label):
    ids = np.arange(len(pred))
    pred = [id2label[p] for p in pred]
    df = pd.DataFrame()
    df['index']=ids
    df['label']=pred
    df.to_csv('pred_bertW.csv', index=False)

def train(args, train_dataset, dev_dataset, test_dataset, model, id2label):
    """ Train the model """
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    t_total = len(train_dataset) // args.batch_size * args.num_train_epochs
    args.warmup_steps = int(0.1 * t_total)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warming up = %d", args.warmup_steps)
    logger.info("  Patience  = %d", args.patience)

    # Added here for reproductibility
    set_seed(args)
    tr_loss = 0.0
    global_step = 1
    best_f1_dev = 0
    cur_patience = 0
    for i in range(int(args.num_train_epochs)):
        random.shuffle(train_dataset)
        epoch_loss = 0.0
        for j in range(0, len(train_dataset), args.batch_size):
            src, seg, label, mask_src = Batch(train_dataset, j, args.batch_size, args.device).get()
            model.train()
            loss = model.get_loss(src, seg, label, mask_src)
            loss = loss.sum()/args.batch_size
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            loss.backward()

            tr_loss += loss.item()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
        logger.info("Finish epoch = %s, loss_epoch = %s", i+1, epoch_loss/global_step)
        dev_f1, _ = prediction(dev_dataset, model, args)
        if dev_f1 > best_f1_dev:
            best_f1_dev = dev_f1
            _, test_pred = prediction(test_dataset, model, args)
            save_df(test_pred, id2label)
            #SAVE
            cur_patience = 0
            logger.info("Better, BEST F1 in DEV = %s, SAVE TEST!", best_f1_dev)
        else:
            cur_patience += 1
            if cur_patience == args.patience:
                logger.info("Early Stopping Not Better, BEST F1 in DEV = %s", best_f1_dev)
                break
            else:
                logger.info("Not Better, BEST F1 in DEV = %s", best_f1_dev)

    return global_step, tr_loss / global_step, best_f1_dev


args_parser = argparse.ArgumentParser()
args_parser.add_argument('--bert_model', default='indobertweet', choices=['indobert', 'indobertweet'], help='select one of models')
args_parser.add_argument('--data_path', default='data/', help='path to all train/test/dev')
args_parser.add_argument('--max_token', type=int, default=128, help='maximum token allowed for 1 instance')
args_parser.add_argument('--batch_size', type=int, default=30, help='batch size')
args_parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
args_parser.add_argument('--weight_decay', type=int, default=0, help='weight decay')
args_parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
args_parser.add_argument('--max_grad_norm', type=float, default=1.0)
args_parser.add_argument('--num_train_epochs', type=int, default=20, help='total epoch')
args_parser.add_argument('--warmup_steps', type=int, default=242, help='warmup_steps, the default value is 10% of total steps')
args_parser.add_argument('--logging_steps', type=int, default=200, help='report stats every certain steps')
args_parser.add_argument('--seed', type=int, default=2021)
args_parser.add_argument('--local_rank', type=int, default=-1)
args_parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
args_parser.add_argument('--no_cuda', default=False)
args = args_parser.parse_args()


# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
args.device = device

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
)

set_seed(args)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

bertdata = BertData(args)

trainset = pd.read_csv(args.data_path+'train_preprocess.tsv', sep='\t')
devset = pd.read_csv(args.data_path+'valid_preprocess.tsv', sep='\t')
testset = pd.read_csv(args.data_path+'test_preprocess_masked_label.tsv', sep='\t')
xtrain, ytrain = list(trainset['text']), list(trainset['label'])
xdev, ydev = list(devset['text']), list(devset['label'])
xtest, ytest = list(testset['text']), list(testset['label'])

label2id, id2label = create_vocab (ytrain)
ytrain =  convert_label2id (label2id, ytrain)
ydev =  convert_label2id (label2id, ydev)
ytest =  convert_label2id (label2id, ytest)
args.vocab_label_size = len(label2id)

model = Model(args, device)
model.to(args.device)
train_dataset = bertdata.preprocess(xtrain, ytrain)
dev_dataset = bertdata.preprocess(xdev, ydev)
test_dataset = bertdata.preprocess(xtest, ytest)

global_step, tr_loss, best_f1_dev= train(args, train_dataset, dev_dataset, test_dataset, model, id2label)
print('Dev set F1', best_f1_dev)
