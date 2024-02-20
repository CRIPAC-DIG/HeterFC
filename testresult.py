
from datautils import wiki_path,dev_retrieved_path,test_retrieved_path,train_retrieved_path,smalltrainpath

from cmath import log
from curses import window
import os
import torch
import torch.nn as nn
import torch.optim as optim
import jsonlines
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from torch_geometric.loader import DataLoader
# from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import get_scheduler, AdamW,get_linear_schedule_with_warmup
from preprocess import preprocess_wt, prepare_graph_data_word_level_attention
from model import T2GV2_noEnt
from feverous_scorer import feverous_score
import torch.nn.functional as F
import random

roberta_pretrained_path = 'roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
train_model_name = 'example.pt'
#### hyperparameters ####
weight_kl = 1.2
windowsize = 2
layers = 2
max_seq_length = 128
model_specific = T2GV2_noEnt
###########################
num_epochs = 3
train_batch_size = 4 ################ old 16
dev_batch_size = 4
accumulation_steps = 2
plm_emd_dim = 1024
evaluation_steps = 800
logging_steps = 10
feverous_score_first = True
#########################
use_gold = False
nei_aug = False
#########################
device = torch.device("cuda:0")
tokenizer = None
if tokenizer is None:
    tokenizer = RobertaTokenizer.from_pretrained(roberta_pretrained_path)  
tokenized_concat_claim_evi_list,tokenized_claim_evi_list, edge_index_list, edge_type_list, \
label_list,num_claim_evi_list,valididx_list,evi_word_cnt,gold_indi = \
preprocess_wt(
test_retrieved_path, tokenizer,maxlength=max_seq_length,windowsize=windowsize)
dev_data = prepare_graph_data_word_level_attention(tokenized_concat_claim_evi_list,
tokenized_claim_evi_list, edge_index_list, edge_type_list,num_claim_evi_list, 
label_list,valididx_list,evi_word_cnt,gold_indi,test=True)
test_dataloader = DataLoader(dev_data, batch_size=dev_batch_size, shuffle=False)
plm_model = RobertaForSequenceClassification.from_pretrained(roberta_pretrained_path, num_labels=3, return_dict=True)

# initialize the model
model = model_specific(plm_model, plm_emd_dim, 512, 256, 3, 128, 3,layers=layers).to(device)
model.load_state_dict(torch.load(train_model_name, map_location=device))
model.eval()
map_verdict_to_index = {0:'NOT ENOUGH INFO', 1:'SUPPORTS', 2:'REFUTES'}
predicted_result = []

for batch in tqdm(test_dataloader):
    batch = batch.to(device)
    # print(batch)
    prob,_ = model(batch.edge_index, batch.edge_type, batch.x, batch.num_claim_evi,batch.attn_mask, batch.valid_idx,batch.evi_word_cnt,batch.concat_token_ids,batch.concat_attn_mask)
    predicted_result += list(prob.argmax(axis=-1).cpu().numpy())

print(len(predicted_result),'has been predicted')
predicted_result = [map_verdict_to_index[i] for i in predicted_result]

with jsonlines.open('tosubmit.csv', 'w') as writer:
    with jsonlines.open(test_retrieved_path) as f:
            for i,line in enumerate(f.iter()):
                if i == 0:
                    writer.write({'header':''}) # skip header line
                    continue
                # if len(line['evidence'][0]['content']) == 0: continue
                line['predicted_label'] = predicted_result[i-1]
                writer.write(line)
print('Predicted {} results, {} written'.format(len(predicted_result),i))