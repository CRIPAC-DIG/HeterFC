from datautils import wiki_path,dev_retrieved_path,test_retrieved_path,train_retrieved_path#,smalltrainpath

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

def setup_seed(seed=0):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        #os.environ['PYTHONHASHSEED'] = str(seed)
def get_optimizer(model, lr, weight_decay = 1e-4, fix_bert = False):
    bert_params, task_params = [], []
    size = 0
    for name, params in model.named_parameters():
        if "roberta" in name:
            bert_params.append((name, params))
        else:
            task_params.append((name, params))
        size += params.nelement()

    # print("bert parameters")
    # for name, params in bert_params:
    #     print('n: {}, shape: {}'.format(name, params.shape))
    # print('*' * 150)
    print("task parameters")
    for name, params in task_params:
        print('n: {}, shape: {}'.format(name, params.shape))
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    nobert_params = task_params

    if fix_bert:
        params = [
            {"params": nobert_params, "weight_decay": weight_decay,"lr": 100 * lr},
        ]
    else:
        params = [
            {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": lr},
            {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": lr},
            {
                "params": [p for n, p in nobert_params],
                "weight_decay": weight_decay,
                "lr": 100*lr
                # "lr": lr
            },
        ]

    optimizer = AdamW(params, correct_bias=False)
    return optimizer
# setup_seed(31)
# kl = F.kl_div


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
evaluation_steps = 1600
logging_steps = 10
feverous_score_first = True
#########################
use_gold = False
nei_aug = False
#########################

device = torch.device("cuda:0")
if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
if not os.path.exists('preprocessed'):
    os.makedirs('preprocessed')

tokenizer = None

if tokenizer is None:
    tokenizer = RobertaTokenizer.from_pretrained(roberta_pretrained_path)  
tokenized_concat_claim_evi_list,tokenized_claim_evi_list, edge_index_list, edge_type_list, \
label_list,num_claim_evi_list,valididx_list,evi_word_cnt,gold_indi = \
preprocess_wt(
dev_retrieved_path, tokenizer,maxlength=max_seq_length,windowsize=windowsize)
dev_data = prepare_graph_data_word_level_attention(tokenized_concat_claim_evi_list,
tokenized_claim_evi_list, edge_index_list, edge_type_list,num_claim_evi_list, 
label_list,valididx_list,evi_word_cnt,gold_indi)
dev_dataloader = DataLoader(dev_data, batch_size=dev_batch_size, shuffle=False)

if tokenizer is None:
    tokenizer = RobertaTokenizer.from_pretrained(roberta_pretrained_path) 
tokenized_concat_claim_evi_list,tokenized_claim_evi_list, edge_index_list, edge_type_list, \
label_list,num_claim_evi_list,valididx_list,evi_word_cnt,gold_indi = \
preprocess_wt(
train_retrieved_path,tokenizer,maxlength=max_seq_length,windowsize=windowsize)
train_data = prepare_graph_data_word_level_attention(tokenized_concat_claim_evi_list,
tokenized_claim_evi_list, edge_index_list, edge_type_list,
num_claim_evi_list, label_list,valididx_list,evi_word_cnt,gold_indi)
train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)


plm_model = RobertaForSequenceClassification.from_pretrained(roberta_pretrained_path, num_labels=3, return_dict=True)

# initialize the model
model = model_specific(plm_model, plm_emd_dim, 512, 256, 3, 128, 3,layers=layers).to(device)
# optimizer = optim.Adam(model.parameters(),
#                        lr=1e-5,
#                        weight_decay=0)
optimizer = get_optimizer(model,lr=1e-5,weight_decay=1e-4,fix_bert=False)
num_training_steps = num_epochs * len(train_dataloader) // accumulation_steps
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=0.2*num_training_steps, num_training_steps=num_training_steps
)

def evaluate(model, dataloader, report_fine_grained_metric=True):
    print('--------- dev Evaluation Start ---------')
    map_verdict_to_index = {0:'NOT ENOUGH INFO', 1:'SUPPORTS', 2:'REFUTES'}
    pred_label, pred_label_and_evi = [], []
    model.eval()
    for batch in dataloader:
        batch = batch.to(device)
        try:
            prob,_ = model(batch.edge_index, batch.edge_type, batch.x, batch.num_claim_evi,batch.attn_mask, batch.valid_idx,batch.evi_word_cnt,batch.concat_token_ids,batch.concat_attn_mask)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            prob,_ = model(batch.edge_index, batch.edge_type, batch.x, batch.num_claim_evi,batch.attn_mask, batch.valid_idx,batch.evi_word_cnt,batch.concat_token_ids,batch.concat_attn_mask)
        pred_label += list(prob.argmax(axis=-1).cpu().numpy())
    pred_label = [map_verdict_to_index[i] for i in pred_label]
    with jsonlines.open(dev_retrieved_path) as f:
        for i, line in enumerate(f.iter()):
            if i == 0:
                continue
            line['predicted_label'] = pred_label[i-1]
            line['evidence'] = [el['content'] for el in line['evidence']]
            for j in range(len(line['evidence'])):
                line['evidence'][j]= [[el.split('_')[0], el.split('_')[1] if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[1:3]), '_'.join(el.split('_')[2:]) if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[3:])] for el in  line['evidence'][j]]
            line['predicted_evidence'] = [[el.split('_')[0], el.split('_')[1] if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[1:3]), '_'.join(el.split('_')[2:]) if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[3:])] for el in line['predicted_evidence']]
            pred_label_and_evi.append(line)
        print('{} predicted results'.format(len(pred_label)))
        print('computing the scores...')
        strict_score, label_accuracy, precision, recall, f1 = feverous_score(pred_label_and_evi)
        if report_fine_grained_metric:
            sup_correct, ref_correct, nei_correct = 0, 0, 0
            sup_cnt, ref_cnt, nei_cnt = 0, 0, 0
            for instance in pred_label_and_evi:
                pred_label = instance["predicted_label"].upper()
                gd_label = instance["label"].upper()
                if pred_label == gd_label:
                    if gd_label == 'SUPPORTS':
                        sup_correct += 1
                    elif gd_label == 'REFUTES':
                        ref_correct += 1
                    elif gd_label == 'NOT ENOUGH INFO':
                        nei_correct += 1
                if gd_label == 'SUPPORTS':
                    sup_cnt += 1
                elif gd_label == 'REFUTES':
                    ref_cnt += 1
                elif gd_label == 'NOT ENOUGH INFO':
                    nei_cnt += 1
            print('Evaluation | SUP ACC: {:.4f} \t REF ACC: {:.4f} \t NEI ACC: {:.4f}'.format(sup_correct/sup_cnt, ref_correct/ref_cnt, nei_correct/nei_cnt))
        return strict_score, label_accuracy

# TODO: add argument information to decide whether train now
# loss_fn = nn.NLLLoss(reduction='mean')
loss_fn = nn.NLLLoss()
loss_ad = nn.BCELoss()
print('------ Start Training! ------')
training_args = PrettyTable()
training_args.field_names = ['Total training steps', 'Batch size', 'Epochs', 'Accumulation steps']
training_args.add_row([num_training_steps, train_batch_size, num_epochs, accumulation_steps])
print(training_args)

early_stop = False
global_steps = 0
logging_loss = 0.
best_dev_feverous_score, best_dev_acc = 0., 0.
early_stop_cnt_la, early_stop_cnt_fs = 0, 0

for e in range(num_epochs):
    print(len(train_dataloader))
    for num, batch in enumerate(train_dataloader):
        model.train()
        batch = batch.to(device)
        try:
            prob,attention_list = model(batch.edge_index, batch.edge_type, batch.x, batch.num_claim_evi, batch.attn_mask, batch.valid_idx,batch.evi_word_cnt,batch.concat_token_ids,batch.concat_attn_mask)
            loss = loss_fn(prob, batch.y)
            
            aided_loss = 0
            for at,gd in zip(attention_list,batch.gold_indicator):
                # print('DEBUG',at,gd)
                aided_loss += loss_ad(at,torch.tensor(gd).float().to(device))
            aided_loss/=len(attention_list)
            if num<5:
                print('DEBUG info:',num,at,gd,aided_loss,loss)
            loss += weight_kl*aided_loss

            loss = loss / accumulation_steps
            loss.backward()
            logging_loss += loss.item()
        except:
            torch.cuda.empty_cache()
            try:
                prob,attention_list = model(batch.edge_index, batch.edge_type, batch.x, batch.num_claim_evi, batch.attn_mask, batch.valid_idx,batch.evi_word_cnt,batch.concat_token_ids,batch.concat_attn_mask)
                loss = loss_fn(prob, batch.y)

                aided_loss = 0
                for at,gd in zip(attention_list,batch.gold_indicator):
                    aided_loss += loss_ad(at,torch.tensor(gd).float().to(device))
                aided_loss/=len(attention_list)
                loss += weight_kl*aided_loss

                loss = loss / accumulation_steps
                loss.backward()
                logging_loss += loss.item() 
            except:
                print('***escape one batch***',batch.evi_word_cnt)

        if (num+1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_steps += 1
            if global_steps % evaluation_steps == 0:
                strict_score, label_accuracy = evaluate(model, dev_dataloader, report_fine_grained_metric=True)
                print('Evaluation | Global Step: {:d} \t FEVEROUS score: {:.4f} \t Label accuracy: {:.4f}'.format(global_steps, strict_score, label_accuracy))
                if strict_score > best_dev_feverous_score:
                    best_dev_feverous_score = strict_score
                    early_stop_cnt_fs = 0
                    if feverous_score_first:
                        print('Save the best model so far...')
                        torch.save(model.state_dict(), train_model_name)
                else:
                    early_stop_cnt_fs += 1
                if label_accuracy > best_dev_acc:
                    best_dev_acc = label_accuracy
                    early_stop_cnt_la = 0
                    if not feverous_score_first:
                        print('Save the best model so far...')
                        torch.save(model.state_dict(), train_model_name)
                else:
                    early_stop_cnt_la += 1
                if feverous_score_first and early_stop_cnt_fs == 222212:
                    print('Current step: {:d} | Current Epoch: {:d}'.format(global_steps, e+1))
                    print('FEVEROUS score is not increased for 5 evaluation steps, force early stopping!')
                    early_stop = True
                    break
                if not feverous_score_first and early_stop_cnt_la == 222212:
                    print('Current step: {:d} | Current Epoch: {:d}'.format(global_steps, e+1))
                    print('FEVEROUS score is not increased for 5 evaluation steps, force early stopping!')
                    early_stop = True
                    break

            if global_steps % logging_steps == 0:
                print('Epoch {:d} | Global step: {:d} \t Loss: {:.4f}'.format(e+1, global_steps, logging_loss / logging_steps))
                logging_loss = 0.
    if early_stop:
        break

        

