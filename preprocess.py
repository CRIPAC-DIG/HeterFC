from annotation_processor import AnnotationProcessor
from datautils import *
from tqdm.contrib import tzip
import re
import json
import torch
import random
import unicodedata
import numpy as np
import string
from tqdm import tqdm as tqddm
def tqdm(x):
    return tqddm(x,mininterval=12)
from torch_geometric.data import Data
import os.path as osp
import pickle
from torch.utils.data import Dataset

class myset(Dataset):
    def __init__(self,datalist):
        self.x = datalist
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return len(self.x)
    



NoneSenseWord = set([
'In','in','At','at','A','a','An','an','And','and','To',
'to','Was','was','Is','is','Were','were','Are','are',
'As','as','The','the','This','this','That','that','am',
'By','by','For','for','via','which','Which','what','What',
'How','how','Of','of','With','with','contains'
 ])

def preprocess_wt(retrieved_evidence_path, tokenizer, maxlength, windowsize=2,fully_connected=False,old=False):
    prefixoffile=''
    if old:
        prefixoffile = 'old_retrieval_'
    if 'train' in retrieved_evidence_path:
        if osp.exists(prefixoffile+'train_first_step_file.pkl'):
            print('exists')
            with open(prefixoffile+'train_first_step_file.pkl','rb') as toread:
                temp = pickle.load(toread)
            claim_evi_list = temp['claim_evi_list']
            num_claim_evi_list = temp['num_claim_evi_list']
            label_list = temp['label_list']
            valididx_list = temp['valididx_list']
            gold_indicator = temp['gold_indicator']
            evi_word_cnt = temp['evi_word_cnt']
            tags_list = temp['tags_list']
            concat_claim_evi_list = temp['concat_claim_evi_list']
            print('loading done')
            del temp
        else:
            trainannotationlist = AnnotationProcessor(retrieved_evidence_path)
            raw = []
            for anno in tqdm(trainannotationlist):
                raw.append((prepare_input(anno,gold=True,indicator=True),anno.verdict))
            
            claim_list = [r[0][0][0] for r in raw]
            evidence_list = [r[0][0][1:] if len(r[0][0])>1 else ['No bias.'] for r in raw]
            label_list = [r[1] for r in raw]
            evidence_type_list = [r[0][1][1:] if len(r[0][1])>1 else ['sentence'] for r in raw]
            gold_indicator = [r[0][2] if len(r[0][0])>1 else [0] for r in raw]

            trainannotationlist = AnnotationProcessor(retrieved_evidence_path)
            raw = []
            for anno in tqdm(trainannotationlist):
                raw.append((prepare_input(anno,gold=False,indicator=True),anno.verdict))
            claim_list.extend([r[0][0][0] for r in raw])
            evidence_list.extend([r[0][0][1:] if len(r[0][0])>1 else ['No bias.'] for r in raw])
            label_list.extend([r[1] for r in raw])
            evidence_type_list.extend([r[0][1][1:] if len(r[0][1])>1 else ['sentence'] for r in raw])
            gold_indicator.extend([r[0][2] if len(r[0][0])>1 else [0] for r in raw])
            concat_claim_evi_list,claim_evi_list,num_claim_evi_list,label_list,valididx_list,evi_word_cnt,tags_list = prepare_plm_input_word_level_attention(claim_list, evidence_list, label_list,evidence_type_list,tokenizer,maxlength,plm='roberta', test_split=False,windowsize=windowsize)
        
            ################
            # from tagme_ent import multi_thread_ents_aggreagate
            # valididx_list,evi_word_cnt,tags_list = multi_thread_ents_aggreagate(valididx_list,evi_word_cnt,tags_list)
            ################1
            with open(prefixoffile+'train_first_step_file.pkl','wb') as to_save:
                pickle.dump({'claim_evi_list':claim_evi_list,'num_claim_evi_list':num_claim_evi_list,
                'label_list':label_list,'valididx_list':valididx_list,'tags_list':tags_list,
                'evi_word_cnt':evi_word_cnt,'gold_indicator':gold_indicator,'concat_claim_evi_list':concat_claim_evi_list},to_save)

        edge_index_list,edge_type_list = construct_het_graph_word_level_attention(tags_list,windowsize,fully_connected=fully_connected)

        return concat_claim_evi_list,claim_evi_list, edge_index_list, edge_type_list, np.array(label_list),num_claim_evi_list,valididx_list,evi_word_cnt,gold_indicator
    
    elif 'dev' in retrieved_evidence_path:
        if osp.exists(prefixoffile+'dev_first_step_file.pkl'):
            print('exists')
            with open(prefixoffile+'dev_first_step_file.pkl','rb') as toread:
                temp = pickle.load(toread)
            claim_evi_list = temp['claim_evi_list']
            num_claim_evi_list = temp['num_claim_evi_list']
            label_list = temp['label_list']
            valididx_list = temp['valididx_list']
            gold_indicator = temp['gold_indicator']
            evi_word_cnt = temp['evi_word_cnt']
            tags_list = temp['tags_list']
            concat_claim_evi_list = temp['concat_claim_evi_list']
            print('loading done')
            del temp
        else:
            devannotationlist = AnnotationProcessor(retrieved_evidence_path)
            raw = []
            tempcnt = 0
            for anno in tqdm(devannotationlist):
                tempcnt+=1
                if tempcnt>10:
                    pass
                raw.append((prepare_input(anno,gold=False),anno.verdict))
            
            claim_list = [r[0][0][0] for r in raw]
            evidence_list = [r[0][0][1:] if len(r[0][0])>1 else ['No bias.'] for r in raw]
            label_list = [r[1] for r in raw]
            evidence_type_list = [r[0][1][1:] if len(r[0][1])>1 else ['sentence'] for r in raw]
            concat_claim_evi_list,claim_evi_list,num_claim_evi_list,label_list,valididx_list,evi_word_cnt,tags_list = prepare_plm_input_word_level_attention(claim_list, evidence_list, label_list,evidence_type_list,tokenizer,maxlength,plm='roberta', test_split=False,windowsize=windowsize)
            ################
            # from tagme_ent import multi_thread_ents_aggreagate
            # valididx_list,evi_word_cnt,tags_list = multi_thread_ents_aggreagate(valididx_list,evi_word_cnt,tags_list)
            ################1
            with open(prefixoffile+'dev_first_step_file.pkl','wb') as to_save:
                pickle.dump({'claim_evi_list':claim_evi_list,'num_claim_evi_list':num_claim_evi_list,
                'label_list':label_list,'valididx_list':valididx_list,'tags_list':tags_list,
                'evi_word_cnt':evi_word_cnt,'gold_indicator':None,'concat_claim_evi_list':concat_claim_evi_list},to_save)

        edge_index_list,edge_type_list = construct_het_graph_word_level_attention(tags_list,windowsize,fully_connected=fully_connected)
        return concat_claim_evi_list,claim_evi_list, edge_index_list, edge_type_list, np.array(label_list),num_claim_evi_list,valididx_list,evi_word_cnt,None
    
    elif 'test' in retrieved_evidence_path:
        if osp.exists(prefixoffile+'test_first_step_file.pkl'):
            with open(prefixoffile+'test_first_step_file.pkl','rb') as toread:
                temp = pickle.load(toread)
            claim_evi_list = temp['claim_evi_list']
            num_claim_evi_list = temp['num_claim_evi_list']
            label_list = temp['label_list']
            valididx_list = temp['valididx_list']
            gold_indicator = temp['gold_indicator']
            evi_word_cnt = temp['evi_word_cnt']
            tags_list = temp['tags_list']
            concat_claim_evi_list = temp['concat_claim_evi_list']
            print('loading done')
            del temp
        else:
            testannotationlist = AnnotationProcessor(retrieved_evidence_path)
            raw = []
            for anno in tqdm(testannotationlist):
                raw.append((prepare_input(anno,gold=False),None))
            claim_list = [r[0][0][0] for r in raw]
            evidence_list = [r[0][0][1:] if len(r[0][0])>1 else ['No bias.'] for r in raw]
            evidence_type_list = [r[0][1][1:] if len(r[0][1])>1 else ['sentence'] for r in raw]
            label_list = []
       

            concat_claim_evi_list,claim_evi_list,num_claim_evi_list,label_list,valididx_list,evi_word_cnt,tags_list = prepare_plm_input_word_level_attention(claim_list, evidence_list, label_list,evidence_type_list,tokenizer,maxlength,plm='roberta', test_split=True,windowsize=windowsize)
            ################
            # from tagme_ent import multi_thread_ents_aggreagate
            # valididx_list,evi_word_cnt,tags_list = multi_thread_ents_aggreagate(valididx_list,evi_word_cnt,tags_list)
            ################1
            with open(prefixoffile+'test_first_step_file.pkl','wb') as to_save:
                pickle.dump({'claim_evi_list':claim_evi_list,'num_claim_evi_list':num_claim_evi_list,
                'label_list':label_list,'valididx_list':valididx_list,'tags_list':tags_list,
                'evi_word_cnt':evi_word_cnt,'gold_indicator':None,'concat_claim_evi_list':concat_claim_evi_list},to_save)
        
        edge_index_list,edge_type_list = construct_het_graph_word_level_attention(tags_list,windowsize)
        return concat_claim_evi_list,claim_evi_list, edge_index_list, edge_type_list, np.array(label_list),num_claim_evi_list,valididx_list,evi_word_cnt,None




def prepare_plm_input_word_level_attention(claim_list, evidence_list, label_list,evidence_type_list,tokenizer, maxlength,plm='roberta', test_split=False,windowsize=None):
    map_verdict_to_index = {'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2}
    
    valididx_list = []
    all_claim_evi_list = []
    num_claim_evi = []
    evi_word_cnt = []
    tags_list = []
    all_concat_claim_evi_list = []

    if plm == 'bert':
        sep_token = ' [SEP] '
        # cls_token = '[CLS] ' # do not need to manually add cls token since the tokenizer will add automatically
        # end_token = ' [SEP]' # do not need to manually add end token since the tokenizer will add automatically
    elif plm == 'roberta':
        sep_token = '</s></s>'
        # cls_token = '<s> '
        # end_token = ' </s>'
    
    claimcnt = 0
    for claim, evi,evi_type in tzip(claim_list, evidence_list,evidence_type_list):
        # if claimcnt==187:
        #     print("caution")
        claim_ori = claim
        ###############################
        ### to avoid too long claim ###
        temp = tokenizer(claim)['input_ids']
        if len(temp)>= round(maxlength*0.7):
            sum = 0
            for i in tokenizer.convert_ids_to_tokens(temp)[:round(maxlength*0.7)]:
                sum += len(i)
            claim = claim[:sum]
        ###############################

        num_claim_evi.append(len(evi) + 1)
        claim_evis = [claim]
        for e in evi:
            claim_evi = claim + sep_token + e
            claim_evis.append(claim_evi)
        
        claim_evis_toks = tokenizer(
            claim_evis, padding='max_length', 
            truncation=True, max_length=maxlength
        )
        
        
        #######
        # print(claim_evis)
        # for tidx in range(len(claim_evis_toks['input_ids'])):
        #     print(tokenizer.convert_ids_to_tokens(claim_evis_toks['input_ids'][tidx]))
        #######

        valididx,tags,wordcnt = generate_valid_idx_word_level_attention(claim_evis_toks['input_ids'],evi_type,tokenizer)
        tags_list.append(tags)
        all_claim_evi_list.append(claim_evis_toks)
        valididx_list.append(valididx)
        evi_word_cnt.append(wordcnt)
        all_concat_claim_evi_list.append( 
            tokenizer(
                claim_ori +' </s> '+' </s> '.join(evi), padding='max_length', 
            truncation=True, max_length=512
            )
        )
        # print(all_concat_claim_evi_list[0].keys())
        claimcnt += 1

    if test_split:
        labels = label_list
    else:
        labels = [map_verdict_to_index[x] for x in label_list] 

    return all_concat_claim_evi_list,all_claim_evi_list, num_claim_evi,labels,valididx_list,evi_word_cnt,tags_list

def generate_valid_idx_word_level_attention(ids,evi_type,tokenizer):
    # （tags 顺序 种类 原始词） （有效 valid 合并原编号） 
    tags = [] # [(ordercnt_of_evi/claim  evitype(table/sentence/claim)  word),...]
    valid_idx = []
    wordcnt = []
    
    cnt_type = 0
    cnt_id = 0

    catch = [] # [(fire,20),(man,21),...]

    def if_empty_jump(word):
        if word=='<s>' or word=='</s>' or word=='<pad>' or word=='Ġ' :#or (word[0] in string.punctuation):
            return True
        else:
            return False
    
    def if_empty(word):
        if word.startswith('Ġ'):
            return True
        else:
            return False

    def empty_catch(typecnt,evitype):
        if len(catch) != 0:
            word = ''
            cnt_id_list = []
            for t in catch:
                word += t[0]
                cnt_id_list.append(t[1])
            catch.clear()

            valid_idx.append(tuple(cnt_id_list))
            tags.append((typecnt,evitype,word))
            # if word not in ['is','contains']:
            #     valid_idx.append(tuple(cnt_id_list))
            #     tags.append((typecnt,evitype,word))
    
    def push_catch(token,cnt_id_now):
        # if token[0] == 'Ġ':
        #     token = token[1:]
        # if token[0] in  string.punctuation:
        #     token = token[1:]

        # To cope with special word like 'Ġ(' , this word should not be in catch.
        while ((token[0]=='Ġ') or (token[0] in string.punctuation)):
            # print('del:',token[0])
            token = token[1:]
            if len(token)==0:
                break
                

        if len(token)>0:
            catch.append((token,cnt_id_now))

    for i in range(len(ids)):
        if i ==0 : # this is claim only
            nowids = ids[i]
            nowtokens = tokenizer.convert_ids_to_tokens(nowids)
            for token in nowtokens:
                # we do not use claim words here
                # if if_empty_jump(token):
                #     empty_catch(cnt_type,'claim')
                # else:
                #     if if_empty(token):
                #         empty_catch(cnt_type,'claim')
                #     push_catch(token,cnt_id)
                cnt_id += 1
        else: # this contains claim and one evidence pair
            cnt2 = 0
            nowids = ids[i]
            nowtokens = tokenizer.convert_ids_to_tokens(nowids)

            for token in nowtokens:
                if token == '</s>':
                    cnt2 += 1
                if cnt2 >= 2:
                    if if_empty_jump(token):
                        empty_catch(cnt_type,evi_type[i-1])
                    else:
                        if if_empty(token):
                            empty_catch(cnt_type,evi_type[i-1])
                        push_catch(token,cnt_id)
                cnt_id += 1
        
        cnt_type += 1

    wordcnt = [0]*(len(ids)-1)
    for i in tags:
        wordcnt[i[0]-1] += 1

    for i in wordcnt:
        if i==0:
            print("Error!! in preprocess [wordcnt]")

    return valid_idx,tags,wordcnt


def prepare_graph_data_word_level_attention(concat_encodings,encodings, edge_index_list, edge_type_list,num_claim_evi_list, labels,valididx_list,evi_word_cnt,gold_indicator,test=False):
    data_list = []
    for idx in tqdm(range(len(encodings))):
        valid_idx = valididx_list[idx]
        token_ids = torch.tensor(encodings[idx]['input_ids'])
        attn_mask = torch.tensor(encodings[idx]['attention_mask'])
        concat_token_ids = torch.tensor(concat_encodings[idx]['input_ids']).unsqueeze(0)
        concat_attn_mask = torch.tensor(concat_encodings[idx]['attention_mask']).unsqueeze(0)
        label = torch.tensor(labels[idx]) if not test else 0
        edge_index = torch.tensor(edge_index_list[idx], dtype=torch.long)
        edge_type = torch.tensor(edge_type_list[idx])

        data = Data(x=token_ids, edge_index=edge_index, y=label)
        data.attn_mask = attn_mask
        data.edge_type = edge_type
        data.valid_idx = valid_idx
        data.num_claim_evi = num_claim_evi_list[idx]
        data.evi_word_cnt = evi_word_cnt[idx]
        data.concat_token_ids = concat_token_ids
        data.concat_attn_mask = concat_attn_mask
        if gold_indicator is not None:
            data.gold_indicator = gold_indicator[idx]

        data_list.append(data)
    return data_list



def construct_het_graph_word_level_attention(tags_list,window_size=2,fully_connected=False):
    # tags: (typecnt,evitype,word)    evitype= 'sentence' or 'table'

    if fully_connected:
        print('build fully connected network')
        return construct_het_graph_word_level_attention_full(tags_list)
    else:
        print('build locally connected network')
    
    edge_index_list = []
    edge_type_list = []


    for tags in tqdm(tags_list):

        num_node = len(tags)
        edge_index= []
        edge_type = []
        
        temp_word_idx_dic = {}
        for i in range(num_node):
            
            if tags[i][2] not in NoneSenseWord: # do not consider NoneSenseWord
                if tags[i][2] not in temp_word_idx_dic.keys():
                    temp_word_idx_dic[tags[i][2]] = [(i,tags[i][0],tags[i][1])] #(node,evidencecnt,evitype)
                else:
                    temp_word_idx_dic[tags[i][2]].append((i,tags[i][0],tags[i][1]))

            for j in range(i,min(num_node,i+window_size+1)):

                if abs(i-j)<=window_size and (tags[i][0]==tags[j][0]): # two word in same evidence/claim
                    edge_index.append([i,j])
                    edge_index.append([j,i])
                    if tags[i][1] == 'table':
                        edge_type.append(1)
                        edge_type.append(1)
                    else:
                        edge_type.append(0)
                        edge_type.append(0)
        
        # cooccurance
        for k in temp_word_idx_dic.keys():
            if len(temp_word_idx_dic[k])<2:
                continue
            for a in temp_word_idx_dic[k]:
                for b in temp_word_idx_dic[k]:
                    if a[0]!=b[0]:
                        if a[1]!=b[1]: # do not link same words if the words locate in same evidence 
                            if a[2]=='sentence' or b[2]=='sentence': # we do not connect words both in table evidence
                                edge_index.append([a[0],b[0]])
                                edge_type.append(2)


        edge_index = np.array(edge_index).T
        edge_type = np.array(edge_type)

        edge_index_list.append(edge_index)
        edge_type_list.append(edge_type)


    return edge_index_list,edge_type_list
#################################################

def construct_het_graph_word_level_attention_full(tags_list):
 # tags: (typecnt,evitype,word)    evitype= 'sentence' or 'table'
    edge_index_list = []
    edge_type_list = []
    print('start building')

    for tags in tqdm(tags_list):
        num_node = len(tags)
        edge_index= []
        edge_type = []
        for i in range(num_node):
            for j in range(i,num_node):

                if tags[i][0]==tags[j][0]: # two word in same evidence/claim
                    edge_index.append([i,j])
                    edge_index.append([j,i])
                    if tags[i][1] == 'table':
                        edge_type.append(1)
                        edge_type.append(1)
                    else:
                        edge_type.append(0)
                        edge_type.append(0)
                else:
                    edge_index.append([i,j])
                    edge_index.append([j,i])
                    edge_type.append(2)
                    edge_type.append(2)


        edge_index = np.array(edge_index).T
        edge_type = np.array(edge_type)

        edge_index_list.append(edge_index)
        edge_type_list.append(edge_type)
    return edge_index_list,edge_type_list


