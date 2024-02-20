from turtle import forward
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv,RGCNConv, GATConv, RGATConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from copy import deepcopy as copy
# from wikipedia2vec import Wikipedia2Vec
# wiki = Wikipedia2Vec.load('entity_pretrained/enwiki_20180420_300d.pkl')

class T2GV2_noEnt(nn.Module):
    name = 'T2GV2_noEnt'
    def __init__(self, plm_model, in_channels, hidden_channels, out_channels, num_relations, linear_hidden_channels, num_class,layers=2):
        super(T2GV2_noEnt, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.trans_entity = nn.Linear(in_channels+300,in_channels)
        self.plm_model = plm_model
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.linear_hidden_channels = linear_hidden_channels
        self.num_class = num_class
        self.dropout = nn.Dropout(0.5)
        self.layers = layers
        # self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        # self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)
        gnnlist = []
        if layers>1:
            layercnt = 1
            while True:
                if layercnt==1: 
                    gnnlist.append(RGCNConv(in_channels, hidden_channels, num_relations))
                elif layercnt==layers:
                    gnnlist.append(RGCNConv(hidden_channels, out_channels, num_relations))
                    break
                else:
                    gnnlist.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
                layercnt+=1
        elif layers == 0:
            gnnlist.append(nn.Linear(in_channels,out_channels,bias=False))
        elif layers == 1:
            gnnlist.append(RGCNConv(in_channels, out_channels, num_relations))
        else:
            print('Initial Error')
        
        self.gnns = nn.ModuleList(gnnlist)
        print(self.gnns,'has added/')
        del gnnlist



        self.linear1 = nn.Linear(out_channels*2+in_channels, linear_hidden_channels)
        self.linear2 = nn.Linear(linear_hidden_channels, num_class)
        self.act_inner = nn.ReLU()
        self.act_out = nn.LogSoftmax(dim=-1)
        self.att_linear0 = nn.Linear(in_channels+out_channels*2,out_channels,bias=False)
        self.att_linear1 = nn.Linear(out_channels,1,bias=False)
        self.softmax = nn.Softmax(dim=0)

    
    def word_select(self,input,valid):
        # Entity words' representations are  fused
        output = []
        for idx in valid:
            if idx[0] == -31:
                # print('DEBUG in model.py ABOUT ENTITY NOTION')
                output.append(
                    torch.mean(input[list(idx[2]),:],axis=0).unsqueeze(0)
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output


    '''
    # This is an older version
    # Entity words' representations are substituted

    def word_select(self,input,valid):
        output = []
        for idx in valid:
            if idx[0] == -31:
                output.append( 
                    self.trans_entity(
                        torch.tensor(wiki.get_entity_vector(idx[1]).reshape(1,300)).float().cuda()
                    )
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output
    '''

    def forward(self, edge_index, edge_type, input_ids, num_claim_evi,attn_mask, valid_idx,word_cnt,concat_token_ids,concat_attn_mask):
        # print(input_ids.shape,attn_mask.shape)############### this is import to check out
        # print("One Time:",input_ids,attn_mask)
        outputs = self.plm_model(input_ids, attn_mask, output_hidden_states=True).hidden_states[-1][:,:,:]
        
        # outputs = self.dropout(outputs)
        x = []
        claim_CLS = [] # claims representation



        claim_evi_cnt = 0
        for i,(num,valid) in enumerate(zip(num_claim_evi,valid_idx)):
            x += self.word_select(outputs[claim_evi_cnt:claim_evi_cnt+num,:,:].reshape(-1,self.in_channels),valid)
            claim_CLS.append(outputs[claim_evi_cnt][0])
            claim_evi_cnt += num
    
        # init_node_rep = torch.concat(raw,dim=0)
        # out_rep_1 = self.rgcn1(init_node_rep, edge_index, edge_type)
        # out_rep_act_1 = self.act_inner(out_rep_1)
        # out_rep_2 = self.rgcn2(out_rep_act_1, edge_index, edge_type)
        # out_rep_act_2 = self.act_inner(out_rep_2).view(-1, self.out_channels)
        x = torch.concat(x,dim=0)
        if self.layers>0:
            for i in range(self.layers):
                x = self.gnns[i](x,edge_index,edge_type)
                x = self.act_inner(x)
            x = x.view(-1,self.out_channels)
        elif self.layers==0:
            x = self.gnns[0](x).view(-1,self.out_channels)

        
        # print(valid_idx)
        graph_rep_list = [] 
        num = 0
        attention_list = []
        for i in range(len(num_claim_evi)):
            # graph_rep_list.append(torch.mean(out_rep_act_2[num:num+len(valid_idx[i])], dim=0))
            per_claim_words = x[num:num+len(valid_idx[i])]
            evidence_reps = []
            temp_sum = 0
            for num_word in word_cnt[i]:
                evidence_reps.append( 
                    torch.concat(
                            [
                                torch.mean(per_claim_words[temp_sum:temp_sum+num_word],dim=0) ,
                                torch.max(per_claim_words[temp_sum:temp_sum+num_word],dim=0)[0]
                            ],
                            dim=0
                        )
                    )
                temp_sum += num_word
            evidence_reps = torch.stack(evidence_reps,dim=0)
            
            p = self.att_linear1(self.act_inner(self.att_linear0(   torch.cat([claim_CLS[i].repeat([evidence_reps.shape[0],1]),evidence_reps],dim=1)    )))
            

            # print('DEBUG alpha',alpha.shape)

            graph_rep_list.append(   ((self.softmax(p).T)@evidence_reps).squeeze()    )
            
            # print('#DEBuG',p,p.shape)
            attention_list.append(self.sigmoid(p).squeeze(1))
     
            num += len(valid_idx[i])
        
        # attention_list = torch.concat(attention_list,dim=0)
        # print('DEBUG attention',len(attention_list),attention_list[0].shape)
        
        
        out_graph_rep = torch.stack(graph_rep_list, dim=0)
        out_graph_rep = torch.concat([out_graph_rep,self.plm_model(concat_token_ids,concat_attn_mask,output_hidden_states=True).hidden_states[-1][:,0,:]],dim=1)
        out_graph_rep = self.dropout(out_graph_rep)
        out_graph_rep = self.act_inner(self.linear1(out_graph_rep))
        out_graph_rep = self.linear2(out_graph_rep)
        out_graph_rep = self.act_out(out_graph_rep)
        out_graph_rep = out_graph_rep.view(-1, self.num_class)
        # return prob,attention_list
        return out_graph_rep,attention_list






class T2GV2_HOMO(nn.Module):
    name = 'T2GV2_HOMO'
    def __init__(self, plm_model, in_channels, hidden_channels, out_channels, num_relations, linear_hidden_channels, num_class,layers=2):
        super(T2GV2_HOMO, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.trans_entity = nn.Linear(in_channels+300,in_channels)
        self.plm_model = plm_model
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.linear_hidden_channels = linear_hidden_channels
        self.num_class = num_class
        self.dropout = nn.Dropout(0.5)
        self.layers = layers
        # self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        # self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)
        gnnlist = []
        if layers>1:
            layercnt = 1
            while True:
                if layercnt==1: 
                    gnnlist.append(GCNConv(in_channels, hidden_channels))
                elif layercnt==layers:
                    gnnlist.append(GCNConv(hidden_channels, out_channels))
                    break
                else:
                    gnnlist.append(GCNConv(hidden_channels, hidden_channels))
                layercnt+=1
        elif layers == 0:
            gnnlist.append(nn.Linear(in_channels,out_channels,bias=False))
        elif layers == 1:
            gnnlist.append(GCNConv(in_channels, out_channels))
        else:
            print('Initial Error')
        
        self.gnns = nn.ModuleList(gnnlist)
        print(self.gnns,'has added/')
        del gnnlist



        self.linear1 = nn.Linear(out_channels*2+in_channels, linear_hidden_channels)
        self.linear2 = nn.Linear(linear_hidden_channels, num_class)
        self.act_inner = nn.ReLU()
        self.act_out = nn.LogSoftmax(dim=-1)
        self.att_linear0 = nn.Linear(in_channels+out_channels*2,out_channels,bias=False)
        self.att_linear1 = nn.Linear(out_channels,1,bias=False)
        self.softmax = nn.Softmax(dim=0)

    
    def word_select(self,input,valid):
        # Entity words' representations are  fused
        output = []
        for idx in valid:
            if idx[0] == -31:
                # print('DEBUG in model.py ABOUT ENTITY NOTION')
                output.append(
                    torch.mean(input[list(idx[2]),:],axis=0).unsqueeze(0)
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output


    '''
    # This is an older version
    # Entity words' representations are substituted

    def word_select(self,input,valid):
        output = []
        for idx in valid:
            if idx[0] == -31:
                output.append( 
                    self.trans_entity(
                        torch.tensor(wiki.get_entity_vector(idx[1]).reshape(1,300)).float().cuda()
                    )
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output
    '''

    def forward(self, edge_index, edge_type, input_ids, num_claim_evi,attn_mask, valid_idx,word_cnt,concat_token_ids,concat_attn_mask):
        # print(input_ids.shape,attn_mask.shape)############### this is import to check out
        # print("One Time:",input_ids,attn_mask)
        outputs = self.plm_model(input_ids, attn_mask, output_hidden_states=True).hidden_states[-1][:,:,:]
        
        # outputs = self.dropout(outputs)
        x = []
        claim_CLS = [] # claims representation



        claim_evi_cnt = 0
        for i,(num,valid) in enumerate(zip(num_claim_evi,valid_idx)):
            x += self.word_select(outputs[claim_evi_cnt:claim_evi_cnt+num,:,:].reshape(-1,self.in_channels),valid)
            claim_CLS.append(outputs[claim_evi_cnt][0])
            claim_evi_cnt += num
    
        # init_node_rep = torch.concat(raw,dim=0)
        # out_rep_1 = self.rgcn1(init_node_rep, edge_index, edge_type)
        # out_rep_act_1 = self.act_inner(out_rep_1)
        # out_rep_2 = self.rgcn2(out_rep_act_1, edge_index, edge_type)
        # out_rep_act_2 = self.act_inner(out_rep_2).view(-1, self.out_channels)
        x = torch.concat(x,dim=0)
        if self.layers>0:
            for i in range(self.layers):
                x = self.gnns[i](x,edge_index)
                x = self.act_inner(x)
            x = x.view(-1,self.out_channels)
        elif self.layers==0:
            x = self.gnns[0](x).view(-1,self.out_channels)

        
        # print(valid_idx)
        graph_rep_list = [] 
        num = 0
        attention_list = []
        for i in range(len(num_claim_evi)):
            # graph_rep_list.append(torch.mean(out_rep_act_2[num:num+len(valid_idx[i])], dim=0))
            per_claim_words = x[num:num+len(valid_idx[i])]
            evidence_reps = []
            temp_sum = 0
            for num_word in word_cnt[i]:
                evidence_reps.append( 
                    torch.concat(
                            [
                                torch.mean(per_claim_words[temp_sum:temp_sum+num_word],dim=0) ,
                                torch.max(per_claim_words[temp_sum:temp_sum+num_word],dim=0)[0]
                            ],
                            dim=0
                        )
                    )
                temp_sum += num_word
            evidence_reps = torch.stack(evidence_reps,dim=0)
            
            p = self.att_linear1(self.act_inner(self.att_linear0(   torch.cat([claim_CLS[i].repeat([evidence_reps.shape[0],1]),evidence_reps],dim=1)    )))
            

            # print('DEBUG alpha',alpha.shape)

            graph_rep_list.append(   ((self.softmax(p).T)@evidence_reps).squeeze()    )
            
            # print('#DEBuG',p,p.shape)
            attention_list.append(self.sigmoid(p).squeeze(1))
     
            num += len(valid_idx[i])
        
        # attention_list = torch.concat(attention_list,dim=0)
        # print('DEBUG attention',len(attention_list),attention_list[0].shape)
        
        
        out_graph_rep = torch.stack(graph_rep_list, dim=0)
        out_graph_rep = torch.concat([out_graph_rep,self.plm_model(concat_token_ids,concat_attn_mask,output_hidden_states=True).hidden_states[-1][:,0,:]],dim=1)
        out_graph_rep = self.dropout(out_graph_rep)
        out_graph_rep = self.act_inner(self.linear1(out_graph_rep))
        out_graph_rep = self.linear2(out_graph_rep)
        out_graph_rep = self.act_out(out_graph_rep)
        out_graph_rep = out_graph_rep.view(-1, self.num_class)
        # return prob,attention_list
        return out_graph_rep,attention_list


class T2GV2_noConcat(nn.Module):
    name = 'T2GV2_noConcat'
    def __init__(self, plm_model, in_channels, hidden_channels, out_channels, num_relations, linear_hidden_channels, num_class,layers=2):
        super(T2GV2_noConcat, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.trans_entity = nn.Linear(in_channels+300,in_channels)
        self.plm_model = plm_model
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.linear_hidden_channels = linear_hidden_channels
        self.num_class = num_class
        self.dropout = nn.Dropout(0.5)
        self.layers = layers
        # self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        # self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)
        gnnlist = []
        if layers>1:
            layercnt = 1
            while True:
                if layercnt==1: 
                    gnnlist.append(RGCNConv(in_channels, hidden_channels, num_relations))
                elif layercnt==layers:
                    gnnlist.append(RGCNConv(hidden_channels, out_channels, num_relations))
                    break
                else:
                    gnnlist.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
                layercnt+=1
        elif layers == 0:
            gnnlist.append(nn.Linear(in_channels,out_channels,bias=False))
        elif layers == 1:
            gnnlist.append(RGCNConv(in_channels, out_channels, num_relations))
        else:
            print('Initial Error')
        
        self.gnns = nn.ModuleList(gnnlist)
        print(self.gnns,'has added/')
        del gnnlist



        self.linear1 = nn.Linear(out_channels*2, linear_hidden_channels)
        self.linear2 = nn.Linear(linear_hidden_channels, num_class)
        self.act_inner = nn.ReLU()
        self.act_out = nn.LogSoftmax(dim=-1)
        self.att_linear0 = nn.Linear(in_channels+out_channels*2,out_channels,bias=False)
        self.att_linear1 = nn.Linear(out_channels,1,bias=False)
        self.softmax = nn.Softmax(dim=0)

    
    def word_select(self,input,valid):
        # Entity words' representations are  fused
        output = []
        for idx in valid:
            if idx[0] == -31:
                # print('DEBUG in model.py ABOUT ENTITY NOTION')
                output.append(
                    torch.mean(input[list(idx[2]),:],axis=0).unsqueeze(0)
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output


    '''
    # This is an older version
    # Entity words' representations are substituted

    def word_select(self,input,valid):
        output = []
        for idx in valid:
            if idx[0] == -31:
                output.append( 
                    self.trans_entity(
                        torch.tensor(wiki.get_entity_vector(idx[1]).reshape(1,300)).float().cuda()
                    )
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output
    '''

    def forward(self, edge_index, edge_type, input_ids, num_claim_evi,attn_mask, valid_idx,word_cnt,concat_token_ids,concat_attn_mask):
        # print(input_ids.shape,attn_mask.shape)############### this is import to check out
        # print("One Time:",input_ids,attn_mask)
        outputs = self.plm_model(input_ids, attn_mask, output_hidden_states=True).hidden_states[-1][:,:,:]
        
        # outputs = self.dropout(outputs)
        x = []
        claim_CLS = [] # claims representation



        claim_evi_cnt = 0
        for i,(num,valid) in enumerate(zip(num_claim_evi,valid_idx)):
            x += self.word_select(outputs[claim_evi_cnt:claim_evi_cnt+num,:,:].reshape(-1,self.in_channels),valid)
            claim_CLS.append(outputs[claim_evi_cnt][0])
            claim_evi_cnt += num
    
        # init_node_rep = torch.concat(raw,dim=0)
        # out_rep_1 = self.rgcn1(init_node_rep, edge_index, edge_type)
        # out_rep_act_1 = self.act_inner(out_rep_1)
        # out_rep_2 = self.rgcn2(out_rep_act_1, edge_index, edge_type)
        # out_rep_act_2 = self.act_inner(out_rep_2).view(-1, self.out_channels)
        x = torch.concat(x,dim=0)
        if self.layers>0:
            for i in range(self.layers):
                x = self.gnns[i](x,edge_index,edge_type)
                x = self.act_inner(x)
            x = x.view(-1,self.out_channels)
        elif self.layers==0:
            x = self.gnns[0](x).view(-1,self.out_channels)

        
        # print(valid_idx)
        graph_rep_list = [] 
        num = 0
        attention_list = []
        for i in range(len(num_claim_evi)):
            # graph_rep_list.append(torch.mean(out_rep_act_2[num:num+len(valid_idx[i])], dim=0))
            per_claim_words = x[num:num+len(valid_idx[i])]
            evidence_reps = []
            temp_sum = 0
            for num_word in word_cnt[i]:
                evidence_reps.append( 
                    torch.concat(
                            [
                                torch.mean(per_claim_words[temp_sum:temp_sum+num_word],dim=0) ,
                                torch.max(per_claim_words[temp_sum:temp_sum+num_word],dim=0)[0]
                            ],
                            dim=0
                        )
                    )
                temp_sum += num_word
            evidence_reps = torch.stack(evidence_reps,dim=0)
            
            p = self.att_linear1(self.act_inner(self.att_linear0(   torch.cat([claim_CLS[i].repeat([evidence_reps.shape[0],1]),evidence_reps],dim=1)    )))
            

            # print('DEBUG alpha',alpha.shape)

            graph_rep_list.append(   ((self.softmax(p).T)@evidence_reps).squeeze()    )
            
            # print('#DEBuG',p,p.shape)
            attention_list.append(self.sigmoid(p).squeeze(1))
     
            num += len(valid_idx[i])
        
        # attention_list = torch.concat(attention_list,dim=0)
        # print('DEBUG attention',len(attention_list),attention_list[0].shape)
        
        
        out_graph_rep = torch.stack(graph_rep_list, dim=0)
        # out_graph_rep = torch.concat([out_graph_rep,self.plm_model(concat_token_ids,concat_attn_mask,output_hidden_states=True).hidden_states[-1][:,0,:]],dim=1)
        out_graph_rep = self.dropout(out_graph_rep)
        out_graph_rep = self.act_inner(self.linear1(out_graph_rep))
        out_graph_rep = self.linear2(out_graph_rep)
        out_graph_rep = self.act_out(out_graph_rep)
        out_graph_rep = out_graph_rep.view(-1, self.num_class)
        # return prob,attention_list
        return out_graph_rep,attention_list

class T2GV2_Mean(nn.Module):
    name = 'T2GV2_Mean'
    def __init__(self, plm_model, in_channels, hidden_channels, out_channels, num_relations, linear_hidden_channels, num_class,layers=2):
        super(T2GV2_Mean, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.trans_entity = nn.Linear(in_channels+300,in_channels)
        self.plm_model = plm_model
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.linear_hidden_channels = linear_hidden_channels
        self.num_class = num_class
        self.dropout = nn.Dropout(0.5)
        self.layers = layers
        # self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        # self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)
        gnnlist = []
        if layers>1:
            layercnt = 1
            while True:
                if layercnt==1: 
                    gnnlist.append(RGCNConv(in_channels, hidden_channels, num_relations))
                elif layercnt==layers:
                    gnnlist.append(RGCNConv(hidden_channels, out_channels, num_relations))
                    break
                else:
                    gnnlist.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
                layercnt+=1
        elif layers == 0:
            gnnlist.append(nn.Linear(in_channels,out_channels,bias=False))
        elif layers == 1:
            gnnlist.append(RGCNConv(in_channels, out_channels, num_relations))
        else:
            print('Initial Error')
        
        self.gnns = nn.ModuleList(gnnlist)
        print(self.gnns,'has added/')
        del gnnlist



        self.linear1 = nn.Linear(out_channels*2+in_channels, linear_hidden_channels)
        self.linear2 = nn.Linear(linear_hidden_channels, num_class)
        self.act_inner = nn.ReLU()
        self.act_out = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=0)

    
    def word_select(self,input,valid):
        # Entity words' representations are  fused
        output = []
        for idx in valid:
            if idx[0] == -31:
                # print('DEBUG in model.py ABOUT ENTITY NOTION')
                output.append(
                    torch.mean(input[list(idx[2]),:],axis=0).unsqueeze(0)
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output


    '''
    # This is an older version
    # Entity words' representations are substituted

    def word_select(self,input,valid):
        output = []
        for idx in valid:
            if idx[0] == -31:
                output.append( 
                    self.trans_entity(
                        torch.tensor(wiki.get_entity_vector(idx[1]).reshape(1,300)).float().cuda()
                    )
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output
    '''

    def forward(self, edge_index, edge_type, input_ids, num_claim_evi,attn_mask, valid_idx,word_cnt,concat_token_ids,concat_attn_mask):
        # print(input_ids.shape,attn_mask.shape)############### this is import to check out
        # print("One Time:",input_ids,attn_mask)
        outputs = self.plm_model(input_ids, attn_mask, output_hidden_states=True).hidden_states[-1][:,:,:]
        
        # outputs = self.dropout(outputs)
        x = []
        claim_CLS = [] # claims representation



        claim_evi_cnt = 0
        for i,(num,valid) in enumerate(zip(num_claim_evi,valid_idx)):
            x += self.word_select(outputs[claim_evi_cnt:claim_evi_cnt+num,:,:].reshape(-1,self.in_channels),valid)
            claim_CLS.append(outputs[claim_evi_cnt][0])
            claim_evi_cnt += num
    
        # init_node_rep = torch.concat(raw,dim=0)
        # out_rep_1 = self.rgcn1(init_node_rep, edge_index, edge_type)
        # out_rep_act_1 = self.act_inner(out_rep_1)
        # out_rep_2 = self.rgcn2(out_rep_act_1, edge_index, edge_type)
        # out_rep_act_2 = self.act_inner(out_rep_2).view(-1, self.out_channels)
        x = torch.concat(x,dim=0)
        if self.layers>0:
            for i in range(self.layers):
                x = self.gnns[i](x,edge_index,edge_type)
                x = self.act_inner(x)
            x = x.view(-1,self.out_channels)
        elif self.layers==0:
            x = self.gnns[0](x).view(-1,self.out_channels)

        
        # print(valid_idx)
        graph_rep_list = [] 
        num = 0
        attention_list = []
        for i in range(len(num_claim_evi)):
            # graph_rep_list.append(torch.mean(out_rep_act_2[num:num+len(valid_idx[i])], dim=0))
            per_claim_words = x[num:num+len(valid_idx[i])]
            evidence_reps = []
            temp_sum = 0
            for num_word in word_cnt[i]:
                evidence_reps.append( 
                    torch.concat(
                            [
                                torch.mean(per_claim_words[temp_sum:temp_sum+num_word],dim=0) ,
                                torch.max(per_claim_words[temp_sum:temp_sum+num_word],dim=0)[0]
                            ],
                            dim=0
                        )
                    )
                temp_sum += num_word
            evidence_reps = torch.stack(evidence_reps,dim=0)
            

            # print('DEBUG alpha',alpha.shape)

            graph_rep_list.append(   torch.mean(evidence_reps,dim=0).squeeze()    )
            
            # print('#DEBuG',p,p.shape)
            attention_list.append(None)
     
            num += len(valid_idx[i])
        
        # attention_list = torch.concat(attention_list,dim=0)
        # print('DEBUG attention',len(attention_list),attention_list[0].shape)
        
        
        out_graph_rep = torch.stack(graph_rep_list, dim=0)
        out_graph_rep = torch.concat([out_graph_rep,self.plm_model(concat_token_ids,concat_attn_mask,output_hidden_states=True).hidden_states[-1][:,0,:]],dim=1)
        out_graph_rep = self.dropout(out_graph_rep)
        out_graph_rep = self.act_inner(self.linear1(out_graph_rep))
        out_graph_rep = self.linear2(out_graph_rep)
        out_graph_rep = self.act_out(out_graph_rep)
        out_graph_rep = out_graph_rep.view(-1, self.num_class)
        # return prob,attention_list
        return out_graph_rep,attention_list


class T2GV2_Max(nn.Module):
    name = 'T2GV2_Max'
    def __init__(self, plm_model, in_channels, hidden_channels, out_channels, num_relations, linear_hidden_channels, num_class,layers=2):
        super(T2GV2_Max, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.trans_entity = nn.Linear(in_channels+300,in_channels)
        self.plm_model = plm_model
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.linear_hidden_channels = linear_hidden_channels
        self.num_class = num_class
        self.dropout = nn.Dropout(0.5)
        self.layers = layers
        # self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        # self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)
        gnnlist = []
        if layers>1:
            layercnt = 1
            while True:
                if layercnt==1: 
                    gnnlist.append(RGCNConv(in_channels, hidden_channels, num_relations))
                elif layercnt==layers:
                    gnnlist.append(RGCNConv(hidden_channels, out_channels, num_relations))
                    break
                else:
                    gnnlist.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
                layercnt+=1
        elif layers == 0:
            gnnlist.append(nn.Linear(in_channels,out_channels,bias=False))
        elif layers == 1:
            gnnlist.append(RGCNConv(in_channels, out_channels, num_relations))
        else:
            print('Initial Error')
        
        self.gnns = nn.ModuleList(gnnlist)
        print(self.gnns,'has added/')
        del gnnlist



        self.linear1 = nn.Linear(out_channels*2+in_channels, linear_hidden_channels)
        self.linear2 = nn.Linear(linear_hidden_channels, num_class)
        self.act_inner = nn.ReLU()
        self.act_out = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=0)

    
    def word_select(self,input,valid):
        # Entity words' representations are  fused
        output = []
        for idx in valid:
            if idx[0] == -31:
                # print('DEBUG in model.py ABOUT ENTITY NOTION')
                output.append(
                    torch.mean(input[list(idx[2]),:],axis=0).unsqueeze(0)
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output


    '''
    # This is an older version
    # Entity words' representations are substituted

    def word_select(self,input,valid):
        output = []
        for idx in valid:
            if idx[0] == -31:
                output.append( 
                    self.trans_entity(
                        torch.tensor(wiki.get_entity_vector(idx[1]).reshape(1,300)).float().cuda()
                    )
                )
            else:
                output.append( torch.mean(input[list(idx),:],axis=0).unsqueeze(0) )
        return output
    '''

    def forward(self, edge_index, edge_type, input_ids, num_claim_evi,attn_mask, valid_idx,word_cnt,concat_token_ids,concat_attn_mask):
        # print(input_ids.shape,attn_mask.shape)############### this is import to check out
        # print("One Time:",input_ids,attn_mask)
        outputs = self.plm_model(input_ids, attn_mask, output_hidden_states=True).hidden_states[-1][:,:,:]
        
        # outputs = self.dropout(outputs)
        x = []
        claim_CLS = [] # claims representation



        claim_evi_cnt = 0
        for i,(num,valid) in enumerate(zip(num_claim_evi,valid_idx)):
            x += self.word_select(outputs[claim_evi_cnt:claim_evi_cnt+num,:,:].reshape(-1,self.in_channels),valid)
            claim_CLS.append(outputs[claim_evi_cnt][0])
            claim_evi_cnt += num
    
        # init_node_rep = torch.concat(raw,dim=0)
        # out_rep_1 = self.rgcn1(init_node_rep, edge_index, edge_type)
        # out_rep_act_1 = self.act_inner(out_rep_1)
        # out_rep_2 = self.rgcn2(out_rep_act_1, edge_index, edge_type)
        # out_rep_act_2 = self.act_inner(out_rep_2).view(-1, self.out_channels)
        x = torch.concat(x,dim=0)
        if self.layers>0:
            for i in range(self.layers):
                x = self.gnns[i](x,edge_index,edge_type)
                x = self.act_inner(x)
            x = x.view(-1,self.out_channels)
        elif self.layers==0:
            x = self.gnns[0](x).view(-1,self.out_channels)

        
        # print(valid_idx)
        graph_rep_list = [] 
        num = 0
        attention_list = []
        for i in range(len(num_claim_evi)):
            # graph_rep_list.append(torch.mean(out_rep_act_2[num:num+len(valid_idx[i])], dim=0))
            per_claim_words = x[num:num+len(valid_idx[i])]
            evidence_reps = []
            temp_sum = 0
            for num_word in word_cnt[i]:
                evidence_reps.append( 
                    torch.concat(
                            [
                                torch.mean(per_claim_words[temp_sum:temp_sum+num_word],dim=0) ,
                                torch.max(per_claim_words[temp_sum:temp_sum+num_word],dim=0)[0]
                            ],
                            dim=0
                        )
                    )
                temp_sum += num_word
            evidence_reps = torch.stack(evidence_reps,dim=0)
            

            # print('DEBUG alpha',alpha.shape)

            graph_rep_list.append(   torch.max(evidence_reps,dim=0)[0].squeeze()    )
            
            # print('#DEBuG',p,p.shape)
            attention_list.append(None)
     
            num += len(valid_idx[i])
        
        # attention_list = torch.concat(attention_list,dim=0)
        # print('DEBUG attention',len(attention_list),attention_list[0].shape)
        
        
        out_graph_rep = torch.stack(graph_rep_list, dim=0)
        out_graph_rep = torch.concat([out_graph_rep,self.plm_model(concat_token_ids,concat_attn_mask,output_hidden_states=True).hidden_states[-1][:,0,:]],dim=1)
        out_graph_rep = self.dropout(out_graph_rep)
        out_graph_rep = self.act_inner(self.linear1(out_graph_rep))
        out_graph_rep = self.linear2(out_graph_rep)
        out_graph_rep = self.act_out(out_graph_rep)
        out_graph_rep = out_graph_rep.view(-1, self.num_class)
        # return prob,attention_list
        return out_graph_rep,attention_list


