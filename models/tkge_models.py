import os
import math
import pickle
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity


class TTransEModel(nn.Module):
    def __init__(self, config):
        super(TTransEModel, self).__init__()
        self.L1_flag = config['L1_flag']
        self.embedding_size = config['embedding_size']
        self.entity_total = config['entity_total']
        self.relation_total = config['relation_total']

        ent_weight = torch.Tensor(self.entity_total, self.embedding_size)
        rel_weight = torch.Tensor(self.relation_total, self.embedding_size)

        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(ent_weight)
        nn.init.xavier_uniform_(rel_weight)


        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)


        self.year_embeddings    = nn.Embedding(24, self.embedding_size, padding_idx=0)
        self.month_embeddings   = nn.Embedding(13, self.embedding_size, padding_idx=0)
        self.day_embeddings     = nn.Embedding(32, self.embedding_size, padding_idx=0)
        self.hour_embeddings    = nn.Embedding(25, self.embedding_size, padding_idx=0)
        self.minutes_embeddings = nn.Embedding(61, self.embedding_size, padding_idx=0)
        self.sec_embeddings     = nn.Embedding(61, self.embedding_size, padding_idx=0)

        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)

        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb

    def forward(self, pos_h, pos_t, pos_r, pos_tem):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_tem_e = self.year_embeddings(pos_tem[:, 0]) + self.month_embeddings(pos_tem[:, 1]) + \
                    self.day_embeddings(pos_tem[:, 2]) + self.hour_embeddings(pos_tem[:, 3]) + \
                    self.minutes_embeddings(pos_tem[:, 4]) + self.sec_embeddings(pos_tem[:, 5])
 
        neg_h = torch.randint_like(pos_h, low=1, high=self.entity_total).to(pos_h.device)
        neg_t = torch.randint_like(pos_t, low=1, high=self.relation_total).to(pos_t.device)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(pos_r)

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e + pos_tem_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e + pos_tem_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e + pos_tem_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e + pos_tem_e - neg_t_e) ** 2, 1)

        #print(pos, neg)
        loss = torch.mean(pos - neg)
        #print(loss.shape)
        return loss

"""
class TADistmultModel(nn.Module):
    def __init__(self, config):
        super(TADistmultModel, self).__init__()
        self.L1_flag = config['L1_flag']
        self.embedding_size = config['embedding_size']
        self.entity_total = config['entity_total']
        self.relation_total = config['relation_total']

        self.criterion = nn.Softplus()
        torch.nn.BCELoss()

        self.dropout = nn.Dropout(config.dropout)
        self.lstm = LSTMModel(self.embedding_size, n_layer=1)

        self.year_embeddings    = nn.Embedding(24, self.embedding_size, padding_idx=0)
        self.month_embeddings   = nn.Embedding(13, self.embedding_size, padding_idx=0)
        self.day_embeddings     = nn.Embedding(32, self.embedding_size, padding_idx=0)
        self.hour_embeddings    = nn.Embedding(25, self.embedding_size, padding_idx=0)
        self.minutes_embeddings = nn.Embedding(61, self.embedding_size, padding_idx=0)
        self.sec_embeddings     = nn.Embedding(61, self.embedding_size, padding_idx=0)

        ent_weight = floatTensor(self.entity_total, self.embedding_size)
        rel_weight = floatTensor(self.relation_total, self.embedding_size)
        #tem_weight = floatTensor(self.tem_total, self.embedding_size)
        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        #nn.init.xavier_uniform(tem_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        #self.tem_embeddings = nn.Embedding(self.tem_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        #self.tem_embeddings.weight = nn.Parameter(tem_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        #normalize_temporal_emb = F.normalize(self.tem_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb
        #self.tem_embeddings.weight.data = normalize_temporal_emb

    def scoring(self, h, t, r):
        return torch.sum(h * t * r, 1, False)

    def forward(self, pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_rseq_e = self.get_rseq(pos_r, pos_tem)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_rseq_e = self.get_rseq(neg_r, neg_tem)

        pos_h_e = self.dropout(pos_h_e)
        pos_t_e = self.dropout(pos_t_e)
        pos_rseq_e = self.dropout(pos_rseq_e)
        neg_h_e = self.dropout(neg_h_e)
        neg_t_e = self.dropout(neg_t_e)
        neg_rseq_e = self.dropout(neg_rseq_e)

        pos = self.scoring(pos_h_e, pos_t_e, pos_rseq_e)
        neg = self.scoring(neg_h_e, neg_t_e, neg_rseq_e)
        return pos, neg

    def get_rseq(self, r, tem):
        r_e = self.rel_embeddings(r)
        r_e = r_e.unsqueeze(0).transpose(0, 1)

        bs = tem.shape[0]  # batch size
        tem_len = tem.shape[1]
        tem = tem.contiguous()
        tem = tem.view(bs * tem_len)
        token_e = self.tem_embeddings(tem)
        token_e = token_e.view(bs, tem_len, self.embedding_size)
        seq_e = torch.cat((r_e, token_e), 1)

        hidden_tem, y = self.lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        return rseq_e


class TATransEModel(nn.Module):
    def __init__(self, config):
        super(TATransEModel, self).__init__()
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        # print(self.relation_total)
        # exit()
        self.tem_total = 32
        self.batch_size = config.batch_size

        self.dropout = nn.Dropout(config.dropout)
        self.lstm = LSTMModel(self.embedding_size, n_layer=1)

        ent_weight = floatTensor(self.entity_total, self.embedding_size)
        rel_weight = floatTensor(self.relation_total, self.embedding_size)
        tem_weight = floatTensor(self.tem_total, self.embedding_size)
        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        nn.init.xavier_uniform(tem_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.tem_embeddings = nn.Embedding(self.tem_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.tem_embeddings.weight = nn.Parameter(tem_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        normalize_temporal_emb = F.normalize(self.tem_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb
        self.tem_embeddings.weight.data = normalize_temporal_emb

    def forward(self,loss_type,entity_total,pos_h, pos_t, pos_r, pos_tem, neg_h, neg_t, neg_r, neg_tem):
        # print(loss_type)
        # exit()
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_rseq_e = self.get_rseq(pos_r, pos_tem)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_rseq_e = self.get_rseq(neg_r, neg_tem)

        pos_h_e = self.dropout(pos_h_e)
        pos_t_e = self.dropout(pos_t_e)
        pos_rseq_e = self.dropout(pos_rseq_e)
        neg_h_e = self.dropout(neg_h_e)
        neg_t_e = self.dropout(neg_t_e)
        neg_rseq_e = self.dropout(neg_rseq_e)

        # ent_embeddings = self.ent_embeddings
        # print((pos_h[:20]))
        # exit()

        if loss_type == 0:
            if self.L1_flag:
                pos = torch.sum(torch.abs(pos_h_e + pos_rseq_e - pos_t_e), 1)
                neg = torch.sum(torch.abs(neg_h_e + neg_rseq_e - neg_t_e), 1)
            else:
                pos = torch.sum((pos_h_e + pos_rseq_e - pos_t_e) ** 2, 1)
                neg = torch.sum((neg_h_e + neg_rseq_e - neg_t_e) ** 2, 1)
            return pos, neg
        else:
            mylist = list(range(entity_total))
            my_list_tensor = torch.tensor(np.array(mylist)).cuda()
            
            pred_pos_t = pos_h_e + pos_rseq_e 
            pred_neg_t = neg_h_e + neg_rseq_e
            pred_pos_h = pos_t_e - pos_rseq_e
            pred_neg_h =  neg_t_e - neg_rseq_e
            ent_embeddings = self.ent_embeddings(my_list_tensor).cuda()

            n = pred_pos_t.size(0)
            # print(n)
            m = ent_embeddings.size(0)
            # print(m)
            d = pred_pos_t.size(1)
            # print(d)
            pred_pos_t = pred_pos_t.unsqueeze(1).expand(n, m, d).cuda()
            my_list_tensor = ent_embeddings.unsqueeze(0).expand(n, m, d).cuda()
            # print(my_list_tensor.shape)
            # exit() torch.pow(x, 2)     # torch.pow(pred_pos_t - my_list_tensor, 2)
            z1 = (1/(torch.sum(torch.pow(pred_pos_t - my_list_tensor, 2), dim = 2)+0.0001)).cuda()
            # print(z[0][:40])
            # print("dada")
            # print(z1.shape)
            # exit()
            pred1 = F.softmax(z1, dim=0)
            # print(pred1.shape)
            # exit()





            n  = pred_pos_h.size(0)
            m = ent_embeddings.size(0)
            d  = pred_pos_h.size(1)
            pred_pos_h = pred_pos_h.unsqueeze(1).expand(n, m, d).cuda()
            # my_list_tensor = ent_embeddings.unsqueeze(0).expand(n, m, d).cuda()
            z2 = (1/(torch.sum(torch.pow(pred_pos_h - my_list_tensor ,2 ), dim = 2)+0.0001)).cuda()
            pred2 = F.softmax(z2, dim=0)

            
            
            # n  = pred_neg_t.size(0)
            # m = ent_embeddings.size(0)
            # d  = pred_neg_t.size(1)
            # pred_neg_t = pred_neg_t.unsqueeze(1).expand(n, m, d).cuda()
            # # my_list_tensor = ent_embeddings.unsqueeze(0).expand(n, m, d).cuda()
            # z3 = (1/(torch.sum(torch.abs(pred_neg_t - my_list_tensor), dim = 2)+0.0001)*100).cuda()
            # pred3 = F.softmax(z3, dim=0)



            # n  = pred_neg_h.size(0)
            # m = ent_embeddings.size(0)
            # d  = pred_neg_h.size(1)
            # pred_neg_h = pred_neg_h.unsqueeze(1).expand(n, m, d).cuda()
            # # my_list_tensor = ent_embeddings.unsqueeze(0).expand(n, m, d).cuda()
            # z4 = (1/(torch.sum(torch.abs(pred_neg_h - my_list_tensor), dim = 2)+0.0001)*100).cuda()
            # pred4 = F.softmax(z3, dim=0)




            pred = torch.cat((pred1, pred2), 0)
            target = torch.cat((pos_t, pos_h), 0)
     



            return pred,target


            



    def get_rseq(self, r, tem):
        r_e = self.rel_embeddings(r)
        r_e = r_e.unsqueeze(0).transpose(0, 1)

        bs = tem.shape[0]  # batch size
        tem_len = tem.shape[1]
        tem = tem.contiguous()
        tem = tem.view(bs * tem_len)
        token_e = self.tem_embeddings(tem)
        token_e = token_e.view(bs, tem_len, self.embedding_size)
        seq_e = torch.cat((r_e, token_e), 1)

        hidden_tem = self.lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        return rseq_e

"""