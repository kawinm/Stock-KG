import torch
import torch.nn as nn 
import math 
import torch.nn.functional as F

from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge, HypergraphConv
from torch_geometric.nn import global_mean_pool

from torchkge.models.translation import TorusEModel
from torchkge.models.bilinear import ComplExModel, HolEModel
from .tkge_models import *

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # TODO: Check div term, correct or wrong
        div_term = torch.exp(torch.arange(0, 2*d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)[:, 0::2]
        #print(pe.shape, position.shape, div_term.shape)
        pe[:, 1::2] = torch.cos(position * div_term)[:, 1::2]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        # return x

# ----------- Model -----------
class Transformer_Ranking(nn.Module):
    
    def __init__(self, W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING = False, USE_GRAPH = False, HYPER_GRAPH = True, USE_KG = True, NUM_NODES = 87):
        super().__init__()

        SEC_EMB, n = 25, 0 # 1 For LSTM Embedding
        if USE_GRAPH:
            n += 1
        if USE_KG:
            n += 2

        self.embeddings = nn.Embedding(105, 10)

        self.pos_enc_x = PositionalEncoding(d_model=D_MODEL, dropout=DROPOUT, max_len=W)
        self.pos_enc_y = PositionalEncoding(d_model=D_MODEL, dropout=DROPOUT, max_len=T)

        self.lstm_encoder = nn.LSTM(input_size = 5, hidden_size = D_MODEL, num_layers = ENC_LAYERS, batch_first = True, bidirectional = False)

        #encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF, batch_first=True )
        #self.transformer_encoder_first = nn.TransformerEncoder(encoder_layer, num_layers=ENC_LAYERS)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF, batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=ENC_LAYERS)

        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.pred = nn.Linear((D_MODEL+(SEC_EMB*n)) * NUM_NODES, NUM_NODES*10)
        self.pred2 = nn.Linear(NUM_NODES*10, NUM_NODES)

        self.hold_pred = nn.Linear(D_MODEL+(SEC_EMB*n), 1)

        self.is_pos = USE_POS_ENCODING
        self.time_steps = T

        

        self.use_graph = USE_GRAPH
        self.is_hyper_graph = HYPER_GRAPH
        if self.use_graph:
            if self.is_hyper_graph:
                self.graph_model = Sequential('x, hyperedge_index', [
                        #(Dropout(p=0.5), 'x -> x'),
                        (HypergraphConv(8, 32, dropout=0.1), 'x, hyperedge_index -> x1'),
                        nn.LeakyReLU(inplace=True),
                        (HypergraphConv(32, 32, dropout=0.1), 'x1, hyperedge_index -> x2'),
                        nn.LeakyReLU(inplace=True),
                        
                        #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                        #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                        #(global_mean_pool, 'x, batch -> x'),
                        nn.Linear(32, SEC_EMB),
                    ])
            else:
                self.graph_model = Sequential('x, edge_index, batch', [
                            #(Dropout(p=0.5), 'x -> x'),
                            (GCNConv(8, 32), 'x, edge_index -> x1'),
                            nn.ReLU(inplace=True),
                            (GCNConv(32, 64), 'x1, edge_index -> x2'),
                            nn.ReLU(inplace=True),
                            #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                            #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                            #(global_mean_pool, 'x, batch -> x'),
                            nn.Linear(64, SEC_EMB),
                        ])
                
        self.use_kg = USE_KG

        config = {
            'entity_total': 6500,
            'relation_total': 57,
            'embedding_size': SEC_EMB,
            'L1_flag': False,
        }
        if self.use_kg:
            #self.relation_kge = TorusEModel(n_entities= 5500, n_relations = 40, emb_dim = SEC_EMB, dissimilarity_type='torus_L2')
            #self.relation_kge = HolEModel(n_entities= 5500, n_relations = 40, emb_dim = SEC_EMB)
             
            self.kge = TTransEModel(config)
        if self.use_kg:
            pass
            #self.temporal_kge = TorusEModel(n_entities= 5500, n_relations = 50, emb_dim = SEC_EMB, dissimilarity_type='torus_L2')
            #self.temporal_kge = HolEModel(n_entities= 5500, n_relations = 40, emb_dim = SEC_EMB) 
        self.num_nodes = NUM_NODES

  
    def forward(self, xb, yb=None, graph=None, kg=None, tkg=None):
        if self.is_pos:
            xb = self.pos_enc_x(xb)
            yb = self.pos_enc_y(yb)
        #yb = torch.cat((yb, emb2), dim=2)

        # # Experiment 1
        x, y = self.lstm_encoder(xb)
        xb = y[0][-1, :, :].unsqueeze(dim=0)          # x: [B, C, W*F
        
        # # Experiment 2
        #W,F = xb.shape[1], xb.shape[2]
        #xb = xb.unsqueeze(dim=0).view(1, -1, W*F)

        # # Experiment 3
        #x = self.transformer_encoder_first(xb).mean(dim=1)
        #xb = x.unsqueeze(dim=0)

        #xb = xb[:, :, 3].squeeze().unsqueeze(dim=0)
        #x = self.transformer_encoder(xb)               # x: [B, C, W*F]
        x = self.fc2(F.dropout(F.leaky_relu(self.fc1(xb)), p=0.2))
        #x = x + xb
        #x = torch.cat((x, emb2), dim=2)

        if self.use_graph and self.is_hyper_graph:
            g_emb = self.graph_model(graph['x'], graph['hyperedge_index']).unsqueeze(dim=0)
            #g_emb = g_emb.repeat(1, self.time_steps, 1)
            x = torch.cat((x, g_emb), dim=2)
        elif self.use_graph and not self.is_hyper_graph:
            g_emb = self.graph_model(graph['x'], graph['edge_list'], graph['batch'])
            #g_emb = g_emb.repeat(1, self.time_steps, 1)        
            x = torch.cat((x, g_emb), dim=1)
        
        kg_loss = torch.zeros(1)
        if self.use_kg:
            #kg_loss = self.relation_kge.scoring_function(kg[0], kg[2], kg[1]).mean()
            #kg_emb, rel_emb = self.relation_kge.get_embeddings()
            #kg_emb = kg_emb[kg[3].long()]

            kg_loss += self.kge(kg[0], kg[2], kg[1])
            kg_emb = self.kge.ent_embeddings.weight[kg[3].long()]
            kg_emb = kg_emb.unsqueeze(dim=0)
            x = torch.cat((x, kg_emb), dim=2)

            #self.relation_kge.normalize_parameters()
        #if self.use_kg:
        #    #head, relation, tail = torch.cat((kg[0], tkg[0])), torch.cat((kg[1], tkg[1])), torch.cat((kg[2], tkg[2]))
        #    kg_loss += self.temporal_kge.scoring_function(tkg[0], tkg[2], tkg[1]).mean()
        #    kg_loss /= 2
        #    kg_emb, rel_emb = self.temporal_kge.get_embeddings()
        #    kg_emb = kg_emb[kg[3].long()]
        #    kg_emb = kg_emb.unsqueeze(dim=0)
        #    x = torch.cat((x, kg_emb), dim=2)

        #    self.temporal_kge.normalize_parameters()

        x = x.view(-1)
        price_pred = self.pred(x)
        price_pred = F.dropout(F.leaky_relu(price_pred), 0.2)
        price_pred = self.pred2(price_pred)
        #price_pred = F.relu(price_pred)

        hold_pred = price_pred #self.hold_pred(x.mean(dim=1)).squeeze(dim=0)
            # x = F.relu(x)
            # x = self.pred2(x)
        return price_pred, kg_loss, hold_pred


# ----------- Model -----------
class Saturation(nn.Module):
    
    def __init__(self, W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING = False, USE_GRAPH = False, HYPER_GRAPH = True, USE_KG = True, NUM_NODES = 87):
        super().__init__()

        SEC_EMB, n = 5, 0 # 1 For LSTM Embedding
        if USE_GRAPH:
            n += 1
        if USE_KG:
            n += 1

        self.lstm_encoder = nn.Linear(D_MODEL, 1)
        self.transformer_encoder = nn.Linear(W, D_MODEL)

        self.use_graph = USE_GRAPH
        self.is_hyper_graph = HYPER_GRAPH
        if self.use_graph:
            if self.is_hyper_graph:
                self.graph_model = Sequential('x, hyperedge_index', [
                        #(Dropout(p=0.5), 'x -> x'),
                        (HypergraphConv(8, 32, dropout=0.1), 'x, hyperedge_index -> x1'),
                        nn.LeakyReLU(inplace=True),
                        (HypergraphConv(32, 32, dropout=0.1), 'x1, hyperedge_index -> x2'),
                        nn.LeakyReLU(inplace=True),
                        
                        #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                        #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                        #(global_mean_pool, 'x, batch -> x'),
                        nn.Linear(32, SEC_EMB),
                    ])
            else:
                self.graph_model = Sequential('x, edge_index, batch', [
                            #(Dropout(p=0.5), 'x -> x'),
                            (GCNConv(8, 32), 'x, edge_index -> x1'),
                            nn.ReLU(inplace=True),
                            (GCNConv(32, 64), 'x1, edge_index -> x2'),
                            nn.ReLU(inplace=True),
                            #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                            #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                            #(global_mean_pool, 'x, batch -> x'),
                            nn.Linear(64, SEC_EMB),
                        ])

        self.pred = nn.Linear(D_MODEL, 1)

    def forward(self, xb, yb=None, graph=None, kg=None):
        x = self.lstm_encoder(xb).squeeze()
        x = self.transformer_encoder(x)               # x: [B, C, W*F]

        if self.use_graph and self.is_hyper_graph:
            g_emb = self.graph_model(graph['x'], graph['hyperedge_index'])
            #g_emb = g_emb.repeat(1, self.time_steps, 1)
            #x = torch.cat((x, g_emb), dim=1)
        elif self.use_graph and not self.is_hyper_graph:
            g_emb = self.graph_model(graph['x'], graph['edge_list'], graph['batch'])
            #g_emb = g_emb.repeat(1, self.time_steps, 1)        
            #x = torch.cat((x, g_emb), dim=1)

        price_pred = self.pred(x)
        return price_pred, 0, 0


