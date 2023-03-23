import torch
import torch.nn as nn 
import math 
import torch.nn.functional as F

from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge, HypergraphConv
from torch_geometric.nn import global_mean_pool

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
class Transformer(nn.Module):
    
    def __init__(self, W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING = False, USE_GRAPH = False, HYPER_GRAPH = True):
        super().__init__()

        SEC_EMB = 0
        if USE_GRAPH:
            SEC_EMB = 5

        self.embeddings = nn.Embedding(105, 10)

        self.pos_enc_x = PositionalEncoding(d_model=D_MODEL, dropout=DROPOUT, max_len=W)
        self.pos_enc_y = PositionalEncoding(d_model=D_MODEL, dropout=DROPOUT, max_len=T)

        self.transf = nn.Transformer(d_model=D_MODEL, nhead=N_HEAD, num_encoder_layers=ENC_LAYERS, 
                                        num_decoder_layers=DEC_LAYERS, dim_feedforward=D_FF, 
                                        dropout=DROPOUT, batch_first=True)

        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(1, D_MODEL)
        self.pred = nn.Linear(D_MODEL+SEC_EMB, 1)
        self.pred2 = nn.Linear(5, 1)

        self.is_pos = USE_POS_ENCODING
        self.time_steps = T

        self.cnn = nn.Conv1d(D_MODEL, D_MODEL, 3)  

        self.use_graph = USE_GRAPH
        self.is_hyper_graph = HYPER_GRAPH
        if self.use_graph:
            if self.is_hyper_graph:
                self.graph_model = Sequential('x, hyperedge_index', [
                        #(Dropout(p=0.5), 'x -> x'),
                        (HypergraphConv(8, 32), 'x, hyperedge_index -> x1'),
                        nn.ReLU(inplace=True),
                        (HypergraphConv(32, 64), 'x1, hyperedge_index -> x2'),
                        nn.ReLU(inplace=True),
                        #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                        #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                        #(global_mean_pool, 'x, batch -> x'),
                        nn.Linear(64, SEC_EMB),
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

    def forward(self, xb, yb=None, sector=None, graph=None):
        
        #emb  = self.embeddings(sector).unsqueeze(dim=1)
        #emb1 = emb.repeat(1, W, 1)
        #emb2 = emb.repeat(1, T, 1)
        #x = torch.cat((xb, x), dim=2)

        xb = self.cnn(xb.transpose(1, 2)).transpose(1, 2)
        if self.training:
            yb = torch.cat((xb[:, -1, 3].unsqueeze(dim=1), yb[:, :-1]), dim = 1).unsqueeze(dim=2)
            #xb = self.fc1(xb)
            #xb = torch.cat((xb, emb1), dim=2)

            yb = self.fc2(yb)

            if self.is_pos:
                xb = self.pos_enc_x(xb)
                yb = self.pos_enc_y(yb)
            #yb = torch.cat((yb, emb2), dim=2)
            x = self.transf(xb, yb)
            #x = torch.cat((x, emb2), dim=2)

            if self.use_graph and self.is_hyper_graph:
                g_emb = self.graph_model(graph['x'], graph['hyperedge_index']).unsqueeze(dim=1)
                g_emb = g_emb.repeat(1, self.time_steps, 1)
                x = torch.cat((x, g_emb), dim=2)
            elif self.use_graph and not self.is_hyper_graph:
                g_emb = self.graph_model(graph['x'], graph['edge_list'], graph['batch']).unsqueeze(dim=1)
                g_emb = g_emb.repeat(1, self.time_steps, 1)
                x = torch.cat((x, g_emb), dim=2)

            x = self.pred(x)
            # x = F.relu(x)
            # x = self.pred2(x)
            return x
            
        else:
            yb = xb[:, -1, 3].unsqueeze(dim=1)
            
            if self.is_pos:
                xb = self.pos_enc_x(xb)
            #xb = self.fc1(xb)
            #xb = torch.cat((xb, emb1), dim=2)
            for i in range(self.time_steps):
                y = yb.unsqueeze(dim=2)
                y = self.fc2(y)

                if self.is_pos:
                    y = self.pos_enc_y(y)[:, :i+1, :]
                #y = torch.cat((y, emb2[:, :i+1]), dim=2)
                x = self.transf(xb, y)
                #x = torch.cat((x, emb2[:, :i+1]), dim=2)

                if self.use_graph and self.is_hyper_graph:
                    g_emb = self.graph_model(graph['x'], graph['hyperedge_index']).unsqueeze(dim=1)
                    g_emb = g_emb.repeat(1, x.shape[1], 1)
                    x = torch.cat((x, g_emb), dim=2)
                elif self.use_graph and not self.is_hyper_graph:
                    g_emb = self.graph_model(graph['x'], graph['edge_list'], graph['batch']).unsqueeze(dim=1)
                    g_emb = g_emb.repeat(1, x.shape[1], 1)
                    x = torch.cat((x, g_emb), dim=2)

                x = self.pred(x)
                # x = F.relu(x)
                # x = self.pred2(x)
                yb = torch.cat((yb, x[:,-1]), dim = 1)        
            return yb[:, 1:]

# ----------- Model -----------
class BILSTM(nn.Module):
    
    def __init__(self, W, T, DROPOUT):
        super().__init__()

        self.embeddings = nn.Embedding(108,10)

        self.lstm = nn.LSTM(input_size = 5, hidden_size = 32, num_layers =1, batch_first = True, bidirectional = True)

        self.dropout = nn.Dropout(DROPOUT)
        self.lin1 = nn.Linear(64, 16)
        self.lin2 = nn.Linear(16, T)

    def forward(self, xb, yb, sector, tsne = False):
        
        #x = self.embeddings(sector).unsqueeze(dim=1).repeat(1, W, 1)
        #x = torch.cat((xb, x), dim=2)
        x, y = self.lstm(xb)
        x = torch.cat((y[0][0, :, :], y[0][1, :, :]), dim = 1)

        emb = self.embeddings(sector.long()).squeeze()
        #print(x.shape, emb.shape)
        #x = torch.cat((x, emb), dim=1)
        x = x.squeeze(dim=0)
        x = self.lin1(x)

        if tsne == True:
            return x 

        x = F.relu(x)
        #x = self.dropout(x)
        x = self.lin2(x)
        return x

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, W, T, D_MODEL):
        super(NLinear, self).__init__()
        self.seq_len = W
        self.pred_len = T
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = D_MODEL
        self.individual = False
        self.embeddings = nn.Embedding(108,10)
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len//2)
            self.Lin2 = nn.Linear(D_MODEL, 1)
            self.Linear2 = nn.Linear(self.pred_len//2, self.pred_len)

    def forward(self, x, y, sec):
        # x: [Batch, Input length, Channel]
        #seq_last = x[:,-1:,:].detach()
        #x = x - seq_last
        

        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            #x = F.dropout(x, 0.1)
            #emb = self.embeddings(sec).unsqueeze(dim=2)
            
            x = self.Lin2(x)
            #x = torch.cat((x,emb), dim=1)
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            x = F.relu(x)
            x = self.Linear2(x.permute(0,2,1)).permute(0,2,1)
            
        #x = x + seq_last
        return x # [Batch, Output length, Channel]