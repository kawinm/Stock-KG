import csv
import os, sys
import math
from queue import PriorityQueue

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

mpl.rcParams['figure.dpi']= 300

from sklearn.metrics import accuracy_score
from torchinfo import summary
from tqdm import tqdm
from utils import (load_or_create_dataset, mean_absolute_percentage_error,
                   load_or_create_dataset_graph,
                   mean_square_error, root_mean_square_error)

from models.models import Transformer, BILSTM, NLinear

from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool

import wandb

GPU = 1
LR = 0.0001
BS = 128
W = 20
T = 5
LOG = False
D_MODEL = 5
N_HEAD  = 5
DROPOUT = 0.2
D_FF    = 64
ENC_LAYERS = 1
DEC_LAYERS = 1
MAX_EPOCH = 100
USE_POS_ENCODING = False
USE_GRAPH = True
HYPER_GRAPH = True
PREDICTION_PROBLEM = 'value'

#MODEL =  "moving_average"  #
MODEL = "ours"
#MODEL = "ours"
#MODEL = "lstm"
#MODEL = "nlinear"

FAST = False
if FAST == True:
    LOG = False

if LOG:
    wandb_config = {
        'Prediction Problem': PREDICTION_PROBLEM,
        'Learning Rate':LR,
        'Batch Size': BS,
        'Window Size': W,
        'Prediction Steps': T,
        'Model': MODEL,
        'Optimizer': 'ADAM',
        'Preprocessing': ['Window Divide Min'],
        'KG': ['NO Company'],
        'Transformer Params': [D_MODEL, N_HEAD, DROPOUT, D_FF, ENC_LAYERS, DEC_LAYERS, USE_POS_ENCODING]
    }

    wandb.init(project="KG-Stock-Graph"+str(T), config=wandb_config)

INDEX = "sp500" 

#save_path = "data/pickle/"+INDEX+"/data-return-W"+str(W)+"-T"+str(T)+".pkl"
save_path = "data/pickle/"+INDEX+"/graph_data-W"+str(W)+"-T"+str(T)+"_"+str(PREDICTION_PROBLEM)+".pkl"

# Copart - 2.23 rmse, 2.1 - NLinear
dataset, sectors_to_id, company_to_id, graph, hyper_data = load_or_create_dataset_graph(INDEX=INDEX, W=W, T=T, save_path=save_path, problem=PREDICTION_PROBLEM, fast=FAST)

if torch.cuda.is_available():
    device = torch.device("cuda:"+str(GPU))
else:
    device = torch.device("cpu")


if not HYPER_GRAPH:
    graph_nodes_batch = torch.zeros(graph.x.shape[0]).to(device)
    graph = graph.to(device)
    graph_data = {
        'x': graph.x,
        'edge_list': graph.edge_index,
        'batch': graph_nodes_batch
    }
else:
    x, hyperedge_index = hyper_data['x'].to(device), hyper_data['hyperedge_index'].to(device)

    print("Graph details: ", x.shape, hyperedge_index.shape)
    graph_data = {
        'x': x,
        'hyperedge_index': hyperedge_index
    }

#print("Range of scaled data: min: {0} max: {1} mean: {2}".format(min([min(x[10]) for x in dataset[0]]), max([max(x[10]) for x in dataset[0]]), np.array([x[10] for x in dataset[0]]).mean()))
#print("Range of scaled data [Test]: min: {0} max: {1} mean: {2}".format(min([min(x[10]) for x in dataset[2]]), max([max(x[10]) for x in dataset[2]]), np.array([x[10] for x in dataset[2]]).mean()))
#print("Number of companies: {0} and sectors: {1}".format(len(company_to_id.values()), len(sectors_to_id.values())))


# ----------- Batching the data -----------
def collate_fn(instn):
    instn = instn[0]

    # df: shape: Companies x W+1 x 5 (5 is the number of features)
    df = torch.Tensor(np.array([x[0] for x in instn])).unsqueeze(dim=2)

    for i in range(1, 5):
        df1 = torch.Tensor(np.array([x[i] for x in instn])).unsqueeze(dim=2)
        df = torch.cat((df, df1), dim=2)

    sector = torch.Tensor(np.array([x[5] for x in instn])).long()
    company = torch.Tensor(np.array([x[6] for x in instn])).long()
    
    # Shape: Companies x TimeSteps
    target = torch.Tensor(np.array([x[8] for x in instn]))

    # Shape: Companies x 1
    scale = torch.Tensor(np.array([x[9] for x in instn]))

    if PREDICTION_PROBLEM == 'returns':
        movement = target >= 0
    elif PREDICTION_PROBLEM == 'value':
        window_last_close = df[:, -1, 3].unsqueeze(dim=1)
        movement = target >= window_last_close

    return (df, sector, company, target, scale, movement.int())

train_loader    = DataLoader(dataset[0], 1, shuffle=True, collate_fn=collate_fn, num_workers=1)
val_loader      = DataLoader(dataset[1], 1, shuffle=True, collate_fn=collate_fn)
test_loader     = DataLoader(dataset[2], 1, shuffle=False, collate_fn=collate_fn)

for i in train_loader:
    print(i)
    break


# Moving Average Baseline 
loss_fn = mean_square_error

def moving_average_predictor(loader, metric_fn, loss_fn = mean_square_error, disp = "Training"):
    ma_loss = 0
    last_loss, mape_last_loss = torch.zeros(T), torch.zeros(T)
    move_loss = 0
    mape_loss = 0
    for xb, sector, company, yb, scale, move_target in tqdm(loader):
        close_val = xb[:, :, 3]
        for i in range(T):
            ma = torch.mean(close_val, dim=1).unsqueeze(dim=1)
            close_val = torch.cat((close_val, ma), dim = 1)
        
        ma_loss += loss_fn(close_val[:, W:], yb)
        mape_loss += metric_fn(close_val[:, W:], yb)

        for i in range(T):
            last_loss[i] += loss_fn(close_val[:, W+i], yb[:,i]).item()
        for i in range(T):
            mape_last_loss[i] += metric_fn(close_val[:, W+i], yb[:,i]).item()

        y_hat = close_val[:, W:].squeeze()
        move = y_hat.squeeze() - xb[:, -1, 3].unsqueeze(dim=1)
        move_pred = move >= 0
        move_loss += (move_pred.int() == move_target).float().mean()

     
    ma_loss /= len(loader)
    last_loss /= len(loader)
    mape_last_loss /= len(loader)
    mape_loss /= len(loader)
    move_loss /= len(loader)
    print("[" + disp + "] Moving Average Loss: MSE:", ma_loss.item(), " RMSE:", (ma_loss ** (1/2)).item(), last_loss ** (1/2))
    
    if disp == 'Training':
        prefix = ''
    elif disp == 'Validation':
        prefix = 'Val '
    else:
        prefix = 'Test '

    log = {prefix+'MSE': ma_loss, prefix+'RMSE': ma_loss ** (1/2), prefix+"MOVEMENT ACC": move_loss, prefix+"MAPE": mape_loss}

    log[prefix+"RMSE_@Step"+str(T)] = last_loss[T-1] ** (1/2)
    log[prefix+"MAPE_@Step"+str(T)] = mape_last_loss[T-1] 
    
    if LOG:
        wandb.log(log)

if MODEL == "moving_average":
    moving_average_predictor(train_loader, mean_absolute_percentage_error, mean_square_error, "Training")
    moving_average_predictor(val_loader, mean_absolute_percentage_error, mean_square_error, "Validation")
    moving_average_predictor(test_loader, mean_absolute_percentage_error, mean_square_error, "Testing")

    sys.exit(0)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.predictor = Transformer(W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING, USE_GRAPH)
        self.gnn_encoder = Sequential('x, edge_index, batch', [
                                #(Dropout(p=0.5), 'x -> x'),
                                (GCNConv(8, 64), 'x, edge_index -> x1'),
                                ReLU(inplace=True),
                                (GCNConv(64, 150), 'x1, edge_index -> x2'),
                                ReLU(inplace=True),
                                #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                                #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                                #(global_mean_pool, 'x, batch -> x'),
                                Linear(150, 5),
                            ])
    
    def forward(self, x, edge_index, batch):
        x = self.gnn_encoder(x, edge_index, batch)
        x = self.predictor(x, edge_index)
        
        return x

model  = Transformer(W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING, USE_GRAPH, HYPER_GRAPH)
#model = NLinear()
#model = BILSTM(W, T, DROPOUT)

#model.load_state_dict(torch.load("models/oursW"+str(W)+"T"+str(T)+".pt"), strict=False)

#summary(model, input_size=[(BS, W, 5), (BS, T), (BS, 1)], device=device)
print(model)
model.to(device)

opt_c = torch.optim.Adam(model.parameters(), lr = LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
loss_fn = F.mse_loss
move_loss_fn = accuracy_score
#model.load_state_dict(torch.load("oursW50T20.pt"))

# ----------- Main Training Loop -----------
def predict(loader, desc):
    epoch_loss, mape, move_loss, rmse_returns_loss = 0, 0, 0, 0
    last_loss, mape_last_loss = torch.zeros(T), torch.zeros(T)
    mini, maxi = float("infinity"), 0

    for xb, sector, company, yb, scale, move_target in tqdm(loader):
        xb = xb.to(device) - 1
        sector = company.to(device) + 1
        yb = yb.to(device) - 1
        scale = scale.to(device)
        move_target = move_target.to(device)

        y_hat = model(xb, yb, sector, graph_data).squeeze()

        yb_scaled = yb #+ 0.5

        #print(loss_fn_c(y_hat.squeeze(), yb), F.binary_cross_entropy(torch.sigmoid(movement).flatten(), move_target.flatten().float()))
        #loss = loss_fn(y_hat, yb_scaled) #F.binary_cross_entropy(move_label.float(), target_label.float())
        loss = F.l1_loss(y_hat, yb_scaled)
        #loss = F.binary_cross_entropy(torch.sigmoid(movement), move_target.float())
        rmse_returns_loss += float(loss)

        y_hat = y_hat + 1
        yb = yb + 1

        if model.training:
            loss.backward()
            opt_c.step()
            opt_c.zero_grad()

        if PREDICTION_PROBLEM == 'returns':
            y_hat = y_hat #- 0.5
            scaled_prices = torch.zeros_like(scale[:, 1:])
            init = scale[:, 0]
            for returns in range(y_hat.shape[1]-1):
                init = (y_hat[:, returns] * init) + init
                scaled_prices[:, returns] += init
            scale = scale[:, 1:]

            mape += mean_absolute_percentage_error(scale, scaled_prices)

            move_pred = y_hat >= 0
            move_loss += (move_pred.int() == move_target).float().mean()

            rmse_loss = loss_fn(scaled_prices, scale) 

            for i in range(T):
                last_loss[i] += loss_fn(scale[:,i], scaled_prices[:,i]).item()
            for i in range(T):
                mape_last_loss[i] += mean_absolute_percentage_error(scale[:,i].squeeze(), scaled_prices[:,i]).item()
        
        elif PREDICTION_PROBLEM == "value":
            mape += mean_absolute_percentage_error(y_hat.squeeze(), yb)

            window_last_close = xb[:, -1, 3].unsqueeze(dim=1)
            move_pred = torch.where(y_hat >= window_last_close, 1, 0)
            move_loss += (move_pred.int() == move_target).float().mean()

            rmse_loss = root_mean_square_error(yb, y_hat, scale) 

            for i in range(T):
                last_loss[i] += root_mean_square_error(yb[:, i], y_hat[:, i], scale).item()
            for i in range(T):
                mape_last_loss[i] += mean_absolute_percentage_error(yb[:, i], y_hat[:, i]).item()
        
        mini = min(mini, y_hat.min().item())
        maxi = max(maxi, y_hat.max().item())  
        
        epoch_loss += float(rmse_loss)

    epoch_loss /= len(loader)
    rmse_returns_loss /= len(loader)
    last_loss  /= len(loader)
    mape_last_loss  /= len(loader)
    move_loss  /= len(loader)
    mape /= len(loader)

    print("[{0}] Movement Prediction Accuracy: {1}".format(desc, move_loss.item()))
    print("[{0}] MAPE: {1}".format(desc, mape.item()))
    print("[{0}] Range of predictions min: {1} max: {2}".format(desc, mini, maxi))
    print("[{0}] Epoch: {1} MSE: {2} RMSE: {3} Returns RMSE: {4}".format(desc, ep+1, epoch_loss, epoch_loss ** (1/2), rmse_returns_loss ** (1/2)))
    print("[{0}] RMSE LAST DAY: : {1} MAPE LAST DAY: {2}".format(desc, (last_loss ** (1/2))[-1].item(), mape_last_loss[-1].item()))

    log = {'MSE': epoch_loss, 'RMSE': epoch_loss ** (1/2), "MOVEMENT ACC": move_loss, "MAPE": mape}

    log["RMSE_@Step"+str(T)] = last_loss[T-1] ** (1/2)
    log["MAPE_@Step"+str(T)] = mape_last_loss[T-1]
    
    if LOG:
        wandb.log(log)

    return epoch_loss

def plot(loader):
    ytrue, ypred = [], []

    rtrue, rpred = [], []
    for xb, sector, company, yb, scale, move_target in tqdm(loader):
        xb = xb.to(device)[:, :, :5] #+ 0.5
        sector = company.to(device) + 1
        yb = yb.to(device)
        scale = scale.to(device)
        move_target = move_target.to(device)

        y_hat = model(xb, yb, sector, graph_data).squeeze()
        y_hat = y_hat #- 0.5

        loss = loss_fn(yb, y_hat) #+ F.binary_cross_entropy(torch.sigmoid(movement).flatten(), move_target.flatten().float())

        scaled_prices = torch.zeros_like(scale[:, 1:])
        init = scale[:, 0]
        for returns in range(y_hat.shape[1]-1):
            init = (y_hat[:, returns] * init) + init
            scaled_prices[:, returns] += init
        scale = scale[:, 1:]

        ypred.extend(scaled_prices[:, 0].detach().cpu().numpy())
        ytrue.extend(scale[:, 0].detach().cpu().numpy())

        rpred.extend(y_hat[:, 0].detach().cpu().numpy())
        rtrue.extend(yb[:, 0].detach().cpu().numpy())

    plt.plot([x for x in range(100)], ytrue[:100], c='b')
    plt.plot([x for x in range(100)], ypred[:100], c='r')
    plt.savefig("plot.jpg")
    plt.close()

    plt.plot([x for x in range(100)], rtrue[:100], c='b')
    plt.plot([x for x in range(100)], rpred[:100], c='r')
    plt.savefig("plot2.jpg")
    plt.close()

best_test_acc = 0
prev_val_loss = float("infinity")
for ep in range(MAX_EPOCH):
    print("Epoch: " + str(ep+1))
    
    model.train()
    train_epoch_loss = predict(train_loader, "TRAINING")

    model.eval()
    with torch.no_grad():
        val_epoch_loss  = predict(val_loader, "VALIDATION")

    #plot(val_loader)

    if ep > MAX_EPOCH//2 and prev_val_loss > val_epoch_loss:
        print("Saving Model")
        torch.save(model.state_dict(), "best_model.pt")
        prev_val_loss = val_epoch_loss

model.load_state_dict(torch.load("best_model.pt"))

model.eval()
with torch.no_grad():
    test_epoch_loss = predict(test_loader, "TESTING")

if LOG:
    wandb.save('model.py')