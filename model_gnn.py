import csv
import os, sys
import pickle
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
from utils import (mean_absolute_percentage_error,
                   load_or_create_dataset_graph,
                   mean_square_error, root_mean_square_error)

from models.models import Transformer_Ranking, Saturation

from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool

from random import randint
import wandb

GPU = 1
LR = 0.0001
BS = 128
W = 20
T = 250
LOG = False
D_MODEL = 20
N_HEAD  = 5
DROPOUT = 0.2
D_FF    = 64
ENC_LAYERS = 1
DEC_LAYERS = 1
MAX_EPOCH = 25
USE_POS_ENCODING = False
USE_GRAPH = True
HYPER_GRAPH = True
USE_KG = True
PREDICTION_PROBLEM = 'value'
RUN = randint(1, 100000)
PLOT = False

tau_choices = [50, 75, 125, 250]
tau_positions = [1, 5, 20, 50, 75, 100, 125, 250]

MODEL = "ours"

print("Experiment SP500 P25 W=20 Run 1 Without Relation KG")
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
print(INDEX)

save_path = "data/pickle/"+INDEX+"/graph_data-P25-W"+str(W)+"-T"+str(T)+"_"+str(PREDICTION_PROBLEM)+".pkl"


dataset, company_to_id, graph, hyper_data = load_or_create_dataset_graph(INDEX=INDEX, W=W, T=T, save_path=save_path, problem=PREDICTION_PROBLEM, fast=FAST)

num_nodes = len(company_to_id.keys())

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

if USE_KG:
    with open('./kg/profile_and_relationship/wikidata/'+INDEX+'_relations_kg.pkl', 'rb') as f:
        relation_kg = pickle.load(f)['kg']
    head, relation, tail = relation_kg[0], relation_kg[1], relation_kg[2]
    head, relation, tail = head.to(device), relation.to(device), tail.to(device)
    relation_kg = (head, relation, tail)
else:
    relation_kg = None

def rank_loss(prediction, ground_truth):
    all_one = torch.ones(prediction.shape[0], 1, dtype=torch.float32).to(device)
    prediction = prediction.unsqueeze(dim=1)
    ground_truth = ground_truth.unsqueeze(dim=1)
    #print(prediction.shape, ground_truth.shape, base_price.shape)
    return_ratio = prediction - 1
    true_return_ratio = ground_truth - 1

    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),                  # C x C
        all_one @ return_ratio.t()                   # C x C
    )
    gt_pw_dif = torch.sub(
        all_one @ true_return_ratio.t(),
        true_return_ratio @ all_one.t()
    )

    rank_loss = torch.mean(
        F.relu(-1*pre_pw_dif * gt_pw_dif )
    )
   
    return rank_loss 

def evaluate(prediction, ground_truth, K):
    return_ratio = prediction - 1
    true_return_ratio = ground_truth - 1

    #print("True top k: ", torch.topk(true_return_ratio.squeeze(), k=3, dim=0))
    #print("Predicted top k: ", torch.topk(return_ratio.squeeze(), k=3, dim=0))
    
    obtained_return_ratio = true_return_ratio[torch.topk(return_ratio, k=K, dim=0)[1]].mean()

    #return_ratio = -1*return_ratio
    #obtained_return_ratio += true_return_ratio[torch.topk(return_ratio.squeeze(), k=K, dim=0)[1]].mean()
    #obtained_return_ratio /= 2

    target_obtained_return_ratio = torch.topk(true_return_ratio, k=K, dim=0)[0].mean()

    expected_return_ratio = torch.topk(return_ratio.squeeze(), k=K, dim=0)[0].mean()

    random = torch.randint(0, prediction.shape[0]-1, (K,))
    random_return_ratio = true_return_ratio[random].mean()

    a_cat_b, counts = torch.cat([torch.topk(return_ratio.squeeze(), k=K, dim=0)[1], torch.topk(true_return_ratio.squeeze(), k=K, dim=0)[1]]).unique(return_counts=True)
    accuracy = a_cat_b[torch.where(counts.gt(1))].shape[0] / K

    return obtained_return_ratio, target_obtained_return_ratio, expected_return_ratio, random_return_ratio, accuracy


top_k_choice = [1, 3, 5, 10]

# ----------- Main Training Loop -----------
def predict(loader, desc):
    epoch_loss, mape, move_loss, rmse_returns_loss, mae_loss = 0, 0, 0, 0, 0
    mini, maxi = float("infinity"), 0
    rr, true_rr, exp_rr, ran_rr, accuracy = torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device)
    
    #tqdm_loader = tqdm(loader) 
    yb_store, yhat_store = [], []
    for xb, company, yb, scale, move_target in loader:
        xb      = xb.to(device)[:, :, 1]
        yb      = yb.to(device) 
        scale   = scale.to(device)
        move_target = move_target.to(device)

        y_hat, kg_loss, hold_pred = model(xb, yb, graph_data, relation_kg)
        y_hat = F.softmax(y_hat.squeeze(), dim = 0)
        true_return_ratio = yb.squeeze() 

        target = torch.topk(true_return_ratio, k=1, dim=0)[1]
        zeros = torch.zeros_like(y_hat)
        zeros[target] = 1


        # store y_hat and yb for plotting
        

        #print("TRR: ", float(true_return_ratio.min()), float(true_return_ratio.max()))
        #print("Pred: ", float(y_hat.min()), float(y_hat.max()))

        #loss = F.mse_loss(y_hat, true_return_ratio) #+ rank_loss(y_hat, true_return_ratio)
        loss = F.binary_cross_entropy(y_hat, zeros)
        if USE_KG:
            loss += kg_loss.mean()

        if model.training:
            opt_c.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt_c.step()

        yb_store.append(yb.detach().cpu().numpy())
        yhat_store.append(y_hat.detach().cpu().numpy())
        
        for index, k in enumerate(top_k_choice):
            crr, ctrr, cerr, crrr, cacc = evaluate(y_hat, true_return_ratio, k)
            ran_rr[index] += crrr
            true_rr[index] += ctrr
            rr[index] += crr
            exp_rr[index] += cerr
            accuracy[index] += cacc

        mae_loss += F.l1_loss(y_hat, true_return_ratio).item()
        rmse_returns_loss += F.mse_loss(y_hat, true_return_ratio).item()
        mape += mean_absolute_percentage_error(y_hat, true_return_ratio)

        move_pred = torch.where(y_hat >= 1, 1, 0)
        move_loss += (move_pred.int() == move_target).float().mean()

        mini = min(mini, y_hat.min().item())
        maxi = max(maxi, y_hat.max().item())  
        loss = F.binary_cross_entropy(y_hat, zeros)
        epoch_loss += float(loss)

    epoch_loss /= len(loader)
    rmse_returns_loss /= len(loader)
    move_loss  /= len(loader)
    mape /= len(loader)
    rr /= len(loader) 
    true_rr /= len(loader)
    exp_rr /= len(loader)
    ran_rr /= len(loader)
    accuracy /= len(loader)
    mae_loss /= len(loader)

    print("[{0}] Movement Prediction Accuracy: {1}, MAPE: {2}".format(desc, move_loss.item(), mape.item()))
    print("[{0}] Range of predictions min: {1} max: {2}".format(desc, mini, maxi))
    print("[{0}] Epoch: {1} MSE: {2} RMSE: {3} Loss: {4} MAE: {5}".format(desc, ep+1, rmse_returns_loss, rmse_returns_loss ** (1/2), epoch_loss, mae_loss))
    
    for index, k in enumerate(top_k_choice):
        print("[{0}] Top {5} Return Ratio: {1} True Return Ratio: {2} Expected Return Ratio: {3} Random Return Ratio: {4} Accuracy: {6}".format(desc, rr[index], true_rr[index], exp_rr[index], ran_rr[index], k, accuracy[index]))
    
    log = {'MSE': epoch_loss, 'RMSE': epoch_loss ** (1/2), "MOVEMENT ACC": move_loss, "MAPE": mape}
    
    if LOG:
        wandb.log(log)

    if PLOT:
        mpl.rcParams['figure.dpi']= 300
        plt.plot(np.array(yb_store).reshape(-1, num_nodes)[:, 0], c='r')
        plt.plot(np.array(yhat_store).reshape(-1, num_nodes)[:, 0], c='b')
        plt.savefig("plots/saturation/R"+ str(RUN)+"-E"+str(ep)+"-T"+str(tau)+ ".png")
        plt.close()

    return epoch_loss, rr, true_rr, exp_rr, ran_rr, move_loss, mape, accuracy, mae_loss



for tau in tau_choices:
    tau_pos = tau_positions.index(tau)

    print("Tau: ", tau, "Tau Position: ", tau_pos)

    # ----------- Batching the data -----------
    def collate_fn(instn):
        instn = instn[0]
        # df: shape: Companies x W+1 x 5 (5 is the number of features)
        #df = torch.Tensor(np.array([x[0] for x in instn])).unsqueeze(dim=2)
        df = torch.Tensor(np.array([x[1] for x in instn])).unsqueeze(dim=2) - torch.Tensor(np.array([x[2] for x in instn])).unsqueeze(dim=2)
        for i in range(3, 5):
            df1 = torch.Tensor(np.array([x[i] for x in instn])).unsqueeze(dim=2)
            df = torch.cat((df, df1), dim=2)
        
        company = torch.Tensor(np.array([x[5] for x in instn])).long()
        
        # Shape: Companies x 1
        target = torch.Tensor(np.array([x[7][tau_pos] for x in instn]))

        # Shape: Companies x 1
        scale = torch.Tensor(np.array([x[8] for x in instn]))

        movement = target >= 1

        return (df, company, target, scale, movement.int())


    start_time = 0
    test_mean_rr, test_mean_trr, test_mean_err, test_mean_rrr = torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device)
    test_mean_move, test_mean_mape, test_mean_mae = 0, 0, 0

    test_mean_acc = torch.zeros(4).to(device)
    print(len(dataset))
    for phase in range(1, 24):
        print("Phase: ", phase)
        train_loader    = DataLoader(dataset[start_time:start_time+250], 1, shuffle=True, collate_fn=collate_fn, num_workers=1)
        val_loader      = DataLoader(dataset[start_time+250:start_time+300], 1, shuffle=True, collate_fn=collate_fn)
        test_loader     = DataLoader(dataset[start_time+300:start_time+400], 1, shuffle=False, collate_fn=collate_fn)

        start_time += 100

        model  = Transformer_Ranking(W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING, USE_GRAPH, HYPER_GRAPH, USE_KG, num_nodes)
        #print(model)
        model.to(device)

        opt_c = torch.optim.Adam(model.parameters(), lr = 6e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        prev_val_loss = float("infinity")
        for ep in range(MAX_EPOCH):
            print("Epoch: " + str(ep+1))
            
            model.train()
            train_epoch_loss, rr, trr, err, rrr, move, mape, accuracy, mae = predict(train_loader, "TRAINING")

            model.eval()
            with torch.no_grad():
                val_epoch_loss, rr, trr, err, rrr, move, mape, accuracy, mae  = predict(val_loader, "VALIDATION")

            #plot(val_loader)

            if (ep > MAX_EPOCH//2 or ep > 10) and prev_val_loss > val_epoch_loss:
                print("Saving Model")
                torch.save(model.state_dict(), "models/saved_models/best_model_"+INDEX+str(W)+"_"+str(T)+"_"+str(RUN)+".pt")
                prev_val_loss = val_epoch_loss

        model.load_state_dict(torch.load("models/saved_models/best_model_"+INDEX+str(W)+"_"+str(T)+"_"+str(RUN)+".pt"))

        model.eval()
        with torch.no_grad():
            test_epoch_loss, rr, trr, err, rrr, move, mape, accuracy, mae = predict(test_loader, "TESTING")
            test_mean_rr += rr
            test_mean_trr += trr
            test_mean_err += err
            test_mean_rrr += rrr
            test_mean_acc += accuracy
            test_mean_move += float(move)
            test_mean_mape += float(mape)
            test_mean_mae += float(mae)
            for index, k in enumerate(top_k_choice):
                print("[Mean - {0}] Top {5} Return Ratio: {1} True Return Ratio: {2} Expected Return Ratio: {3} Random Return Ratio: {4} Accuracy: {6}".format("TESTING", test_mean_rr[index]/phase, test_mean_trr[index]/phase, test_mean_err[index]/phase, test_mean_rrr[index]/phase, k, test_mean_acc[index]/phase))
            print("[Mean - {0}] Movement Accuracy: {1} Mean MAPE: {2} Mean MAE: {3}".format("TESTING", test_mean_move/phase, test_mean_mape/phase, test_mean_mae/phase))
        if LOG:
            wandb.save('model.py')

    phase = 23
    print("Tau: ", tau)
    for index, k in enumerate(top_k_choice):
        print("[Mean - {0}] Top {5} {1} {2} {3} {4} {6}".format("TESTING", test_mean_rr[index]/phase, test_mean_trr[index]/phase, test_mean_err[index]/phase, test_mean_rrr[index]/phase, k, test_mean_acc[index]/phase))
    print("[Mean - {0}] {1} {2} {3}".format("TESTING", test_mean_move/phase, test_mean_mape/phase, test_mean_mae/phase))



