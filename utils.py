import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

import torch 
from torch_geometric.data import Data

# ------- H1: DATASET UTILS -------------

# ------- H2: Preprocess Data -------------
def window_returns(df, W, T, sectors_to_id, company_to_id, ticker, sector):
    """
        Returns the window of input data + targets are returns
    """
    SMOOTH = 0.00001
    list_df = [(
                    (df['Open'][i+1:i+W+1].values  - df['Open'][i:i+W].values)  / df['Open'][i:i+W].values, 
                    (df['High'][i+1:i+W+1].values  - df['High'][i:i+W].values)  / df['High'][i:i+W].values,
                    (df['Low'][i+1:i+W+1].values   - df['Low'][i:i+W].values)   / df['Low'][i:i+W].values,
                    (df['Close'][i+1:i+W+1].values - df['Close'][i:i+W].values) / df['Close'][i:i+W].values,
                    (df['Volume'][i+1:i+W+1].values - df['Volume'][i:i+W].values) / (df['Volume'][i:i+W]+SMOOTH).values,
                    sectors_to_id[sector], 
                    company_to_id[ticker],  
                    df[i+1:i+W+1]['Date'], 
                    (df['Close'][i+W+1:i+W+T+1].values - df['Close'][i+W:i+W+T].values) / df['Close'][i+W:i+W+T].values, 
                    df['Close'][i+W:i+W+T+1]
                ) 
                for i in range(df.shape[0]-W-T)
            ]

    return list_df

def window_scale_divison(df, W, T, sectors_to_id, company_to_id, ticker, sector):
    """
        Returns the window of input and target values scaled by dividing with the
        last value of the previous window.

        Problems: With large W and T
    """
    SMOOTH = 0.00001
    list_df = [(
                    (df['Open'][i+1:i+W+1] / df['Open'][i:i+1].values).values, 
                    (df['High'][i+1:i+W+1] / df['High'][i:i+1].values).values,           
                    (df['Low'][i+1:i+W+1] / df['Low'][i:i+1].values).values, 
                    (df['Close'][i+1:i+W+1] / df['Close'][i:i+1].values).values,           
                    (df['Volume'][i+1:i+W+1] / (df['Volume'][i:i+1].values+SMOOTH)).values, 
                    sectors_to_id[sector], 
                    company_to_id[ticker],  
                    df[i+1:i+W+1]['Date'], 
                    (df['Close'][i+W+1:i+W+T+1] / df['Close'][i:i+1].values).values, 
                    df['Close'][i:i+1]
                ) 
                for i in range(df.shape[0]-W-T)
            ]
    return list_df

def window_scale_window_max(df, W, T, sectors_to_id, company_to_id, ticker, sector):
    """
        Returns the window of input and target values scaled by dividing with the
        maximum value for that feature in the window.
    """
    list_df = [((df['Open'][i+1:i+W+1] / df['Open'][i+1:i+W+1].max()).values, 
                        (df['High'][i+1:i+W+1] / df['High'][i+1:i+W+1].max()).values,           
                        (df['Low'][i+1:i+W+1] / df['Low'][i+1:i+W+1].max()).values, 
                        (df['Close'][i+1:i+W+1] / df['Close'][i+1:i+W+1].max()).values,           
                        (df['Volume'][i+1:i+W+1] / (df['Volume'][i+1:i+W+1].max()+0.00001)).values, 
                        sectors_to_id[sector], 
                        company_to_id[ticker],  
                        df[i+1:i+W+1]['Date'],
                        (df['Close'][i+W+1:i+W+T+1] / df['Close'][i+1:i+W+1].max()).values, 
                        df['Close'][i+1:i+W+1].max()) 
                        for i in range(df.shape[0]-W-T)]
    return list_df

# ------- H2: Create Data -------------
def create_dataset(INDEX, W, T, problem='value', fast = False):

    directory = "data/" + INDEX + "/"

    sectors_to_id, company_to_id = {}, {}
    sector_id, company_id        = 0, 0

    dataset = [[], [], []]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            print(filename)
            sector, name, ticker, start, end = filename.split("-")
            df = pd.read_csv(f)

            if df.shape[0] <= 2600:    # 13 years
                print("Skipping file: Less Training-Testing samples [{0} samples]".format(df.shape[0]))
                continue


            if sector not in sectors_to_id:
                sectors_to_id[sector] = sector_id
                sector_id += 1
            if ticker not in company_to_id:
                company_to_id[ticker] = company_id
                company_id += 1

            stock_split_dates = df['Stock Splits'].notnull()
            for i in range(len(df)):
                if stock_split_dates[i]:
                    split = str(df.iloc[i]['Stock Splits'])
                    split = int(split.split(":")[0]) / int(split.split(":")[1])
                    df.iloc[i, df.columns.get_loc('Stock Splits')] = split

            df = df.fillna(0)

            if problem == 'returns':
                list_df = window_returns(df, W, T, sectors_to_id, company_to_id, ticker, sector)
            else:
                list_df = window_scale_divison(df, W, T, sectors_to_id, company_to_id, ticker, sector)
            
            # ----- Data Split by Percentage ------
            #train, test_df = train_test_split(list_df, test_size=0.2, shuffle=False)
            #train_df, val_df  = train_test_split(train, test_size=0.2, shuffle=False)

            # ----- Data Split by Days ---------
            test_df = list_df[-400:]
            val_df  = list_df[-600:-400]
            train_df = list_df[-2600:-600]

            dataset[0].extend(train_df)
            dataset[1].extend(val_df)
            dataset[2].extend(test_df)

            print("Company: ", name, len(train_df), len(val_df), len(test_df))
            #print(df.max())

            if fast:
                break

    return dataset, sectors_to_id, company_to_id

def create_batch_dataset(INDEX, W, T, problem='value', fast = False):

    directory = "data/" + INDEX + "/"

    sectors_to_id, company_to_id = {}, {}
    sector_id, company_id        = 0, 0

    dataset = [[], [], []]
    df_map = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            print(filename)
            sector, name, ticker, start, end = filename.split("-")
            df = pd.read_csv(f)

            if df.shape[0] <= 2600:    # 13 years
                print("Skipping file: Less Training-Testing samples [{0} samples]".format(df.shape[0]))
                continue

            if sector not in sectors_to_id:
                sectors_to_id[sector] = sector_id
                sector_id += 1
            if ticker not in company_to_id:
                company_to_id[ticker] = company_id
                company_id += 1
            
            
            """ "Stock Split removed"
            stock_split_dates = df['Stock Splits'].notnull()
            for i in range(len(df)):
                if stock_split_dates[i]:
                    split = str(df.iloc[i]['Stock Splits'])
                    split = int(split.split(":")[0]) / int(split.split(":")[1])
                    df.iloc[i, df.columns.get_loc('Stock Splits')] = split

            df = df.fillna(0) """

            if df.shape[0] > 2600:
                df = df.iloc[-2600:]
            
            if problem == 'returns':
                list_df = window_returns(df, W, T, sectors_to_id, company_to_id, ticker, sector)
            else:
                list_df = window_scale_divison(df, W, T, sectors_to_id, company_to_id, ticker, sector)

            df_map[company_to_id[ticker]] = list_df 

    for i in range(len(list_df)):
        cur_data = []
        for j in range(company_id):
            cur_data.append(df_map[j][i])

        if i < 2000:
            dataset[0].append(cur_data) 
        elif i < 2200:
            dataset[1].append(cur_data)
        else:
            dataset[2].append(cur_data)

    sector_graph = open("kg/sector/sector_hypergraph_nasdaq100.txt", "r").readlines()

    # Unidirectional Homogeneous Graph
    sector_map = {}
    graph_dataset = []
    for lines in sector_graph[1:]:
        lines = lines[:-1]
        tickers = lines.split("\t")[2:]
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                if tickers[i] not in company_to_id or tickers[j] not in company_to_id:
                    continue
                if tickers[i] + "-" + tickers[j] not in sector_map:
                    graph_dataset.append([company_to_id[tickers[i]], company_to_id[tickers[j]]])
                    graph_dataset.append([company_to_id[tickers[j]], company_to_id[tickers[i]]])
                    sector_map[tickers[i] + "-" + tickers[j]] = 1
                    sector_map[tickers[j] + "-" + tickers[i]] = 1

    edge_index = torch.Tensor(graph_dataset).t().contiguous()
    x = torch.randn(len(company_to_id.items()), 8)

    graph_data = Data(x=x, edge_index=edge_index.long())

    hyperedge_index, hyper_x = get_sector_hypergraph(company_to_id)
    hyper_data = {
        'hyperedge_index': hyperedge_index,
        'x': hyper_x
    }
    
    return dataset, sectors_to_id, company_to_id, graph_data, hyper_data


def load_dataset(save_path):
    with open(save_path, 'rb') as handle:
        b = pickle.load(handle)

    dataset = [[], [], []]
    dataset[0] = b['train']
    dataset[1] = b['val']
    dataset[2] = b['test']
    sectors_to_id = b['sectors']
    company_to_id = b['company']

    return dataset, sectors_to_id, company_to_id

def save_dataset(save_path, dataset, sectors_to_id, company_to_id):
    print("--- Saving Dataset ---")
    save_data = {'train': dataset[0], 'val': dataset[1], 'test': dataset[2], \
                    'sectors': sectors_to_id, 'company': company_to_id}

    with open(save_path, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_or_create_dataset(INDEX, W, T, save_path, problem, fast):
    if fast == True:
        print("--- Creating Dataset ---")
        dataset, sectors_to_id, company_to_id = create_dataset(INDEX, W, T, problem, fast)
    elif os.path.isfile(save_path):
        print("--- File exists: Loading Dataset ---")
        dataset, sectors_to_id, company_to_id = load_dataset(save_path)
    else:
        print("--- Creating Dataset ---")
        dataset, sectors_to_id, company_to_id = create_dataset(INDEX, W, T, problem, fast)
        save_dataset(save_path, dataset, sectors_to_id, company_to_id)

    return dataset, sectors_to_id, company_to_id

def load_dataset_graph(save_path):
    with open(save_path, 'rb') as handle:
        b = pickle.load(handle)

    dataset = [[], [], []]
    dataset[0] = b['train']
    dataset[1] = b['val']
    dataset[2] = b['test']
    sectors_to_id = b['sectors']
    company_to_id = b['company']
    graph = b['graph']
    hyper_data = b['hyper_graph']

    return dataset, sectors_to_id, company_to_id, graph, hyper_data

def save_dataset_graph(save_path, dataset, sectors_to_id, company_to_id, graph, hyper_data):
    print("--- Saving Dataset ---")
    save_data = {'train': dataset[0], 'val': dataset[1], 'test': dataset[2], \
                    'sectors': sectors_to_id, 'company': company_to_id, 'graph': graph,
                    'hyper_graph': hyper_data}

    with open(save_path, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_or_create_dataset_graph(INDEX, W, T, save_path, problem, fast):
    if fast == True:
        print("--- Creating Dataset ---")
        dataset, sectors_to_id, company_to_id, graph, hyper_data = create_batch_dataset(INDEX, W, T, problem, fast)
    elif os.path.isfile(save_path):
        print("--- File exists: Loading Dataset ---")
        dataset, sectors_to_id, company_to_id, graph, hyper_data = load_dataset_graph(save_path)
    else:
        print("--- Creating Dataset ---")
        dataset, sectors_to_id, company_to_id, graph, hyper_data = create_batch_dataset(INDEX, W, T, problem, fast)
        save_dataset_graph(save_path, dataset, sectors_to_id, company_to_id, graph, hyper_data)

    return dataset, sectors_to_id, company_to_id, graph, hyper_data
    
    
# METRICS UTILS

def mean_absolute_percentage_error(y_true, y_pred): 
    return (((y_true - y_pred) / y_true).abs()).mean() * 100
    #return ((y_true - y_pred).abs()).mean() * 100

def root_mean_square_error(y_true, y_pred, scale = None): 
    if scale == None:
        return ((y_true - y_pred) ** 2).mean() ** (1/2)
    else:
        return (((y_true - y_pred)*scale) ** 2).mean() ** (1/2)

def mean_square_error(y_true, y_pred, scale = None):
    if scale == None:
        return ((y_true - y_pred) ** 2).mean() 
    else:
        return (((y_true - y_pred)*scale.unsqueeze(dim=1)) ** 2).mean() 

# 3. KG Loader

def get_sector_hypergraph(company_to_id):
    # HyperGraph
    sector_graph = open("kg/sector/sector_hypergraph_nasdaq100.txt", "r").readlines()

    n = len(company_to_id.items())
    hyperedge_index = [[], []]
    edge = 0
    for lines in sector_graph[1:]:
        lines = lines[:-1]
        tickers = lines.split("\t")[2:]
        for i in range(len(tickers)):
            if tickers[i] not in company_to_id:
                continue
            hyperedge_index[0].extend([company_to_id[tickers[i]]])
            hyperedge_index[1].extend([edge])
        edge += 1

    hyperedge_index = torch.Tensor(hyperedge_index).long()
    hyper_x = torch.randn(len(company_to_id.items()), 8)

    print("Number of edges in hypergraph: ", hyperedge_index.shape)
    print(hyper_x.shape)

    return hyperedge_index, hyper_x

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--index', type=str, default="AAPL")
    #parser.add_argument('--window', type=int, default=10)
    #parser.add_argument('--test_size', type=float, default=0.2)

    #create_dataset("nasdaq100", 50, 5)
    d, s, c, g, h = create_batch_dataset("nasdaq100", 50, 5)
    print(len(d[0]), len(d[0][0]), d[0][0][0])

