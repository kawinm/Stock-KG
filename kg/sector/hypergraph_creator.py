import pandas as pd
import os

df      = pd.read_excel("Selecting-A-GICS-Industry-Code.xlsx", sheet_name="Companies in GICS and SICs ")
desc_df = pd.read_excel("Selecting-A-GICS-Industry-Code.xlsx", sheet_name="GICS Descriptions")

def create_hypergraph(index):
    INDEX = index

    directory = "../../data/" + INDEX + "/"

    ticker_list = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            ticker, name = filename.split("-")
            ticker_list[ticker] = [name]

    sector_map = {}
    for index, row in df.iterrows():
        if row[1] in ticker_list:
            for j in range(6, 10):
                if row[j] not in sector_map:
                    sector_map[row[j]] = [row[1]]
                else:
                    sector_map[row[j]] = sector_map[row[j]] + [row[1]]

    sector_desc = {}
    for index, row in desc_df.iterrows():
        if type(row[0]) == int:
            sector_desc[row[0]] = row[1]
        if type(row[2]) == int:
            sector_desc[row[2]] = row[3]
        if type(row[4]) == int:
            sector_desc[row[4]] = row[5]
        if type(row[6]) == int:
            sector_desc[row[6]] = row[7]

    sector_map = dict(sorted(sector_map.items()))

    sector_hypergraph = open("sector_hypergraph_"+INDEX+".txt", "w")
    sector_hypergraph.write("sector\tdescription\tnode_list\n")

    for sector, node_list in sector_map.items():
        node_list_str = "\t".join([str(x) for x in node_list])
        sector_hypergraph.write("{0}\t{1}\t{2}\n".format(sector, sector_desc[int(sector)], node_list_str))
    sector_hypergraph.close()

    return True

create_hypergraph("sp500")