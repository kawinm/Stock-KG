def plot(loader):
    ytrue, ypred = [], []

    rtrue, rpred = [], []
    for xb, company, yb, scale, move_target in tqdm(loader):
        xb = xb.to(device)[:, :, :5] #+ 0.5
        yb = yb.to(device)
        scale = scale.to(device)
        move_target = move_target.to(device)

        y_hat = model(xb, yb, graph_data).squeeze()
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

def top_k_plot(prediction, ground_truth, base_price, i):
    #mpl.rcParams['figure.dpi']= 300
    prediction = prediction.unsqueeze(dim=1)
    ground_truth = ground_truth.unsqueeze(dim=1)
    #print(prediction.shape, ground_truth.shape, base_price.shape)
    return_ratio = (prediction - base_price) / base_price
    true_return_ratio = (ground_truth - base_price) / base_price

    true = torch.topk(true_return_ratio.squeeze(), k=5, dim=0)
    pred = torch.topk(return_ratio.squeeze(), k=5, dim=0)

    fig, axs = plt.subplots(3, figsize=(14, 10))
    fig.tight_layout()
    plt.subplots_adjust(bottom=2, top=3)
    axs[0].stem([i for i in range(5)], true[0].tolist(), 'ro')
    label = []
    for i in true[1].tolist():
        label.append(inv_comp_map[i])
    axs[0].set(xticklabels=label, xticks=[i for i in range(5)], title="Max possible RR")

    axs[1].stem([i for i in range(5)], pred[0].tolist(), 'ro')
    label = []
    for i in pred[1].tolist():
        label.append(inv_comp_map[i])
    axs[1].set(xticklabels=label, xticks=[i for i in range(5)], title="Expected RR")

    axs[2].stem([i for i in range(5)], true_return_ratio[pred[1]].tolist(), 'ro')
    label = []
    for i in pred[1].tolist():
        label.append(inv_comp_map[i])
    axs[2].set(xticklabels=label, xticks=[i for i in range(5)], title="Obtained RR")
    plt.savefig("plots/nasdaq100/top_10/"+str(i)+".png")
    #print("True top k: ", torch.topk(true_return_ratio.squeeze(), k=3, dim=0))
    #print("Predicted top k: ", torch.topk(return_ratio.squeeze(), k=3, dim=0))
