import torch.nn as nn
import torch
import os
import numpy as np
import json
import argparse
from datetime import datetime
import time
import sys
from delivery_location_discovery.dataset import get_class_data_loader
from delivery_location_discovery.models import MLP
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='adam: decay of second order momentum of gradient')
    parser.add_argument('--sample_interval', type=int, default=10000,
                        help='interval between validation')
    parser.add_argument('--harved_epoch', type=int, default=5,
                        help='halved at every x interval')

    opt = parser.parse_args()

    base_dir = '../data/'
    with open(os.path.join(base_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    dist_thresh, min_stay_time = params['preprocessing']['dist_thresh'], params['preprocessing']['min_stay_time']
    behavior_min_trips = 10
    behavior_min_rate = 0.8
    train_rate = 0.8
    val_rate = 0.1
    seed = 2017
    batch_delivery_times = 5
    gt_cluster_dist_thresh = 30
    min_delvs = 2
    min_conf = 0.51
    geocoding_tolerance = 1000
    clus_dist_thresh = 50
    model_name = 'MLP'

    # load data
    selection_learning_sample_path = os.path.join(base_dir, 'result_DLInf',
                                                  'learning_samples_S{}-R{}_BD{}_D{}_LQ{}-{}_seed{}/'.format(
                                                      behavior_min_trips, behavior_min_rate, batch_delivery_times,
                                                      clus_dist_thresh, min_delvs, min_conf, seed))
    train_dl, in_features = get_class_data_loader(selection_learning_sample_path, opt.batch_size, 'train')
    val_dl, _ = get_class_data_loader(selection_learning_sample_path, 16, 'val')

    hidden_units = 16
    model = MLP(hidden_units, inp_dim=in_features).to(device)

    # save path
    save_path = os.path.join(base_dir, 'result_DLInf',
                             'saved_model/loc_selector_S{}_R{}_BD{}_D{}_LQ{}-{}_seed{}/{}/H{}_{}/'.format(
                                 behavior_min_trips, behavior_min_rate, batch_delivery_times, clus_dist_thresh,
                                 min_delvs, min_conf, seed, model_name, hidden_units,
                                 time.strftime("%Y%m%d%H%M%S")))
    os.makedirs(save_path, exist_ok=True)

    # train
    criterion = nn.CrossEntropyLoss().to(device)
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.b1, opt.b2))

    iter = 0
    f1s = [float('-inf')]
    for epoch in range(opt.n_epochs):
        train_loss = 0
        ep_time = datetime.now()
        for i, (x_addr_type, x_dense, x_time_dist, y) in enumerate(train_dl):
            model.train()
            optimizer.zero_grad()

            pred = model(x_addr_type, x_dense, x_time_dist)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (
            epoch, opt.n_epochs, i, len(train_dl), loss.item()))

            # counting training loss
            train_loss += loss.item() * len(y)

            iter += 1
            # validation phase
            if iter % opt.sample_interval == 0:
                model.eval()
                valid_time = datetime.now()
                pred_all = []
                pred_all_np = []
                y_all = []
                for j, (x_addr_type, x_dense, x_time_dist, y) in enumerate(val_dl):
                    pred = model(x_addr_type, x_dense, x_time_dist)
                    pred_all.append(pred)
                    y_all.append(y)
                    pred_all_np.append(torch.max(pred, 1)[1].cpu().detach().numpy())
                pred_all = torch.cat(pred_all, 0)
                y_all = torch.cat(y_all, 0)
                val_loss = criterion(pred_all, y_all)

                y_all = y_all.cpu().detach().numpy()
                pred_all = np.concatenate(pred_all_np)
                precision = precision_score(y_all, pred_all)
                recall = recall_score(y_all, pred_all)
                f1 = f1_score(y_all, pred_all)
                acc = accuracy_score(y_all, pred_all)
                if f1 > np.max(f1s):
                    print(
                        "iter\t{}\tloss\t{:.6f}\time\t{}\n".format(iter, loss.item(), datetime.now() - valid_time))
                    torch.save(model.state_dict(), '{}/final_model.pt'.format(save_path))
                    f = open('{}/results.txt'.format(save_path), 'a')
                    f.write(
                        "epoch\t{}\titer\t{}\tACC\t{:.6f}\tPreci\t{:.6f}\tRecall\t{:.6f}\tF1\t{:.6f}\n".format(
                            epoch, iter, acc, precision, recall, f1))

                    f.close()
                # losses.append(val_loss.item())
                f1s.append(f1)

        # halve the learning rate
        if epoch % opt.harved_epoch == 0 and epoch != 0:
            lr /= 2
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
            f = open('{}/results.txt'.format(save_path), 'a')
            f.write("half the learning rate!\n")
            f.close()

        print('=================time cost: {}==================='.format(datetime.now() - ep_time))
