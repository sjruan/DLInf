import torch.nn as nn
import torch
import os
import numpy as np
import json
import argparse
from datetime import datetime
import time
from delivery_location_discovery.dataset import get_data_loader
from delivery_location_discovery.models import LocMatcher, LocMatcherPN


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
    parser.add_argument('--sample_interval', type=int, default=100,
                        help='interval between validation')
    parser.add_argument('--harved_epoch', type=int, default=5,
                        help='halved at every x interval')

    parser.add_argument('--model_name', type=str, help='model name')
    # geocoding / addr
    parser.add_argument('--lc_type', type=str, default='geocoding', help='location commonality type')
    # hc / grid
    parser.add_argument('--clus_method', type=str, default='hc', help='clustering method')
    parser.add_argument('--clus_dist', type=int, help='clustering distance')
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
    model_name = opt.model_name
    inverted_instance_type = opt.lc_type
    clus_dist_thresh = opt.clus_dist

    clus_method_suffix = '_www10' if opt.clus_method == 'grid' else ''
    lc_type_suffix = '_addr_inverted' if opt.lc_type == 'addr' else ''

    # load data
    selection_learning_sample_path = os.path.join(base_dir, 'result_DLInf',
                                                  'learning_samples_selection_S{}-R{}_BD{}_D{}_LQ{}-{}_seed{}{}{}/'.format(
                                                      behavior_min_trips, behavior_min_rate, batch_delivery_times,
                                                      clus_dist_thresh, min_delvs, min_conf, seed, clus_method_suffix,
                                                      lc_type_suffix))
    train_dl, in_features = get_data_loader(selection_learning_sample_path, opt.batch_size, model_name, 'train')
    val_dl, _ = get_data_loader(selection_learning_sample_path, 16, model_name, 'val')

    # create model
    if model_name in ['LocMatcher', 'LocMatcher-nD', 'LocMatcher-nT', 'LocMatcher-nP', 'LocMatcher-nLC', 'LocMatcher-nTC']:
        hidden_units, nb_heads, nb_layers = 32, 2, 3
        params = (hidden_units, nb_heads, nb_layers)
        model = LocMatcher(hidden_units, nb_heads=nb_heads, nb_layers=nb_layers, loc_inp_dim=in_features).to(
            device)
    elif model_name == 'LocMatcher-nA':
        hidden_units, nb_heads, nb_layers = 32, 2, 3
        params = (hidden_units, nb_heads, nb_layers)
        model = LocMatcher(hidden_units, nb_heads=nb_heads, nb_layers=nb_layers, loc_inp_dim=in_features,
                                 use_addr=False).to(device)
    elif model_name == 'LocMatcherPN':
        hidden_units = 16
        params = hidden_units
        model = LocMatcherPN(hidden_units, loc_inp_dim=in_features).to(device)
    else:
        raise Exception('unknown model')

    # save path
    save_path = os.path.join(base_dir, 'result_DLInf',
                             'saved_model/loc_selector_S{}_R{}_BD{}_D{}_LQ{}-{}_seed{}{}{}/{}/H{}_{}/'.format(
                                 behavior_min_trips, behavior_min_rate, batch_delivery_times, clus_dist_thresh,
                                 min_delvs, min_conf, seed, clus_method_suffix, lc_type_suffix,
                                 model_name, params, time.strftime("%Y%m%d%H%M%S")))
    os.makedirs(save_path, exist_ok=True)

    # train
    criterion = nn.NLLLoss().to(device)
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.b1, opt.b2))

    iter = 0
    losses = [float('inf')]
    for epoch in range(opt.n_epochs):
        train_loss = 0
        ep_time = datetime.now()
        for i, (X_addr, X_addr_type, X_loc_dense_seq, X_time_dist_seq, data_length, Y) in enumerate(train_dl):
            model.train()
            optimizer.zero_grad()

            # pred which is better
            pred = model(X_addr, X_addr_type, X_loc_dense_seq, X_time_dist_seq, data_length)
            loss = 0.0
            for k in range(len(pred)):
                loss += criterion(pred[k].unsqueeze(dim=0), Y[k].unsqueeze(dim=0))

            loss.backward()
            optimizer.step()

            print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch, opt.n_epochs, i, len(train_dl), loss.item()))

            # counting training loss
            train_loss += loss.item() * len(Y)

            iter += 1
            # validation phase
            if iter % opt.sample_interval == 0:
                model.eval()
                valid_time = datetime.now()
                pred_all = []
                pred_all_np = []
                y_all = []
                val_loss = 0.0
                nb_samples = 0
                nb_correct = 0
                for j, (X_addr, X_addr_type, X_loc_dense_seq, X_time_dist_seq, data_length, Y) in enumerate(val_dl):
                    pred = model(X_addr, X_addr_type, X_loc_dense_seq, X_time_dist_seq, data_length)
                    nb_samples += len(pred)
                    for k in range(len(pred)):
                        val_loss += criterion(pred[k].unsqueeze(dim=0), Y[k].unsqueeze(dim=0)).item()
                        pred_idx = torch.argmax(pred[k])
                        if Y[k] == pred_idx:
                            nb_correct += 1
                acc = nb_correct / nb_samples
                if val_loss < np.min(losses):
                    print("iter\t{}\tloss\t{:.6f}\ttime\t{}\n".format(iter, loss.item(), datetime.now() - valid_time))
                    torch.save(model.state_dict(),
                               '{}/final_model.pt'.format(save_path))
                    f = open('{}/results.txt'.format(save_path), 'a')
                    f.write(
                        "epoch\t{}\titer\t{}\tACC\t{:.6f}\n".format(epoch, iter, acc))
                    f.close()
                losses.append(val_loss)

        # halve the learning rate
        if epoch % opt.harved_epoch == 0 and epoch != 0:
            lr /= 2
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
            f = open('{}/results.txt'.format(save_path), 'a')
            f.write("half the learning rate!\n")
            f.close()

        print('=================time cost: {}==================='.format(datetime.now() - ep_time))
