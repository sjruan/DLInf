from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
import numpy as np
import os
import pickle
import time


def get_classifier(name):
    class_weight = {1: 80, 0: 20}
    if name == 'RF':
        return RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, class_weight=class_weight)
    elif name == 'GBDT':
        return GradientBoostingClassifier(random_state=0, n_estimators=n_estimators)
    elif name == 'SVM':
        return SGDClassifier(class_weight=class_weight)
    else:
        raise NotImplemented('not implemented')


def get_data(data_path, mode, name):
    X = np.load(os.path.join(data_path, mode, 'X.npy'))
    quant_X = X[:, 1:]
    if name not in ['RF', 'GBDT', 'XGB']:
        scaler = preprocessing.StandardScaler()
        train_quant_X = np.load(os.path.join(data_path, 'train', 'X.npy'))[:, 1:]
        scaler.fit(train_quant_X)
        quant_X = scaler.transform(quant_X)
    cate_X = np.zeros((X.shape[0], 21))
    cate_X[range(len(X)), X[:, 0].astype(int)] = 1
    X = np.concatenate([cate_X, quant_X], axis=1)
    Y = np.load(os.path.join(data_path, mode, 'Y.npy'))
    return X, Y


if __name__ == '__main__':
    base_dir = '../data/'
    clus_dist_thresh = 50
    behavior_min_trips = 10
    behavior_min_rate = 0.8
    train_rate = 0.8
    val_rate = 0.1
    seed = 2017
    batch_delivery_times = 5
    min_delvs = 2
    min_conf = 0.51
    geocoding_tolerance = 1000
    inverted_instance_type = 'geocoding'
    global_result_path = base_dir + 'result_DLInf/'
    classification_learning_sample_path = os.path.join(global_result_path,
                                                       'learning_samples_S{}-R{}_BD{}_D{}_LQ{}-{}_seed{}/'.format(
                                                           behavior_min_trips, behavior_min_rate, batch_delivery_times,
                                                           clus_dist_thresh, min_delvs, min_conf, seed))
    model_name = 'RF'
    max_depth = 10
    n_estimators = 400
    params = (max_depth, n_estimators)

    # model_name = 'GBDT'
    # n_estimators = 150
    # params = n_estimators

    # model_name = 'SVM'
    # params = ''

    save_path = os.path.join(global_result_path,
                             'saved_model/loc_selector_S{}_R{}_BD{}_D{}_LQ{}-{}_seed{}/{}/H{}_{}/'.format(
                                 behavior_min_trips, behavior_min_rate, batch_delivery_times,
                                 clus_dist_thresh,
                                 min_delvs, min_conf, seed, model_name, params,
                                 time.strftime("%Y%m%d%H%M%S")))
    os.makedirs(save_path, exist_ok=True)
    clf = get_classifier(model_name)
    train_X, train_Y = get_data(classification_learning_sample_path, 'train', model_name)
    sample_weight = None
    if model_name == 'GBDT':
        sample_weight = np.zeros(len(train_Y))
        sample_weight[train_Y == 0] = 20
        sample_weight[train_Y == 1] = 80
    clf.fit(train_X, train_Y, sample_weight=sample_weight)
    with open('{}/final_model.pt'.format(save_path), 'wb') as f:
        pickle.dump(clf, f)
