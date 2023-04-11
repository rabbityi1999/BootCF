import numpy as np


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=False, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val, epoch):
        self.epoch_count = epoch
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        return self.num_round >= self.max_round, (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance


def np_metric(pred, real):
    mae = np.mean(np.abs(pred - real))
    rmse = np.sqrt(np.mean((pred - real) ** 2))
    pcc = np.corrcoef(pred.reshape(-1), real.reshape(-1))[0][1]
    mask = real != 0
    mape = np.mean(np.abs(real[mask] - pred[mask]) / real[mask])
    return mae, mape, rmse, pcc


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Data:
    def __init__(self, sources, destinations, timestamps, n_nodes):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.n_interactions = len(sources)
        self.unique_nodes = set(list(range(n_nodes)))
        self.n_unique_nodes = n_nodes


def get_data(config):
    n_nodes = config["n_nodes"]
    whole_data = np.load(config["data_path"])
    od_matrix = np.load(config["matrix_path"])
    back_points = np.load(config["point_path"])
    all_time = (config["train_day"] + config["val_day"] + config["test_day"]) * config["day_cycle"] - config[
        "input_len"]
    val_time, test_time = (config["train_day"]) * config["day_cycle"] - config["input_len"], (
            config["train_day"] + config["val_day"]) * config["day_cycle"] - config["input_len"]
    sources = whole_data[::config["sample"], 0].astype("int")
    destinations = whole_data[::config["sample"], 1].astype("int")
    timestamps = whole_data[::config["sample"], 2].astype("int")
    node_features = np.diag(np.ones(n_nodes))
    full_data = Data(sources, destinations, timestamps, n_nodes)
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))

    return n_nodes, node_features, full_data, val_time, test_time, all_time, od_matrix, back_points
