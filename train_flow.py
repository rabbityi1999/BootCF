import logging
import time
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
import random
import tqdm
import shutil
from utils import EarlyStopMonitor
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

config = {
    "NYTaxi": {
        "data_path": "output/bootstrap/results/NYTaxi_.pkl",
        "matrix_path": "data/NYTaxi/od_matrix.npy",
        "day_cycle": 48,
        "train_day": 3,
        "val_day": 1,
        "test_day": 1,
        "day_start": 0,
        "day_end": 48,
        "n_nodes": 63,
        "batch_size": 8
    }
}

### Argument and global variables
parser = argparse.ArgumentParser('Crowd flow training')
parser.add_argument('--data', type=str, help='Dataset name (eg. NYTaxi or BJMetro)', default='NYTaxi')
parser.add_argument('--emd_path', type=str, help='Embedding Path', default='')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--suffix', type=str, default='', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='Path of the best model')
parser.add_argument('--n_epoch', type=int, default=1000, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=3, help='Number of network layers')
parser.add_argument('--interdim', type=int, default=10, help='interdim for gconv')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--patience', type=int, default=1000, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cpu", help='Idx for the gpu to use: cpu, cuda:0, etc.')
parser.add_argument('--task', type=str, default="i", help='what task to predict: i, o or od')
parser.add_argument('--loss', type=str, default="mse", help='Loss function')


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, a):
        out = [x]
        x1 = self.nconv(x, a)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, a)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class PredictionLayer(nn.Module):
    def __init__(self, embedding_dim, n_nodes):
        super(PredictionLayer, self).__init__()
        self.n_nodes = n_nodes
        self.w = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2), 1),
        )

    def forward(self, embeddings):
        B, N, F = embeddings.shape
        A = embeddings.repeat([1, 1, self.n_nodes]).reshape([B, self.n_nodes * self.n_nodes, -1])
        C = embeddings.repeat([1, self.n_nodes, 1])
        return self.w(torch.cat([A, C], dim=2)).reshape([B, self.n_nodes, self.n_nodes])


class PredictionLayer_Flow(nn.Module):
    def __init__(self, embedding_dim, n_nodes, layers=3, interdim=16, device='cuda:0'):
        super(PredictionLayer_Flow, self).__init__()
        self.layers = layers
        self.idx = torch.arange(n_nodes).to(device)
        residual_channels = embedding_dim
        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim,
                      out_channels=residual_channels,
                      kernel_size=(1, 1)),
        )

        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.nodevec1 = nn.Parameter(torch.randn(n_nodes, interdim).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(interdim, n_nodes).to(device), requires_grad=True).to(device)

        for i in range(self.layers):
            self.skip_convs.append(nn.Conv2d(in_channels=residual_channels,
                                             out_channels=residual_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            self.gconv.append(gcn(residual_channels, residual_channels, 0.3, support_len=1))

        self.w = nn.Sequential(
            nn.Conv2d(in_channels=residual_channels,
                      out_channels=int(residual_channels / 2),
                      kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=int(residual_channels / 2),
                      out_channels=int(residual_channels / 4),
                      kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=int(residual_channels / 4),
                      out_channels=int(residual_channels / 8),
                      kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=int(residual_channels / 8),
                      out_channels=1,
                      kernel_size=(1, 1)),
        )

    def forward(self, embeddings):
        B, N, _ = embeddings.shape
        embeddings = embeddings.transpose(1, 2).contiguous().unsqueeze(-1)
        x = embeddings
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        skip = 0
        for i in range(self.layers):
            residual = x
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            x = self.gconv[i](x, adp)
            x = x + residual
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.w(x))
        return x.reshape([B, N])


def get_od_data(config):
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    train_day = config["train_day"]
    val_day = config["val_day"]
    test_day = config["test_day"]
    embeddings = pickle.load(open(config["data_path"], "rb"))
    embeddings = embeddings["embeddings"]
    full_steps, n_nodes, embedding_dim = embeddings.shape
    od_matrix = np.load(config["matrix_path"]).reshape([-1, day_cycle, n_nodes, n_nodes])[:, day_start:day_end]
    od_matrix = od_matrix.reshape([-1, n_nodes, n_nodes])
    if day_start == 0:
        offset = 1
        od_matrix = od_matrix[1:]
    else:
        offset = 0
    full_set = np.arange(full_steps + offset)
    train_set = full_set[offset:train_day * (day_end - day_start)] - offset
    val_set = full_set[train_day * (day_end - day_start):(train_day + val_day) * (day_end - day_start)] - offset
    test_set = full_set[(train_day + val_day) * (day_end - day_start):] - offset
    data_loaders = {"train": DataLoader(train_set, shuffle=True, batch_size=config["batch_size"], drop_last=False),
                    "val": DataLoader(val_set, shuffle=False, batch_size=config["batch_size"], drop_last=False),
                    "test": DataLoader(test_set, shuffle=False, batch_size=config["batch_size"], drop_last=False)}

    return n_nodes, embedding_dim, od_matrix, embeddings, data_loaders, (train_set, val_set, test_set)


def calculate_metrics(stacked_prediction, stacked_label):
    stacked_prediction[stacked_prediction < 0] = 0
    reshaped_prediction = stacked_prediction.reshape(-1)
    reshaped_label = stacked_label.reshape(-1)
    mse = mean_squared_error(reshaped_prediction, reshaped_label)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(reshaped_prediction, reshaped_label)
    pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
    smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (
            np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))
    return (mse, rmse, mae, pcc, smape)


def predict_flow(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    NUM_EPOCH = args.n_epoch
    device = args.device
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr

    Path(f"./output/{args.task}/saved_models/").mkdir(parents=True, exist_ok=True)
    Path(f"./output/{args.task}/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f"./output/{args.task}/saved_models/{args.data}_{args.suffix}.pth"
    get_checkpoint_path = lambda epoch: f"./output/{args.task}/saved_checkpoints/{args.data}_{args.suffix}_{epoch}.pth"
    results_path = f"./output/{args.task}/results/{args.data}_{args.suffix}.pkl"
    Path(f"./output/{args.task}/results/").mkdir(parents=True, exist_ok=True)

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(f"./output/{args.task}/log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"./output/{args.task}/log/{str(time.time())}_{args.data}_{args.suffix}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    if args.emd_path != "":
        print("assigned embedding")
        config[DATA]["data_path"] = args.emd_path

    ### Extract data for training, validation and testing
    n_nodes, embedding_dim, od_matrix, embeddings, data_loaders, data_sets = get_od_data(config[DATA])
    if args.task == "od":
        model = PredictionLayer(embedding_dim=embedding_dim, n_nodes=n_nodes)
    elif args.task in ["i", "o"]:
        model = PredictionLayer_Flow(embedding_dim=embedding_dim, n_nodes=n_nodes, layers=args.n_layer,
                                     interdim=args.interdim, device=args.device)

    if args.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss == "mae":
        criterion = torch.nn.L1Loss()

    model = model.to(device)

    val_mses = []
    epoch_times = []
    total_epoch_times = []
    train_mses = []
    if args.best == "":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
        ifstop = False
        for epoch in range(NUM_EPOCH):
            print("================================Epoch: %d================================" % epoch)
            start_epoch = time.time()
            logger.info('start {} epoch'.format(epoch))
            for phase in ["train", "val"]:
                label, prediction = [], []
                if phase == "train":
                    model = model.train()
                else:
                    model = model.eval()
                batch_range = tqdm.tqdm(data_loaders[phase])
                for ind in batch_range:
                    batch_embeddings = torch.Tensor(embeddings[ind]).to(device)
                    od_matrix_real = od_matrix[ind]
                    if args.task == "od":
                        real_data = od_matrix_real
                    elif args.task == "i":
                        real_data = np.sum(od_matrix_real, axis=1)
                    elif args.task == "o":
                        real_data = np.sum(od_matrix_real, axis=2)
                    else:
                        raise NotImplementedError
                    predicted_data = model(batch_embeddings)
                    if phase == "train":
                        optimizer.zero_grad()
                        loss = criterion(predicted_data, torch.Tensor(real_data).to(device))
                        loss.backward()
                        optimizer.step()
                        batch_range.set_description(f"train_loss: {loss.item()};")
                    label.append(real_data)
                    prediction.append(predicted_data.cpu().detach().numpy())
                concated_label = np.concatenate(label)
                concated_prediction = np.concatenate(prediction)
                metrics = calculate_metrics(concated_prediction, concated_label)
                logger.info(
                    'Epoch {} {} metric: mse, rmse, mae, pcc, smape, {}, {}, {}, {}, {}'.format(epoch, phase, *metrics))
                if phase == "train":
                    train_mses.append(metrics[0])
                elif phase == "val":
                    val_mses.append(metrics[0])
                    # Early stopping
                    ifstop, ifimprove = early_stopper.early_stop_check(metrics[0], epoch)
                    if ifstop:
                        logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                    else:
                        logger.info('No improvement over {} epochs'.format(early_stopper.num_round))
                        torch.save({"statedict": model.state_dict()}, get_checkpoint_path(epoch))
            if ifstop:
                break
            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)
            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))

        # Save temporary results
        pickle.dump({
            "val_mses": val_mses,
            "train_losses": train_mses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('Saving model')
        shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
        logger.info('Model saved')
        best_model_param = torch.load(get_checkpoint_path(early_stopper.best_epoch))
    else:
        best_model_param = torch.load(args.best)

    # load model parameters, memories from best epoch on val dataset
    model.load_state_dict(best_model_param["statedict"])
    # Test
    print("================================Test================================")

    model = model.eval()
    batch_range = tqdm.tqdm(data_loaders["test"])
    label, prediction = [], []
    for ind in batch_range:
        batch_embeddings = torch.Tensor(embeddings[ind]).to(device)
        predicted_data = model(batch_embeddings)
        od_matrix_real = od_matrix[ind]
        if args.task == "od":
            real_data = od_matrix_real
        elif args.task == "i":
            real_data = np.sum(od_matrix_real, axis=1)
        elif args.task == "o":
            real_data = np.sum(od_matrix_real, axis=2)
        else:
            raise NotImplementedError
        label.append(real_data)
        prediction.append(predicted_data.cpu().detach().numpy())
    concated_label = np.concatenate(label)
    concated_prediction = np.concatenate(prediction)
    test_metrics = calculate_metrics(concated_prediction, concated_label)

    logger.info(
        'Test statistics:-- mse: {}, rmse: {}, mae: {}, pcc: {}, smape:{}'.format(*test_metrics))
    # Save results for this run
    pickle.dump({
        "val_mses": val_mses,
        "test_mse": test_metrics[0],
        "test_rmse": test_metrics[1],
        "test_mae": test_metrics[2],
        "test_pcc": test_metrics[3],
        "test_smape": test_metrics[4],
        "epoch_times": epoch_times,
        "train_losses": train_mses,
        "total_epoch_times": total_epoch_times,
        "pred": concated_prediction,
        "real": concated_label
    }, open(results_path, "wb"))


if __name__ == '__main__':
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    predict_flow(args)
