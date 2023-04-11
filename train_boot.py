import logging
import time
import sys
import argparse
import pickle
import shutil
import random
from tqdm import trange, tqdm
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from BootCF import MLP_Predictor
from BootCF import BootCF
from utils import EarlyStopMonitor, np_metric, get_data

config = {
    "NYTaxi": {
        "data_path": "data/NYTaxi/NYTaxi.npy",
        "matrix_path": "data/NYTaxi/od_matrix.npy",
        "point_path": "data/NYTaxi/back_points.npy",
        "feature_path": "data/NYTaxi/feature.npy",
        "input_len": 1800,
        "output_len": 1800,
        "day_cycle": 86400,
        "train_day": 3,
        "val_day": 1,
        "test_day": 1,
        "day_start": -1,
        "day_end": 86401,
        "sample": 1,
        "n_nodes": 63
    }
}

parser = argparse.ArgumentParser('BootCF training')
parser.add_argument('--data', type=str, help='Dataset name', default='NYTaxi')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--suffix', type=str, default='', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='Path of the best model')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cuda:0", help='Idx for the gpu to use: cpu, cuda:0, etc.')
parser.add_argument('--model', type=str, default="BootCF", help='Which model to use')
parser.add_argument('--message_dim', type=int, default=128, help='Dimensions of the messages')
parser.add_argument('--pred_len', type=int, default=1, help='number of timestamps to predict')
parser.add_argument('--memory_dim', type=int, default=128, help='Dimensions of the memory for ''each node')
parser.add_argument('--lamb', type=float, default=1.0, help='Lamb of different time scales')
parser.add_argument('--ratio', type=float, default=0.4, help='augment ratio')
parser.add_argument('--beta', type=float, default=0.9, help='ema ratio')


def main(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    NUM_EPOCH = args.n_epoch
    device = args.device
    DATA = args.data
    LEARNING_RATE = args.lr
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim

    input_len = config[DATA]["input_len"]
    output_len = config[DATA]["output_len"]
    day_cycle = config[DATA]["day_cycle"]
    day_start = config[DATA]["day_start"]
    day_end = config[DATA]["day_end"]

    Path("./output/bootstrap/saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./output/bootstrap/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'./output/bootstrap/saved_models/{args.data}_{args.suffix}.pth'
    get_checkpoint_path = lambda epoch: f'./output/bootstrap/saved_checkpoints/{args.data}_{args.suffix}_{epoch}.pth'
    results_path = f"./output/bootstrap/results/{args.data}_{args.suffix}.pkl"
    Path("./output/bootstrap/results/").mkdir(parents=True, exist_ok=True)

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("./output/bootstrap/log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"./output/bootstrap/log/{str(time.time())}_{args.data}_{args.suffix}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    ### Extract data for training, validation and testing
    n_nodes, node_features, full_data, val_time, test_time, all_time, od_matrix, back_points = get_data(config[DATA])

    online_encoder = BootCF(device=device, n_nodes=n_nodes, node_features=node_features,
                            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                            output=output_len, lamb=args.lamb).to(device)
    online_predictor = MLP_Predictor(MEMORY_DIM * 2, MEMORY_DIM * 2, MEMORY_DIM * 2).to(device)
    target_encoder = BootCF(device=device, n_nodes=n_nodes, node_features=node_features,
                            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                            output=output_len, lamb=args.lamb).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    val_losses = []
    total_epoch_times = []
    train_losses = []
    optimizer = torch.optim.Adam(list(online_encoder.parameters()) + list(online_predictor.parameters()),
                                 lr=LEARNING_RATE)
    beta = args.beta
    if args.best == "":
        early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)
        num_batch = val_time // input_len
        for epoch in range(NUM_EPOCH):
            print("================================Epoch: %d================================" % epoch)
            start_epoch = time.time()
            logger.info('start {} epoch'.format(epoch))
            m_loss = []

            online_encoder.init_memory()
            target_encoder.init_memory()
            online_encoder = online_encoder.train()
            batch_range = trange(num_batch)
            embeddings = []
            for j in batch_range:
                mm = 1 - (1 - beta) * (np.cos(np.pi * (j + num_batch * epoch) / (NUM_EPOCH * num_batch)) + 1) / 2.0
                # Training
                now_time = (j + 1) * input_len
                if now_time % day_cycle < day_start or now_time % day_cycle >= day_end:
                    continue
                head, tail = back_points[j], back_points[j + 1]
                source_batch, destination_batch = full_data.sources[head:tail], full_data.destinations[head:tail]
                timestamps_batch = torch.Tensor(full_data.timestamps[head:tail]).to(device)
                time_diffs_batch = torch.Tensor(- full_data.timestamps[head:tail] + now_time).to(device)
                online_embeddings = online_encoder(source_nodes=source_batch, destination_nodes=destination_batch,
                                                   timestamps=timestamps_batch, now_time=now_time,
                                                   time_diffs=time_diffs_batch)
                predicted_embeddings = online_predictor(online_embeddings)
                embeddings.append(online_embeddings.detach().cpu().numpy())

                # Data augmentation for target branch
                add_number = int(len(source_batch) * (1 + args.ratio))
                if (j >= 1):
                    head_now, tail_now = back_points[j - 1], back_points[j + 2]
                else:
                    head_now, tail_now = back_points[j], back_points[j + 2]
                source_batch2, destination_batch2 = full_data.sources[head_now:tail_now], full_data.destinations[
                                                                                          head_now:tail_now]
                timestamps_batch2 = torch.Tensor(full_data.timestamps[head_now: tail_now]).to(device)
                choice_index = torch.randperm(len(source_batch2))[:add_number]
                source_batch2 = source_batch2[choice_index]
                destination_batch2 = destination_batch2[choice_index]
                timestamps_batch2 = timestamps_batch2[choice_index]
                head2, tail2 = head, tail
                time_diffs_batch2 = torch.Tensor(-timestamps_batch2.detach().cpu() + now_time).to(device)

                target_embeddings = target_encoder(source_nodes=source_batch2, destination_nodes=destination_batch2,
                                                   timestamps=timestamps_batch2, now_time=full_data.timestamps[tail2],
                                                   time_diffs=time_diffs_batch2)

                # Updata online encoder parameters by backpropogation
                loss = 1 - cosine_similarity(predicted_embeddings, target_embeddings.detach(), dim=-1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())
                batch_range.set_description(f"train_loss: {m_loss[-1]} ;")

                # Update target encoder parameters by EMA
                for param_q, param_k in zip(online_encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)
                target_encoder.restore_memory(online_encoder.backup_memory())

            # Validation
            stacked_embeddings = np.stack(embeddings)
            print("================================Val================================")
            _, _, val_metrics = eval_flow_prediction(model=online_encoder, data=full_data,
                                                     back_points=back_points,
                                                     st=val_time, eval_st=val_time,
                                                     ed=test_time,
                                                     device=device,
                                                     config=config[DATA],
                                                     od_matrix=od_matrix,
                                                     train_embeddings=stacked_embeddings)
            val_loss = val_metrics[3]
            val_losses.append(val_loss)
            train_losses.append(np.mean(m_loss))
            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            # Save temporary results
            pickle.dump({
                "val_losses": val_losses,
                "train_losses": train_losses,
                "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            logger.info('Epoch mean train loss: {}'.format(np.mean(m_loss)))
            logger.info('Epoch val metrics: {}'.format(val_metrics))
            ifstop, ifimprove = early_stopper.early_stop_check(val_loss, epoch)
            if ifstop:
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                break
            else:
                torch.save(
                    {"statedict": online_encoder.state_dict(), "memory": online_encoder.backup_memory()},
                    get_checkpoint_path(epoch))
        logger.info('Saving BootCF model')
        shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
        logger.info('BootCF model saved')
        best_model_param = torch.load(get_checkpoint_path(early_stopper.best_epoch))
    else:
        best_model_param = torch.load(args.best)

    # load model parameters, memories from best epoch on val dataset
    online_encoder.load_state_dict(best_model_param["statedict"])
    online_encoder.init_memory()
    # Test and generate node representations for the whole dataset
    print("================================Test================================")
    res_embeddings, test_predict, test_metrics = eval_flow_prediction(model=online_encoder, data=full_data,
                                                                      back_points=back_points, st=0,
                                                                      eval_st=test_time,
                                                                      ed=all_time,
                                                                      device=device, config=config[DATA],
                                                                      od_matrix=od_matrix,
                                                                      train_embeddings=None)

    logger.info('Test statistics:-- loss: {}'.format(test_metrics))

    # Save results for this run
    pickle.dump({
        "embeddings": res_embeddings,
        "val_losses": val_losses,
        "test_metrics": test_metrics,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times,
        "test_predict": test_predict
    }, open(results_path, "wb"))


def eval_flow_prediction(model, data, back_points, st, eval_st, ed, device, config, od_matrix, train_embeddings):
    input_len = config["input_len"]
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    embeddings_list = []
    predict_list = []
    with torch.no_grad():
        model = model.eval()
        num_test_batch = (ed - st) // input_len
        for j in tqdm(range(num_test_batch)):
            begin_time = j * input_len + st
            now_time = (j + 1) * input_len + st
            if now_time % day_cycle < day_start or now_time % day_cycle >= day_end:
                continue
            head, tail = back_points[begin_time // input_len], back_points[now_time // input_len]
            source_batch, destination_batch = data.sources[head:tail], data.destinations[head:tail]
            timestamps_batch = torch.Tensor(data.timestamps[head:tail]).to(device)
            time_diffs_batch = torch.Tensor(-data.timestamps[head:tail] + now_time).to(device)
            embedding = model(source_nodes=source_batch, destination_nodes=destination_batch,
                              timestamps=timestamps_batch, now_time=now_time,
                              time_diffs=time_diffs_batch)
            embeddings_list.append(embedding.detach().cpu().numpy())

    stacked_embeddings = np.stack(embeddings_list)  # B N F
    train_time = np.arange(1, eval_st // input_len + 1) * input_len
    train_mask = np.logical_and(train_time % day_cycle >= day_start, train_time % day_cycle < day_end)
    if train_embeddings is None:
        train_embeddings = stacked_embeddings[:np.sum(train_mask)]
        test_embeddings = stacked_embeddings[np.sum(train_mask):]
    else:
        test_embeddings = stacked_embeddings
    train_len, n_nodes, feature_dim = train_embeddings.shape
    raw_train_labels = od_matrix[1: eval_st // input_len + 1]

    train_labels = raw_train_labels[train_mask]

    test_time = np.arange((eval_st - st) // input_len, (ed - st) // input_len) * input_len
    test_mask = np.logical_and(test_time % day_cycle >= day_start, test_time % day_cycle < day_end)
    raw_real = od_matrix[eval_st // input_len + 1:ed // input_len + 1]
    real = raw_real[test_mask]

    model = torch.nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.LeakyReLU(),
        nn.Linear(feature_dim, int(feature_dim / 2)),
        nn.LeakyReLU(),
        nn.Linear(int(feature_dim / 2), 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    train_loader = DataLoader(np.arange(train_embeddings.shape[0]), shuffle=True, batch_size=8, drop_last=False)
    test_loader = DataLoader(np.arange(test_embeddings.shape[0]), shuffle=False, batch_size=8, drop_last=False)
    n_epoch = 10
    model = model.train()
    epoch_range = trange(n_epoch)
    for i in epoch_range:
        m_loss = []
        for ind in train_loader:
            batch_embeddings = torch.Tensor(train_embeddings[ind]).to(device)
            od_matrix_real = train_labels[ind]
            real_data = np.sum(od_matrix_real, axis=2)
            predicted_data = model(batch_embeddings).squeeze()
            optimizer.zero_grad()
            loss = criterion(predicted_data, torch.Tensor(real_data).to(device))
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())
        epoch_range.set_description(f"val_loss as epoch {i}: {np.mean(m_loss)};")
    model = model.eval()
    batch_range = tqdm(test_loader)
    label, prediction = [], []
    for ind in batch_range:
        batch_embeddings = torch.Tensor(test_embeddings[ind]).to(device)
        predicted_data = model(batch_embeddings).squeeze()
        od_matrix_real = real[ind]
        real_data = np.sum(od_matrix_real, axis=2)
        label.append(real_data)
        prediction.append(predicted_data.cpu().detach().numpy())
    concated_label = np.concatenate(label)
    concated_prediction = np.concatenate(prediction)
    val_metrics = np_metric(concated_prediction, concated_label)

    return stacked_embeddings, predict_list, val_metrics


if __name__ == '__main__':
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    main(args)
