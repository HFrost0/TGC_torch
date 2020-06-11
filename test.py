import os
import random
from time import time
import numpy as np
import torch
from torch_geometric.nn import GCNConv, SAGEConv, NNConv, GATConv
from torch_geometric.data import Data
import torch.nn.functional as F
from model_self import get_loss
from evaluator import evaluate
import json


def get_edge_data(relation_file):
    relation_encoding = np.load(relation_file)
    edge_index = []
    edge_feature = []
    for i in range(relation_encoding.shape[0]):
        for j in range(relation_encoding.shape[1]):
            if np.sum(relation_encoding[i][j]):
                edge_index.append([i, j])
                edge_feature.append(relation_encoding[i][j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_feature = torch.tensor(edge_feature, dtype=torch.float)
    return edge_index, edge_feature


def load_temp_data():
    return np.load('./temp_data/eod_data.npy'), \
           np.load('./temp_data/mask_data.npy'), \
           np.load('./temp_data/gt_data.npy'), \
           np.load('./temp_data/price_data.npy')


def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = parameters['seq']
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))


steps = 1
valid_index = 756
test_index = 1008
market_name = 'NASDAQ'
parameters = {'seq': 16, 'unit': 64, 'lr': 0.001, 'alpha': 0.1}
relation_name = 'wikidata'
data_path = 'D:/datasets/TGC_data/2013-01-01'
tickers_fname = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
market_name = 'NASDAQ'
rname_tail = {'sector_industry': '_industry_relation.npy', 'wikidata': '_wiki_relation.npy'}

tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname), dtype=str, delimiter='\t', skip_header=False)
batch_size = len(tickers)
print('#tickers selected:', len(tickers))

# load data
edge_index, edge_feature = get_edge_data(
    os.path.join(data_path, '..', 'relation', relation_name, market_name + rname_tail[relation_name]))
edge_index.to('cuda')
edge_feature.to('cuda')
eod_data, mask_data, gt_data, price_data = load_temp_data()
trade_dates = mask_data.shape[1]


class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv = GATConv(64, 64)
        self.lstm_cell = torch.nn.LSTMCell(5, 64)
        self.h0 = torch.zeros(1026, 64).to('cuda')
        self.c0 = torch.zeros(1026, 64).to('cuda')
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.LeakyReLU())

    def forward(self, data):
        inputs, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        hx = self.h0
        cx = self.c0
        for x in inputs.split(1, 1):
            x.squeeze_()
            hx, cx = self.lstm_cell(x, (hx, cx))
        x = self.conv(hx, edge_index)
        prediction = self.fc(x)
        return prediction


model = GATNet()
model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), parameters['lr'])

best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)


def validate(start_index, end_index):
    """
    get loss on validate/test set
    """
    with torch.no_grad():
        cur_valid_pred = np.zeros([len(tickers), end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([len(tickers), end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([len(tickers), end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - parameters['seq'] - steps + 1, end_index - parameters['seq'] - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.tensor(x).to('cuda'),
                get_batch(cur_offset)
            )
            data = Data(x=data_batch, edge_index=edge_index, edge_attr=edge_feature).to('cuda')
            prediction = model(data)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     batch_size, parameters['alpha'])
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf


records = {
    'train_loss': [], 'train_reg_loss': [], 'train_rank_loss': [],
    'val_loss': [], 'val_reg_loss': [], 'val_rank_loss': [], 'val_perf': [],
    'test_loss': [], 'test_reg_loss': [], 'test_rank_loss': [], 'test_perf': [],
}
for epoch in range(30):
    t1 = time()
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    for j in range(valid_index - parameters['seq'] - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.tensor(x).to('cuda'),
            get_batch(batch_offsets[j])
        )
        data = Data(x=data_batch, edge_index=edge_index, edge_attr=edge_feature).to('cuda')
        optimizer.zero_grad()
        prediction = model(data)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            batch_size, parameters['alpha'])
        # update model
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    # train loss
    # loss = reg_loss(mse) + alpha*rank_loss
    tra_loss = tra_loss / (valid_index - parameters['seq'] - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - parameters['seq'] - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - parameters['seq'] - steps + 1)
    print('\n\nTrain : loss:{} reg_loss:{} rank_loss:{}'.format(tra_loss, tra_reg_loss, tra_rank_loss))
    records['train_loss'].append(tra_loss)
    records['train_reg_loss'].append(tra_reg_loss)
    records['train_rank_loss'].append(tra_rank_loss)

    # show performance on valid set
    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print('Valid : loss:{} reg_loss:{} rank_loss:{}'.format(val_loss, val_reg_loss, val_rank_loss))
    print('\t Valid performance:', val_perf)
    records['val_loss'].append(val_loss)
    records['val_reg_loss'].append(val_reg_loss)
    records['val_rank_loss'].append(val_rank_loss)
    records['val_perf'].append(val_perf)

    # show performance on valid set
    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{} reg_loss:{} rank_loss:{}'.format(test_loss, test_reg_loss, test_rank_loss))
    print('\t Test performance:', test_perf)
    records['test_loss'].append(test_loss)
    records['test_reg_loss'].append(test_reg_loss)
    records['test_rank_loss'].append(test_rank_loss)
    records['test_perf'].append(test_perf)

    # best result
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        # In this place, remove some var that wouldn't be printed
        # without copy.copy()
        best_valid_perf = val_perf
        best_test_perf = test_perf
        print('Better valid loss:', best_valid_loss)
    t4 = time()
    print('epoch:', epoch, ('time: %.4f ' % (t4 - t1)))
print('\nBest Valid performance:', best_valid_perf)
print('Best Test performance:', best_test_perf)

with open('./records/{}_records.json'.format(type(model).__name__), 'w') as f:
    f.write(json.dumps(records))
