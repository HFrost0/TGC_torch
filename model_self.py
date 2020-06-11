"""模型与loss实现"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch_geometric


def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to('cuda')
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    # return ratio's mse loss
    reg_loss = functional.mse_loss(ground_truth * mask, return_ratio * mask)
    # 公式(4-6)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t().contiguous(),
        all_one @ return_ratio.t().contiguous()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t().contiguous(),
        ground_truth @ all_one.t().contiguous()
    )
    mask_pw = mask @ mask.t().contiguous()
    rank_loss = torch.mean(
        functional.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class GraphModule(nn.Module):
    def __init__(self, batch_size, fea_shape, rel_encoding, rel_mask, inner_prod=False):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = fea_shape
        self.inner_prod = inner_prod
        self.relation = torch.tensor(rel_encoding, dtype=torch.float32, requires_grad=False).to('cuda')
        self.rel_mask = torch.tensor(rel_mask, dtype=torch.float32, requires_grad=False).to('cuda')
        self.all_one = torch.ones(self.batch_size, 1, dtype=torch.float32).to('cuda')
        # trainable
        # self.rel_weight = torch.ones(43, 1, requires_grad=True)
        # self.rel_bias = torch.zeros(batch_size, 1, requires_grad=True)
        self.rel_weight = nn.Linear(43, 1)
        if self.inner_prod is False:
            self.head_weight = nn.Linear(fea_shape, 1)
            self.tail_weight = nn.Linear(fea_shape, 1)

    def forward(self, inputs):
        rel_weight = self.rel_weight(self.relation)
        if self.inner_prod:
            inner_weight = inputs @ inputs.t().contiguous()
            weight = inner_weight @ rel_weight[:, :, -1]
        else:
            all_one = self.all_one
            head_weight = self.head_weight(inputs)
            tail_weight = self.tail_weight(inputs)
            weight = (head_weight @ all_one.t().contiguous() +
                      all_one @ tail_weight.t().contiguous()) + rel_weight[:, :, -1]
        weight_masked = functional.softmax(self.rel_mask + weight, dim=0)
        outputs = weight_masked @ inputs
        return outputs


class StockLSTM(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.lstm_cell = nn.LSTMCell(5, 64)
        self.h0 = torch.zeros(self.batch_size, 64).to('cuda')
        self.c0 = torch.zeros(self.batch_size, 64).to('cuda')
        self.fc = nn.Linear(64, 1)

    def forward(self, inputs):
        hx = self.h0
        cx = self.c0
        for x in inputs.split(1, 1):
            x.squeeze_()
            hx, cx = self.lstm_cell(x, (hx, cx))
        prediction = functional.leaky_relu(self.fc(hx))
        return prediction


class RelationLSTM(nn.Module):
    def __init__(self, batch_size, rel_encoding, rel_mask, inner_prod=False):
        super().__init__()
        self.batch_size = batch_size
        self.lstm_cell = nn.LSTMCell(5, 64)
        self.h0 = torch.zeros(self.batch_size, 64).to('cuda')
        self.c0 = torch.zeros(self.batch_size, 64).to('cuda')
        self.graph_layer = GraphModule(batch_size, 64, rel_encoding, rel_mask, inner_prod)
        self.fc = nn.Linear(64, 1)

    def forward(self, inputs):
        hx = self.h0
        cx = self.c0
        for x in inputs.split(1, 1):
            x.squeeze_()
            hx, cx = self.lstm_cell(x, (hx, cx))
        outputs_proped = self.graph_layer(hx)
        outputs_concated = torch.cat((hx, outputs_proped), dim=1)
        # prediction = functional.leaky_relu(self.fc(outputs_concated))
        prediction = functional.leaky_relu(self.fc(outputs_proped))

        return prediction


class MyModule(nn.Module):
    def __init__(self, batch_size, rel_encoding, rel_mask, inner_prod=False):
        super().__init__()
        self.batch_size = batch_size
        self.lstm_cell = nn.LSTMCell(5, 64)
        self.h0 = torch.zeros(self.batch_size, 64).to('cuda')
        self.c0 = torch.zeros(self.batch_size, 64).to('cuda')
        self.graph_layer = GraphModule(batch_size, 5, rel_encoding, rel_mask, inner_prod)
        self.predict_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.LeakyReLU())

    def forward(self, inputs):
        hx = self.h0
        cx = self.c0
        for x in inputs.split(1, 1):
            x.squeeze_()
            x = self.graph_layer(x)
            hx, cx = self.lstm_cell(x, (hx, cx))
        prediction = self.predict_layer(hx)
        return prediction

