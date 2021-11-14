import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    # return ratio's mse loss
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    # formula (4-6)
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
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class GraphModule(nn.Module):
    def __init__(self, batch_size, fea_shape, rel_encoding, rel_mask, inner_prod=False):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = fea_shape
        self.inner_prod = inner_prod
        self.relation = nn.Parameter(torch.tensor(rel_encoding, dtype=torch.float32, requires_grad=False))
        self.rel_mask = nn.Parameter(torch.tensor(rel_mask, dtype=torch.float32, requires_grad=False))
        self.all_one = nn.Parameter(torch.ones(self.batch_size, 1, dtype=torch.float32, requires_grad=False))
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
        weight_masked = F.softmax(self.rel_mask + weight, dim=0)
        outputs = weight_masked @ inputs
        return outputs


class StockLSTM(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.lstm_cell = nn.LSTM(5, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, inputs):
        x, _ = self.lstm(inputs)
        x = x[:, -1, :]
        prediction = F.leaky_relu(self.fc(x))
        return prediction


class RelationLSTM(nn.Module):
    def __init__(self, batch_size, rel_encoding, rel_mask, inner_prod=False):
        super().__init__()
        self.batch_size = batch_size
        self.lstm = nn.LSTM(5, 64, batch_first=True)
        self.graph_layer = GraphModule(batch_size, 64, rel_encoding, rel_mask, inner_prod)
        self.fc = nn.Linear(64 * 2, 1)

    def forward(self, inputs):
        x, _ = self.lstm(inputs)
        x = x[:, -1, :]
        outputs_graph = self.graph_layer(x)
        outputs_cat = torch.cat([x, outputs_graph], dim=1)
        prediction = F.leaky_relu(self.fc(outputs_cat))
        return prediction
