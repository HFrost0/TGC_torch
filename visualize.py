import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def draw_loss(model_name='RelationLSTM', loss_name='loss', start_index=1):
    with open('./records/{}_records.json'.format(model_name), 'r') as f:
        records = json.load(f)
    plt.plot(range(start_index, len(records['train_{}'.format(loss_name)]) + 1),
             records['train_{}'.format(loss_name)][start_index - 1:], label='train loss')
    plt.plot(range(start_index, len(records['val_{}'.format(loss_name)]) + 1),
             records['val_{}'.format(loss_name)][start_index - 1:], label='validate loss')
    plt.plot(range(start_index, len(records['test_{}'.format(loss_name)]) + 1),
             records['test_{}'.format(loss_name)][start_index - 1:], label='test loss')
    plt.xlabel('epoch')
    plt.ylabel(loss_name)
    plt.title(
        '{} {}-{} epoch {}'.format(model_name, start_index, len(records['train_{}'.format(loss_name)]), loss_name))
    plt.legend()
    plt.show()


def draw_performance(model_names, sets='test', mark='mse', start_index=1):
    for name in model_names:
        with open('./records/{}_records.json'.format(name), 'r') as f:
            records = json.load(f)
        data = [x[mark] for x in records['{}_perf'.format(sets)]][start_index - 1:]
        plt.plot(range(start_index, len(records['{}_perf'.format(sets)]) + 1), data, label=name)
        print("{}  {}\nmin:{}\nmax:{}".format(name, mark, min(data), max(data)))
    plt.xlabel('epoch')
    plt.ylabel(mark)
    plt.title('{} \nperformance on {} sets'.format(model_names, sets))
    plt.legend()
    plt.show()


def test(model_names, sets='val'):
    for name in model_names:
        with open('./records/{}_records.json'.format(name), 'r') as f:
            records = json.load(f)
        epoch = np.argmin(records['val_loss'])
        print(name, 'mrr:', records[sets + '_perf'][epoch]['mrrt'], 'irr:', records[sets + '_perf'][epoch]['btl'])


def draw_rel_weight():
    rel_weight = np.load('temp_data/rel_weight.npy').reshape(43)
    rel_num = np.load('temp_data/rel_num.npy')
    rel_type = np.arange(1, 44)

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('rel type')
    ax1.set_ylabel('rel_weight', color=color)
    ax1.plot(rel_type, rel_weight, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('rel_num', color=color)  # we already handled the x-label with ax1
    ax2.plot(rel_type, rel_num, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1.grid()
    plt.title('Different relation weight and number for each type')
    plt.show()


if __name__ == '__main__':
    test(['TGCNet', 'DGCNet', 'MyModule'])
    # draw_loss(model_name='Net', start_index=1)
    draw_performance(model_names=['TGCNet', 'DGCNet', 'MyModule'], mark='btl',
                     start_index=5)
    # draw_performance(model_names=['RelationLSTM', 'GCNNet', 'Net'], mark='mrrt', start_index=1)
    # draw_performance(model_names=['RelationLSTM', 'GCNNet', 'Net'], mark='btl', start_index=1)

    # draw_rel_weight()
