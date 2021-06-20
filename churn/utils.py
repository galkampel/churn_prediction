import torch.nn as nn
from collections import OrderedDict


def set_ordered_dict(num_layers, in_features, **params):
    if num_layers == 1:
        return OrderedDict([
            ('fc_cls', nn.Linear(in_features, 1)),
            ('sigmoid', nn.Sigmoid())
        ])
    elif num_layers == 2:
        out_features = params.get('out_features', 16)
        layers = [('fc1', nn.Linear(in_features, out_features))]
        apply_bn = params.get('apply_bn', False)
        if apply_bn:
            layers.append(('bn', nn.BatchNorm1d(out_features)))
        layers.append(('relu', nn.ReLU()))
        p = params.get('p', 0.0)
        if p > 0.0:
            layers.append(('dropout', nn.Dropout(p=p)))
        layers.append(('fc_cls', nn.Linear(out_features, 1)))
        layers.append(('sigmoid', nn.Sigmoid()))
        ordered_dict = OrderedDict(layers)
        return ordered_dict
