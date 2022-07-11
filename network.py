import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveLinear(nn.Linear):
    def __init__(self, in_feats, out_feats, is_first=False, is_last=False, bias=True):
        super(ProgressiveLinear, self).__init__(in_feats, out_feats, bias=bias)
        self.is_first = is_first
        self.is_last = is_last
        
        # width of each sub-network & an index of selected subnet
        self.subnet_widths = [out_feats] if self.is_first else [in_feats]
        self.subnet_index = len(self.subnet_widths) - 1
    
    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight, self.bias)
        else:
            width = self.subnet_widths[self.subnet_index]
            i = slice(None) if self.is_last else slice(width)
            j = slice(None) if self.is_first else slice(width)
            return F.linear(x, self.weight[i, j], self.bias[i])


class ProgressiveSiren(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_hidden_layers, out_feats, device, 
                outermost_linear=False, in_omega_0=30., hidden_omega_0=30.):
        super(ProgressiveSiren, self).__init__()

        # subnet_widths = [hidden_feats] if len(hidden_feats) == 0 else hidden_feats
        # hidden_feats = subnet_widths[-1]

        self.net = [ProgressiveLinear(in_feats, hidden_feats, is_first=True)] + \
                   [ProgressiveLinear(hidden_feats, hidden_feats) for _ in range(n_hidden_layers)] + \
                   [ProgressiveLinear(hidden_feats, out_feats, is_last=True)]
        
        self.net = nn.ModuleList(self.net)
        self.in_omega_0 = in_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.outermost_linear = outermost_linear
        self.discarded_weight = 0
        self.runnable_widths = []
        self.device = device

        # initialize weights
        with torch.no_grad():
            for l in self.net:
                initialize_siren_weights(l.weight, self.hidden_omega_0, l.is_first)
    
    def forward(self, x):
        for l in self.net:
            o = self.in_omega_0 if l.is_first else self.hidden_omega_0
            if l.is_last:
                x = o * l(x)
            else:
                x = torch.sin(o * l(x))
        return x

    def forward_residual(self, x):
        for l in self.net:
            o = self.in_omega_0 if l.is_first else self.hidden_omega_0
            if l.is_last:
                if self.training:
                    W_cpy = l.weight.clone()
                else:
                    width = l.subnet_widths[l.subnet_index]
                    W_cpy = l.weight[:, :width].clone()
                # set weights of smaller sub-network zero
                if l.subnet_index != 0:
                    prev_index = l.subnet_index - 1
                    small_width = l.subnet_widths[prev_index]
                    W_cpy[:, :small_width] = 0
                # linear transformation (no bias)
                x = o * F.linear(x, W_cpy, torch.zeros_like(l.bias))
            else:
                x = torch.sin(o * l(x))
        return x
    
    def grow_width(self, width):
        with torch.no_grad():
            device = torch.device(self.device)
            for l in self.net:
                # current weight dimension
                out_feats, in_feats = l.weight.shape
                # increased weight dimension
                out_feats_new = out_feats if l.is_last else out_feats + width
                in_feats_new = in_feats if l.is_first else in_feats + width
                # make empty weight matrix and bias vector
                W_new = nn.parameter.Parameter(torch.empty(out_feats_new, in_feats_new, device=device, requires_grad=True))
                b_new = nn.parameter.Parameter(torch.empty(out_feats_new, device=device, requires_grad=True))
                # initialize weight and bias
                initialize_siren_weights(W_new, self.hidden_omega_0, l.is_first)
                initialize_siren_bias(b_new, fan_in=l.weight.shape[1])
                # replace the part of the weight matrix with the trained weight matrix
                W_new[:out_feats, :in_feats] = l.weight
                b_new[:out_feats] = l.bias
                # update weight and bias of each layer
                l.weight = W_new
                if not l.is_last:
                    l.bias = b_new
                # append width to subnet_widths and update subnet_index of each layer
                l.subnet_widths += [l.subnet_widths[-1] + width]
                l.subnet_index += 1
                # cut weight connections from small sub-network to large sub-network
                if not l.is_first and not l.is_last:
                    l.weight[:out_feats, in_feats:] = 0
                    self.discarded_weight += (l.weight[:out_feats, in_feats:].shape[0] * l.weight[:out_feats, in_feats:].shape[1])
                #!NOTE progressive initialization
                if not l.is_first:
                    l.weight[out_feats:, in_feats:] = 0
    
    def select_subnet(self, subnet_index):
        for l in self.net: l.subnet_index = subnet_index

    def freeze_subnet(self, subnet_index):
        for l in self.net:
            width = l.subnet_widths[subnet_index]
            i = slice(None) if l.is_last else slice(width)
            j = slice(width) if l.is_last else slice(None)
            l.weight.grad[i, j] = 0
            l.bias.grad[i] = 0


def initialize_siren_weights(W, hidden_omega_0, is_first=False):
    fan_in = W.shape[1]
    # calculate an upper bound of uniform distribution
    u = 1 / fan_in if is_first else np.sqrt(6/fan_in) / hidden_omega_0
    # initialize weights
    nn.init.uniform_(W, -u, u)


def initialize_siren_bias(b, fan_in):
    # calculate an upper bound of uniform distribution
    u = 1/np.sqrt(fan_in)
    # initialize bias
    nn.init.uniform_(b, -u, u)
