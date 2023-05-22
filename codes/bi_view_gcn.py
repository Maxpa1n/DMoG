#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset
import torch as th
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv, EdgeWeightNorm, GraphConv
import dgl


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, input_dim, outpt_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.outpt_dim = outpt_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.norm = nn.LayerNorm(input_dim)

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers-1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o`
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
            h = self.norm(h)
        return h


class BaseGNN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, num_hidden_layers=1, dropout=0, use_cuda=False):
        super(BaseGNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.norm = EdgeWeightNorm(norm='both')

        # create gnn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers-1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o`
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, w):

        '''
        g: word graph
        h: word embedding
        w: graph weight
        '''

        norm_edge_weight = self.norm(g, w)
        for layer in self.layers:
            h = layer(g, h, edge_weight=norm_edge_weight)
        return h


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class OntologyRGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.input_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers else None
        return RelGraphConv(self.input_dim, self.input_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True,
                            dropout=self.dropout)
    def build_output_layer(self):
        act =  None
        return RelGraphConv(self.input_dim, self.outpt_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True, bias=False,
                            dropout=self.dropout)

class WordGCN(BaseGNN):
    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers else None
        return GraphConv(self.input_dim, self.output_dim, norm='both', weight=True, bias=True, activation=act)
    
    def build_output_layer(self):
        act = None
        return GraphConv(self.input_dim, self.output_dim,  norm='both', weight=True, bias=False, activation=act)
