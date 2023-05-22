import torch
import torch.nn as nn
import dgl
from scipy.sparse import csc_matrix
from dgl.nn.pytorch import EdgeWeightNorm, GraphConv


def gen_dgl_graph(path_graph, path_embed):
    graph = torch.load(path_graph)
    embd = torch.load(path_embed).clone().detach()
    weight = graph['weight']
    weight = csc_matrix(weight.numpy())
    row = weight.tocoo().row.tolist()
    col = weight.tocoo().col.tolist()
    data = weight.tocoo().data.tolist()
    g_w = dgl.graph((col, row))
    g_w.edata['w'] = torch.tensor(data)
    g_w.ndata['emb'] = embd
    return g_w


class WordGnn(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(WordGnn, self).__init__()
        self.norm = EdgeWeightNorm(norm='both')
        self.conv1 = GraphConv(embed_size, hidden_size, norm='none', weight=True, bias=True)
        self.conv2 = GraphConv(hidden_size, hidden_size, norm='none', weight=True, bias=True)

    def forward(self, g):
        norm_edge_weight = self.norm(g, g.edata["w"])
        res = self.conv1(g, g.ndata["emb"], edge_weight=norm_edge_weight)
        res = self.conv2(g, res, edge_weight=norm_edge_weight)
        return res


if __name__ == '__main__':
    g_w = gen_dgl_graph("../data/DB100K-zero-shot/ontology_dense/word_graph.pytorch",
                        "../data/DB100K-zero-shot/ontology_dense/word_embedding.pytorch")
    print(g_w)
    wgnn = WordGnn(100, 50)
    res = wgnn(g_w)
    print(res)
