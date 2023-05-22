import os
import numpy as np
import json
import dgl
import torch
from scipy.sparse import csc_matrix


def get_word_graph_data(args, logging):

    '''
    weight [node, node]
    emb [node, embd_size]
    '''
    graph = torch.load(args.word_graph_path)
    emb = torch.load(args.word_embed_path)
    weight = graph["weight"]
    logging.info("word graph {}".format(len(weight)))
    return weight, emb


def build_graph_from_adj(adj):
    adj = csc_matrix(adj.numpy())
    row = adj.tocoo().row.tolist()
    col = adj.tocoo().col.tolist()
    data = adj.tocoo().data.tolist()
    g_w = dgl.graph((col, row))
    return g_w, torch.tensor(data)


def get_ontology_data(args, logging):

    '''
    get ontology_dense
    r
    '''
    with open(os.path.join(args.ontology_data_path, 'ontology_en2id.json'), 'r', encoding='utf8') as fin:
        on_ent2id = json.load((fin))

    with open(os.path.join(args.ontology_data_path, 'ontology_rel2id.json'), 'r', encoding='utf8') as fin:
        on_rel2id = json.load((fin))

    n_on_entity = len(on_ent2id)
    n_on_relation = len(on_rel2id)

    args.n_on_entity = n_on_entity
    args.n_on_relation = n_on_relation

    with open(os.path.join(args.ontology_data_path, 'ontology_graph_id.txt'), 'r', encoding='utf8') as fin:
        ontology_triples = []
        for i in fin.readlines():
            h, r, t = i.split()
            ontology_triples.append([int(h), int(r), int(t)])
    ontology_triples = np.array(ontology_triples)
    logging.info('#ontology_dense trian: %d' % len(ontology_triples))
    adj_list, degrees = get_adj_and_degrees(n_on_entity, ontology_triples)
    return ontology_triples, adj_list, degrees


def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i, triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees


def build_graph_from_triplets(num_nodes, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """

    # def comp_deg_norm(g):
    #     g = g.local_var()
    #     in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    #     norm = 1.0 / in_deg
    #     norm[np.isinf(norm)] = 0
    #     return norm

    g = dgl.graph(([], []))
    g.add_nodes(num_nodes)
    src, rel, dst = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    g.add_edges(src, dst)
    # norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel.astype('int64'), None
    # , norm.astype('int64')
