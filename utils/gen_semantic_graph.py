import torch

word_emb_path = "../data/dbpedia_ontology/ontology/word_embedding.pytorch"
save_path = "../data/dbpedia_ontology/ontology/word_graph.pytorch"
import torch.nn.functional as F
from scipy.sparse import csc_matrix
import dgl

word_embedding = torch.load(word_emb_path)
none_node = torch.where(torch.sum(word_embedding, dim=-1) == 0)[0]
print(word_embedding.shape)
num_entity = word_embedding.shape[0]
word_embedding = F.normalize(word_embedding)
word_embedding_trans = word_embedding.t()
score = torch.matmul(word_embedding, word_embedding_trans)
# a = torch.dist(word_embedding[0], word_embedding[1])
# man_dist = [[0 for _ in range(num_entity)] for _ in range(num_entity)]

# for i in range(num_entity):
#     for j in range(num_entity):
#         d = torch.dist(word_embedding[i], word_embedding[j])
#         man_dist[i][j] = d

# print(d)
# print(a)
print(score)
weight = torch.where(score > 0.95, score, torch.tensor(0.0))
graph = torch.where(weight <= 0.2, weight, torch.tensor(1.0))
graph = graph.to(torch.int64)
weight = weight - torch.eye(len(weight))
weight[:, none_node] = 0
weight[none_node, :] = 0
weight = weight + torch.eye(len(weight))
print(score.shape)
print(weight)
print(graph)
print(graph.sum())
word_graph = {"graph": graph,
              "weight": weight}
torch.save(word_graph, save_path)
# torch.save("../data/DB100K-zero-shot/ontology_dense/word_weight.pytorch", weight)

# for i, k in enumerate(graph):
#     print(i)
#     print(sum(k))
adj = csc_matrix(weight.numpy())
print(len(adj.tocoo().row.tolist()))
print(len(adj.tocoo().col.tolist()))
print(len(adj.tocoo().data.tolist()))
