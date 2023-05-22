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

from aggregation import AggregateR, AggregateTR, AggregateSoft, AggregateWeight, AggregateTriple, AggregateEntity,  AggregateHR, MoE
from bi_view_gcn import OntologyRGCN, WordGCN
from dataloader import TestDataset, TestCandidateDataset

AGG_MOD = {
    'R': AggregateR,
    'RT': AggregateTR,
    'WEIGHT': AggregateWeight,
    'SOFT': AggregateSoft,
    'TRIPLE': AggregateTriple,
    'ENTITY': AggregateEntity,
    'HR': AggregateHR,
    "MOE": MoE
}


class KGEModelBiView(nn.Module):
    def __init__(self, args, model_name, nentity, nrelation, hidden_dim, gamma,  # Embedding
                 num_nodes, n_hidden, num_rels, n_bases, num_hidden_layers=1, dropout=0, use_cuda=False,  # RGCN
                 double_entity_embedding=False, double_relation_embedding=False):  # Embedding
        super(KGEModelBiView, self).__init__()
        self.n_hidden = n_hidden * 2 if double_relation_embedding else n_hidden  # RotatE double embedding
        
        # translated-base model information
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        # ontology graph encoder
        self.onto_rgcn = OntologyRGCN(num_nodes, self.n_hidden, self.hidden_dim, num_rels, n_bases, num_hidden_layers, dropout,
                                      use_cuda)
        # word graph encoder
        self.word_gcn = WordGCN(num_nodes, self.n_hidden, self.hidden_dim, num_hidden_layers, dropout,
                                use_cuda)

        # margin
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        # model inti value
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        #  double embedding Rotate
        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        # entity embedding
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # relation embedding
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # projection_matirx
        self.proj = nn.Parameter(torch.zeros(self.hidden_dim,self.n_hidden))
        nn.init.uniform_(
            tensor=self.proj,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # self.view_trans = nn.Parameter(torch.zeros(self.hidden_dim*2,self.n_hidden))
        # nn.init.uniform_(
        #     tensor=self.view_trans,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )

        # relation aggregate concat two graph  *2
        self.agg_model = AGG_MOD[args.agg](self.n_hidden, self.relation_dim)

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample,
                g_o, node_id, edge_type, edge_norm,
                g_w, word_embedding, rel_weight,
                mode='single'):
        if mode == 'single':
            batch_size, negative_sample_size, ontology_sample, part = sample[0].size(0), 1, sample[1], sample[0]
            head = torch.index_select(self.entity_embedding, dim=0, index=part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=part[:, 2]).unsqueeze(1)

        elif mode == 'head-batch':
            positive_sample, negative_sample_head, ontology_sample = sample
            batch_size, negative_sample_size = negative_sample_head.size(0), negative_sample_head.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_head.view(-1)).view(
                batch_size, negative_sample_size, -1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=positive_sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=positive_sample[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            positive_sample, negative_sample_tail, ontology_sample = sample
            batch_size, negative_sample_size = negative_sample_tail.size(0), negative_sample_tail.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=positive_sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_tail.view(-1)).view(
                batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,  # negative 1
            'DistMult': self.DistMult,  # negative 10
            'ComplEx': self.ComplEx,  # negative 10
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        onto_embed = self.onto_rgcn.forward(g_o, node_id, edge_type, edge_norm)
        word_embed = self.word_gcn.forward(g_w, word_embedding, rel_weight)

        h_o = torch.index_select(onto_embed, dim=0, index=ontology_sample[:, 0]).unsqueeze(1)
        r_o = torch.index_select(onto_embed, dim=0, index=ontology_sample[:, 1]).unsqueeze(1)
        t_o = torch.index_select(onto_embed, dim=0, index=ontology_sample[:, 2]).unsqueeze(1)

        h_w = torch.index_select(word_embed, dim=0, index=ontology_sample[:, 0]).unsqueeze(1)
        r_w = torch.index_select(word_embed, dim=0, index=ontology_sample[:, 1]).unsqueeze(1)
        t_w = torch.index_select(word_embed, dim=0, index=ontology_sample[:, 2]).unsqueeze(1)

        h_a = torch.matmul(h_o,self.proj)+torch.matmul(h_w,self.proj)
        r_a = torch.matmul(r_o,self.proj)+torch.matmul(r_w,self.proj)
        t_a = torch.matmul(t_o,self.proj)+torch.matmul(t_w,self.proj)

        # h_a = torch.matmul(torch.cat([h_o, h_w],dim=-1),self.view_trans)
        # r_a = torch.matmul(torch.cat([r_o, r_w],dim=-1),self.view_trans)
        # t_a = torch.matmul(torch.cat([r_o, r_w],dim=-1),self.view_trans)

        # h_a = h_o + h_w
        # r_a = r_o + r_w
        # t_a = t_w + t_o

        r = self.agg_model(relation, h_a, r_a, t_a)

        r = torch.matmul(r,self.proj.t())

        if self.model_name in model_func:
            score = model_func[self.model_name](head, r, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score,r_o.cpu().numpy(),r.cpu().numpy()

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator,
                   g_o, node_id, edge_type, edge_norm, # ontology graph
                   g_w, word_embedding, rel_weight,
                   args):
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, ontology_sample, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            ontology_sample = ontology_sample.cuda()

        negative_score = model((positive_sample, negative_sample, ontology_sample),
                               g_o, node_id, edge_type, edge_norm,
                               g_w, word_embedding, rel_weight,
                               mode=mode)
        positive_score = model((positive_sample, ontology_sample),
                               g_o, node_id, edge_type, edge_norm,
                               g_w, word_embedding, rel_weight)

        negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples,
                  g_o, node_id, edge_type, edge_norm,
                  g_w, word_embedding, rel_weight,
                  args):
        model.eval()

        # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation
        if args.only_test:
            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.n_kg_entity,
                    args.n_kg_relation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=int(args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )


            test_dataset_list = [test_dataloader_tail]
        else:
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.n_kg_entity,
                    args.n_kg_relation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=int(args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.n_kg_entity,
                    args.n_kg_relation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=int(args.cpu_num),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_tail,test_dataloader_head]

        unseen_logs = []
        seen_logs = []
        seen_save_logs = []
        unseen_save_logs = []
        seen_r_embedding_dic = {}
        unseen_r_embedding_dic = {}
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        if args.cuda:
            g_o = g_o.to(args.gpu)
            node_id = node_id.cuda()
            edge_type = edge_type.cuda()
            g_w = g_w.to(args.gpu)
            word_embedding = word_embedding.cuda()
            rel_weight = rel_weight.cuda()

            # edge_norm = edge_norm.cuda()

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, ontology_sample, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        ontology_sample = ontology_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score,r_o,r_a = model((positive_sample, negative_sample, ontology_sample),
                                    g_o, node_id, edge_type, edge_norm,
                                    g_w, word_embedding, rel_weight,
                                    mode)

                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        _,r,_ = positive_sample[i].tolist()
                        r_o_represent = r_o[i]
                        r_a_represent = r_a[i]

                        if args.only_test:
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        else:
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        if r in args.seen_relation:
                            seen_logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })
                            seen_save_logs.append({
                            'Triple': positive_sample[i].cpu().numpy().tolist(),
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })
                            if str(r) not in seen_r_embedding_dic:
                                seen_r_embedding_dic[str(r)] = {"o":r_o_represent,
                                                            "a":r_a_represent}
                            else:
                                seen_r_embedding_dic[str(r)]["o"] = np.concatenate((seen_r_embedding_dic[str(r)]["o"],r_o_represent),axis=0)
                                seen_r_embedding_dic[str(r)]["a"] = np.concatenate((seen_r_embedding_dic[str(r)]["a"],r_a_represent),axis=0)
                        else:
                            unseen_logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })
                            unseen_save_logs.append({
                            'Triple': positive_sample[i].cpu().numpy().tolist(),
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                            if str(r) not in unseen_r_embedding_dic:
                                unseen_r_embedding_dic[str(r)] = {"o":r_o_represent,
                                                            "a":r_a_represent}
                            else:
                                unseen_r_embedding_dic[str(r)]["o"] = np.concatenate((unseen_r_embedding_dic[str(r)]["o"],r_o_represent),axis=0)
                                unseen_r_embedding_dic[str(r)]["a"] = np.concatenate((unseen_r_embedding_dic[str(r)]["a"],r_a_represent),axis=0)


                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        seen_metrics = {}
        unseen_metrics = {}
        all_metrics = {}

        logging.info("seen relation triples:{}, unseen relation triples:{}".format(len(seen_logs),len(unseen_logs)))


        for metric in unseen_logs[0].keys():
            seen_metrics[metric] = sum([log[metric] for log in seen_logs]) / (len(seen_logs)+0.000001)
        
        for metric in unseen_logs[0].keys():
            unseen_metrics[metric] = sum([log[metric] for log in unseen_logs]) / (len(unseen_logs)+0.00001)
        
        for metric in unseen_logs[0].keys():
            all_metrics[metric] = sum([log[metric] for log in seen_logs+unseen_logs]) / (len(seen_logs)+len(unseen_logs)+0.000001)

        return all_metrics,seen_metrics,unseen_metrics,seen_save_logs,unseen_save_logs,seen_r_embedding_dic,unseen_r_embedding_dic

