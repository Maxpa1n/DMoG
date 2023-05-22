#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import datetime

import numpy as np
import torch

from torch.utils.data import DataLoader

from bi_view_model import KGEModelBiView
from rgcn_data import get_ontology_data, get_adj_and_degrees, build_graph_from_triplets, get_word_graph_data, \
    build_graph_from_adj
from dataloader import TrainDataset, DataPrefetcher
from dataloader import BidirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="use GPU number")
    parser.add_argument('--agg', type=str, default='R')
    parser.add_argument('--only_test', action='store_true')

    # word graph
    parser.add_argument("--word_embed_path", type=str,
                        help="word embedding path")
    parser.add_argument("--word_graph_path", type=str,
                        help="word graph path")
    # ontology graph
    parser.add_argument("--n_basses", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
                        help="number of hidden units")
    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
                        help="number of minimum training epochs")
    parser.add_argument("--eval-batch-size", type=int, default=500,
                        help="batch size when evaluating")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--negative-sample", type=int, default=10,
                        help="number of negative samples per positive sample")
    parser.add_argument("--ontology_data_path", type=str, required=True,
                        help="ontology data path")
    parser.add_argument("--edge_sampler", type=str, default="uniform",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    # instance argument -----------------------------
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--fine_turning', action='store_true')

    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.do_train:
        args.save_path = os.path.join(args.save_path, args.data_path.split('/')[-1])
        args.save_path = os.path.join(args.save_path, args.model)
        args.save_path = os.path.join(args.save_path, args.agg)
        args.save_path = os.path.join(args.save_path, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        # args.save_path += datetime.datetime.now().strftime("%Y%m%d%H%M")
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    # --------------------------------ontology entity/relation to id---------------------------------
    with open(os.path.join(args.ontology_data_path, 'ontology_en2id.json'), 'r', encoding='utf8') as fin:
        on_ent2id = json.load(fin)

    with open(os.path.join(args.ontology_data_path, 'ontology_rel2id.json'), 'r', encoding='utf8') as fin:
        on_rel2id = json.load(fin)

    # --------------------------------kg entity/relation to id---------------------------------
    with open(os.path.join(args.data_path, 'instance_en2id.json'), 'r', encoding='utf8') as fin:
        kg_ent2id = json.load(fin)
    with open(os.path.join(args.data_path, 'instance_rel2id.json'), 'r', encoding='utf8') as fin:
        kg_rel2id = json.load(fin)

    # ---------------------data information----------------------
    n_on_entity = len(on_ent2id)
    n_on_relation = len(on_rel2id)

    n_kg_entity = len(kg_ent2id)
    n_kg_relation = len(kg_rel2id)

    args.n_on_entity = n_on_entity
    args.n_on_relation = n_on_relation

    args.n_kg_entity = n_kg_entity
    args.n_kg_relation = n_kg_relation

    logging.info('--------------------------------Data info--------------------------------')
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)

    logging.info('#ontology_entity: %d' % n_on_entity)
    logging.info('#ontology_elation: %d' % n_on_relation)

    logging.info('#instance_entity: %d' % args.n_kg_entity)
    logging.info('#instance_relation: %d' % args.n_kg_relation)

    # --------------------------------instance data----------------------------------
    # train
    with open(os.path.join(args.data_path, 'train_id.json'), 'r', encoding='utf8') as fin:
        train_triples = json.load(fin)
    logging.info('#instance train: %d' % len(train_triples))
    logging.info('#train path:{}'.format(args.data_path))

    # valid
    with open(os.path.join(args.data_path, 'valid_id.json'), 'r', encoding='utf8') as fin:
        valid_triples = json.load(fin)
    logging.info('#instance valid: %d' % len(valid_triples))
    logging.info('#valid path:{}'.format(args.data_path))

    # test
    with open(os.path.join(args.data_path, 'test_id.json'), 'r', encoding='utf8') as fin:
        test_triples = json.load(fin)
    logging.info('#instance test: %d' % len(test_triples))
    logging.info('#test path:{}'.format(args.data_path))

    unseen_relation = set()
    seen_relation = set()

    data_information = {"train":0,
            "valid":{"seen":0,"unseen":0},
            "test":{"seen":0,"unseen":0},
            "seen_relations":{"lenth":0,"list":[]},
            "unseen_relations":{"lenth":0,"list":[]}}
    data_information["train"] = len(train_triples)

    for i in train_triples:
        _,r,_ = i["instance"]
        seen_relation.add(r)

    for i in valid_triples:
        _,r,_ = i["instance"]
        if r  not in seen_relation:
            unseen_relation.add(r)
            data_information["valid"]["unseen"]+=1
        else:
            data_information["valid"]["seen"]+=1

    
    for i in test_triples:
        _,r,_ = i["instance"]
        if r not in seen_relation:
            unseen_relation.add(r)
            data_information["test"]["unseen"]+=1
        else:
            data_information["test"]["seen"]+=1

    all_true_triples = [i['instance'] for i in train_triples] \
                       + [i['instance'] for i in valid_triples] \
                       + [i['instance'] for i in test_triples]

    args.unseen_relation = list(unseen_relation)
    args.seen_relation = list(seen_relation)

    
    
    data_information["seen_relations"]["lenth"] = len(args.seen_relation)
    data_information["seen_relations"]["list"] = args.seen_relation
    data_information["unseen_relations"]["lenth"] = len(args.unseen_relation)
    data_information["unseen_relations"]["list"] = args.unseen_relation

    with open(os.path.join(args.data_path, 'data_information.json'), 'w') as f:
        json.dump(data_information,f)

    if args.only_test:
        with open(os.path.join(args.data_path, 'test_candidates_id.json'), 'r') as f:
            all_true_triples = json.load(f)

    # ------------------------------cuda setting------------------------------
    use_cuda = args.gpu > -1 and torch.cuda.is_available() and args.cuda
    # bug of dgl
    torch.cuda.set_device(args.gpu)
    # ------------------------------ontology data-----------------------------
    ontology_triples, adj_list, degrees = get_ontology_data(args, logging)
    g_o, rel, edge_norm = build_graph_from_triplets(n_on_entity, ontology_triples)

    node_id = torch.arange(0, args.n_on_entity, dtype=torch.long).view(-1, 1)
    edge_type = torch.from_numpy(rel)

    deg = g_o.in_degrees(range(g_o.number_of_nodes())).float().view(-1, 1)
    if use_cuda:
        g_o = g_o.to(args.gpu)
        node_id, deg = node_id.cuda(), deg.cuda()
        edge_type = edge_type.cuda()
    # ----------------------------word graph data------------g_w, word_embedding, rel_weight,-----------------
    word_weight, word_embedding = get_word_graph_data(args, logging)
    g_w, rel_weight = build_graph_from_adj(word_weight)
    if use_cuda:
        g_w = g_w.to(args.gpu)
        rel_weight = rel_weight.cuda()
        word_embedding = word_embedding.cuda()

    # ---------------------------model----------------------------------------
    kge_model = KGEModelBiView(
        args,
        model_name=args.model,
        nentity=n_kg_entity,
        nrelation=n_kg_relation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        # on
        num_nodes=n_on_entity,
        n_hidden=args.n_hidden,
        num_rels=n_on_relation,
        n_bases=args.n_basses,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        use_cuda=use_cuda)

    logging.info('--------------------------------Model Parameter Configuration:--------------------------------')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if use_cuda:
        kge_model = kge_model.cuda()

    # --------------------------------training dataset&optimizer---------------------------------
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, n_kg_entity, n_on_relation, args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, n_kg_entity, n_on_relation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        # train_iterator = DataPrefetcher(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Randomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('--------------------------------Start Training--------------------------------')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.do_train:
        logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []

        best_h1 = -1
        best_step = 0

        # Training Loop
        for step in range(init_step, args.max_steps):
            log = kge_model.train_step(kge_model, optimizer, train_iterator,
                                       g_o, node_id, edge_type, edge_norm,
                                       g_w, word_embedding, rel_weight,
                                       args)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.log_steps == 0:
                logging.info('----------------Training average----------------')
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training Loss:', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                logging.info('----------------Evaluating on Valid Dataset----------------')
                all_metircs, seen_metrics,unseen_metrics,_, _ = \
                kge_model.test_step(kge_model, valid_triples, all_true_triples,
                                                 g_o, node_id, edge_type, edge_norm,
                                                 g_w, word_embedding, rel_weight,
                                                 args)

                log_metrics('Valid SEEN', step, seen_metrics)
                logging.info('-------------------------------')
                log_metrics('Valid UNSEEN', step, unseen_metrics)
                logging.info('-------------------------------')
                log_metrics('Valid ALL', step, all_metircs)
                

                if all_metircs['HITS@10'] > best_h1:
                    save_model(kge_model, optimizer, save_variable_list, args)
                    best_h1 = all_metircs['HITS@10']
                    best_step = step
                


                logging.info('Best HITS@10 {}, Best step {}'.format(best_h1, best_step))

    if args.do_valid:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        # ls = torch.nn.MSELoss()
        # o = ls(torch.eye(100,100),(kge_model.proj@kge_model.proj.t()))
        logging.info('----------------Evaluating on Valid Dataset----------------')

        all_metircs, seen_metrics,unseen_metrics,_, _ = \
                kge_model.test_step(kge_model, valid_triples, all_true_triples,
                                                 g_o, node_id, edge_type, edge_norm,
                                                 g_w, word_embedding, rel_weight,
                                                 args)

        log_metrics('Valid SEEN', step, seen_metrics)
        logging.info('-------------------------------')
        log_metrics('Valid UNSEEN', step, unseen_metrics)
        logging.info('-------------------------------')
        log_metrics('Valid ALL', step, all_metircs)
        

    if args.do_test:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        logging.info('----------------Evaluating on Test Dataset----------------')

        all_metircs, seen_metrics,unseen_metrics,seen_logs, unseen_logs = \
                kge_model.test_step(kge_model, test_triples, all_true_triples,
                                                 g_o, node_id, edge_type, edge_norm,
                                                 g_w, word_embedding, rel_weight,
                                                 args)

        with open(os.path.join(args.save_path, 'result_seen.json'), 'w', encoding='utf8') as f:
            json.dump(seen_logs, f)
        with open(os.path.join(args.save_path, 'result_unseen.json'), 'w', encoding='utf8') as f:
            json.dump(unseen_logs, f)
        log_metrics('Test SEEN', step, seen_metrics)
        logging.info('-------------------------------')
        log_metrics('Test UNSEEN', step, unseen_metrics)
        logging.info('-------------------------------')
        log_metrics('Test ALL', step, all_metircs)


if __name__ == '__main__':
    main(parse_args())
