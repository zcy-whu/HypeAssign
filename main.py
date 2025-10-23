import math
import os
import pickle
import time
from datetime import datetime
from util import calcRegLoss, accuracy_at_one_multi_label, EarlyStopping, Logger, getOriginData
from HyperGraphHelper import HyperGraphHelper
from Node import Node
from Model import Model
import torch
import argparse
import sys

t = time.strftime("-%Y%m%d-%H%M%S", time.localtime())  
sys.stdout = Logger('./data/log/log' + t + '.txt', sys.stdout)

def run(args, chunk_id):
    # load data
    (train_data, train_data_create, train_data_modified, train_data_removed, train_data_report_closed,
     train_data_comment, train_data_similar, train_fixers, pathDict,
     val_data, val_data_create, val_data_modified, val_data_removed, val_data_report, val_data_comment,
     val_data_similar, val_fixers,
     test_data, test_data_create, test_data_modified, test_data_removed, test_data_report, test_data_comment,
     test_data_similar, test_fixers) = getOriginData_zcy(args.dataset, chunk_id)

    issueCreatedTimeMap = {}  
    for _, issue in train_data.iterrows(): 
        issueCreatedTimeMap[issue['_id']] = issue['created_at']

    graph = HyperGraphHelper.SaveAndGetGraph(args, chunk_id, train_data, train_data_create, train_data_modified,
                                                 train_data_removed, train_data_report_closed, train_data_comment,
                                                 train_data_similar, train_fixers, pathDict,
                                                 val_data, val_data_create, val_data_modified, val_data_removed,
                                                 val_data_report, val_data_comment, val_data_similar, val_fixers,
                                                 test_data, test_data_create, test_data_modified, test_data_removed,
                                                 test_data_report, test_data_comment, test_data_similar, test_fixers,
                                                 issueCreatedTimeMap)
    train_positive_indices = []
    for _, issue in train_data.iterrows():
        tmp = []
        fixers = train_fixers[issue['_id']]
        for f in fixers:
            node_1 = graph.get_node_by_content(Node.STR_NODE_TYPE_DEVELOPER, f)
            tmp.append(node_1.id - len(graph.node_type['issue']))
        train_positive_indices.append(tmp)

    val_positive_indices = []
    for _, issue in val_data.iterrows():
        tmp = []
        fixers = val_fixers[issue['_id']]
        for f in fixers:
            node_1 = graph.get_node_by_content(Node.STR_NODE_TYPE_DEVELOPER, f)
            tmp.append(node_1.id - len(graph.node_type['issue']))
        val_positive_indices.append(tmp)

    test_positive_indices = []
    for _, issue in test_data.iterrows():
        tmp = []
        fixers = test_fixers[issue['_id']]
        for f in fixers:
            node_1 = graph.get_node_by_content(Node.STR_NODE_TYPE_DEVELOPER, f)
            tmp.append(node_1.id - len(graph.node_type['issue']))
        test_positive_indices.append(tmp)

    model = Model(graph=graph, issue_num=len(graph.node_type['issue']),
                  dev_num=len(graph.node_type['developer']), file_num=len(graph.node_type['file']), latdim=32,
                  hyperNum=128, layer_num=2, temp=args.temp, leaky=args.leaky,
                  train_data=train_data, val_data=val_data, test_data=test_data,
                  train_positive_indices=train_positive_indices, val_positive_indices=val_positive_indices, test_positive_indices=test_positive_indices)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=10, verbose=False, path=f'{args.model_out_path}checkpoint_{chunk_id}.pt')
    for epoch in range(1000):
        model.train()
        hingeLoss, sslLoss, all_pred, hingeLoss_val, hingeLoss_test = model.calcLosses()
        sslLoss = sslLoss * args.ssl_reg
        regLoss = calcRegLoss(model) * args.reg
        loss = hingeLoss + regLoss + sslLoss

        accuracy, mrr, hit_k = accuracy_at_one_multi_label(all_pred, train_positive_indices)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f'epoch: {epoch}, loss: {loss}, hingeLoss: {hingeLoss}, hingeLoss_val: {hingeLoss_val}, hingeLoss_test: {hingeLoss_test}, sslLoss: {sslLoss}, accuracy: {accuracy}, mrr: {mrr}, hit_k: {hit_k}')

        early_stopping(loss, model)  

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if epoch % 50 == 0:
            model.eval()
            val_hingeLoss, val_sslLoss, val_pred = model.predict('val')
            val_accuracy, val_mrr, val_hit_k = accuracy_at_one_multi_label(val_pred, val_positive_indices)
            print(
                f'val hingeLoss: {val_hingeLoss}, val sslLoss: {val_sslLoss}, val accuracy: {val_accuracy}, val mrr: {val_mrr}, val hit_k: {val_hit_k}')

            test_hingeLoss, test_sslLoss, test_pred = model.predict('test')
            test_accuracy, test_mrr, test_hit_k = accuracy_at_one_multi_label(test_pred, test_positive_indices)
            print(
                f'test hingeLoss: {test_hingeLoss}, test sslLoss: {test_sslLoss}, test accuracy: {test_accuracy}, test mrr: {test_mrr}, test hit_k: {test_hit_k}')

    model.load_state_dict(torch.load(f'{args.model_out_path}/checkpoint_{chunk_id}.pt'))
    model.eval()
    test_hingeLoss, test_sslLoss, test_pred = model.predict('test')
    test_accuracy, test_mrr, test_hit_k = accuracy_at_one_multi_label(test_pred, test_positive_indices)
    print(
        f'test hingeLoss: {test_hingeLoss}, test sslLoss: {test_sslLoss}, test accuracy: {test_accuracy}, test mrr: {test_mrr}, test hit_k: {test_hit_k}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dotCMS_core, eclipse_che, hazelcast_hazelcast, prestodb_presto, wildfly_wildfly
    parser.add_argument('--dataset', type=str, default='prestodb_presto', help='Dataset')
    parser.add_argument('--K', type=int, default=10, help='')
    parser.add_argument('--c', type=int, default=1, help='')
    parser.add_argument('--report', type=int, default=4, help='the weight of report relation')
    parser.add_argument('--comment', type=int, default=3, help='the weight of comment relation')
    parser.add_argument('--create', type=int, default=1, help='the weight of create relation')
    parser.add_argument('--remove', type=int, default=1, help='the weight of remove relation')
    parser.add_argument('--modify', type=float, default=0.5, help='the weight of modify relation')
    parser.add_argument('--similar', type=float, default=0.5, help='the weight of similar relation')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ssl_reg', default=0.1, type=float, help='weight for ssl loss') # 1e-3
    parser.add_argument('--temp', default=0.1, type=float, help='temperature of contrastive learning')
    parser.add_argument('--leaky', default=0.5, type=float, help='slope of leaky relu')
    parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')
    # dotCMS_core, eclipse_che, hazelcast_hazelcast, prestodb_presto, wildfly_wildfly
    parser.add_argument('--model_out_path', default='./data/prestodb_presto/model_out/', help='file name to save model and training record')

    args = parser.parse_args()
    for i in range(9, 10):
        print(f"{args.dataset} remove *create/remove/similar* chunk {i} start---------------------------------------------------------------------------------")
        run(args, i)
        print(f"{args.dataset} remove *create/remove/similar* chunk {i} end-----------------------------------------------------------------------------------")
