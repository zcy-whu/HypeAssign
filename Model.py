import torch as t
from torch import nn
import torch.nn.functional as F
from Node import Node

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
class Model(nn.Module):
    def __init__(self, graph, issue_num, dev_num, file_num, latdim, hyperNum, layer_num, temp, leaky,
                 train_data, val_data, test_data, train_positive_indices, val_positive_indices, test_positive_indices):
        super(Model, self).__init__()

        self.graph = graph
        self.issue_num = issue_num
        self.dev_num = dev_num
        self.file_num = file_num
        self.latdim = latdim
        self.hyperNum = hyperNum
        self.gnn_layer = layer_num
        self.leaky = leaky

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_positive_indices = train_positive_indices
        self.val_positive_indices = val_positive_indices
        self.test_positive_indices = test_positive_indices

        self.iEmbeds = nn.Parameter(init(t.empty(issue_num, latdim)))
        self.dEmbeds = nn.Parameter(init(t.empty(dev_num, latdim)))
        self.fEmbeds = nn.Parameter(init(t.empty(file_num, latdim)))
        self.gcnLayer = GCNLayer(self.leaky)
        self.hgnnLayer = HGNNLayer(self.leaky)
        self.iHyper = nn.Parameter(init(t.empty(latdim, hyperNum)))  
        self.dHyper = nn.Parameter(init(t.empty(latdim, hyperNum)))
        self.fHyper = nn.Parameter(init(t.empty(latdim, hyperNum)))

        self.fc1 = nn.Linear(latdim * 2, latdim)
        self.fc2 = nn.Linear(latdim, 1)
        self.loss_function = MultiLabelRankingLoss()
        self.temp = temp
        self.keepRate = 1.0

    def forward(self, adj, keepRate):
        embeds = t.concat([self.iEmbeds, self.dEmbeds, self.fEmbeds], dim=0)  
        lats = [embeds]
        gnnLats = []
        hyperLats = []
        iiHyper = self.iEmbeds @ self.iHyper  
        ddHyper = self.dEmbeds @ self.dHyper
        ffHyper = self.fEmbeds @ self.fHyper

        for i in range(self.gnn_layer):
            temEmbeds = self.gcnLayer(adj, lats[-1])  
            hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1 - keepRate), lats[-1][:self.issue_num])  
            hyperDLat = self.hgnnLayer(F.dropout(ddHyper, p=1 - keepRate), lats[-1][self.issue_num:(self.issue_num + self.dev_num)])  
            hyperFLat = self.hgnnLayer(F.dropout(ffHyper, p=1 - keepRate), lats[-1][-self.file_num:]) 
            gnnLats.append(temEmbeds)
            hyperLats.append(t.concat([hyperILat, hyperDLat, hyperFLat], dim=0))
            lats.append(temEmbeds + hyperLats[-1])
        embeds = sum(lats)
        return embeds, gnnLats, hyperLats

    def calcLosses(self):
        adj = t.tensor(self.graph.A, dtype=t.float32)
        embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, self.keepRate)

        iEmbeds, dEmbeds, fEmbeds = embeds[:self.issue_num], embeds[self.issue_num:(self.issue_num + self.dev_num)], embeds[-self.file_num:]
        score_matrix = t.mm(iEmbeds, dEmbeds.T)

        hingeLoss = self.loss_function(score_matrix[:len(self.train_data)], self.train_positive_indices, self.dev_num)
        hingeLoss_val = self.loss_function(score_matrix[len(self.train_data):len(self.train_data)+len(self.val_data)], self.val_positive_indices, self.dev_num)
        hingeLoss_test = self.loss_function(score_matrix[-len(self.test_data):], self.test_positive_indices, self.dev_num)


        sslLoss = 0
        for i in range(self.gnn_layer):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += (contrastLoss(embeds1[:self.issue_num], embeds2[:self.issue_num], self.temp) + contrastLoss(
                embeds1[self.issue_num:(self.issue_num + self.dev_num)], embeds2[self.issue_num:(self.issue_num + self.dev_num)], self.temp)
                        + contrastLoss(embeds1[-self.file_num:], embeds2[-self.file_num:], self.temp))


        return hingeLoss, sslLoss, score_matrix[:len(self.train_data)], hingeLoss_val, hingeLoss_test

    def predict(self, mode):
        with t.no_grad():
            adj = t.tensor(self.graph.A, dtype=t.float32)
            embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, 1.0)
            iEmbeds, dEmbeds, fEmbeds = embeds[:self.issue_num], embeds[self.issue_num:(self.issue_num + self.dev_num)], embeds[-self.file_num:]
            score_matrix = t.mm(iEmbeds, dEmbeds.T)
            if mode == 'val':
                pred_score = score_matrix[len(self.train_data):len(self.train_data) + len(self.val_data)]
                hingeLoss = self.loss_function(pred_score, self.val_positive_indices, self.dev_num)
            elif mode == 'test':
                pred_score = score_matrix[-len(self.test_data):]
                hingeLoss = self.loss_function(pred_score, self.test_positive_indices, self.dev_num)

            sslLoss = 0
            for i in range(self.gnn_layer):
                embeds1 = gcnEmbedsLst[i].detach()
                embeds2 = hyperEmbedsLst[i]
                sslLoss += (contrastLoss(embeds1[:self.issue_num], embeds2[:self.issue_num], self.temp) + contrastLoss(
                    embeds1[self.issue_num:(self.issue_num + self.dev_num)], embeds2[self.issue_num:(self.issue_num + self.dev_num)], self.temp)
                            + contrastLoss(embeds1[-self.file_num:], embeds2[-self.file_num:], self.temp))

            return hingeLoss, sslLoss, pred_score

class GCNLayer(nn.Module):
    def __init__(self, leaky):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, adj, embeds):
        return self.act(t.spmm(adj, embeds))


class HGNNLayer(nn.Module):
    def __init__(self, leaky):
        super(HGNNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, adj, embeds):
        lat = self.act(adj.T @ embeds)
        ret = self.act(adj @ lat)
        return ret


class MultiLabelRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MultiLabelRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, positive_indices, num_developers):
        assert scores.size(0) == len(positive_indices)

        loss = 0.0
        for i, pos_idxs in enumerate(positive_indices):
            neg_idxs = t.tensor([j for j in range(num_developers) if j not in pos_idxs], dtype=t.long, device=scores.device)

            pos_scores = scores[i, pos_idxs]
            neg_scores = scores[i, neg_idxs]

            hinge_losses = F.relu(neg_scores.unsqueeze(1) - pos_scores.unsqueeze(0) + self.margin)
            hinge_losses = hinge_losses.view(-1) 
            loss += hinge_losses.sum()

        num_pos_neg_pairs = sum(len(pos_idxs) * (num_developers - len(pos_idxs)) for pos_idxs in positive_indices)
        if num_pos_neg_pairs == 0:  
            return t.tensor(0.0, device=scores.device)
        loss /= num_pos_neg_pairs
        return loss


def contrastLoss(embeds1, embeds2, temp):
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    nume = t.exp(t.sum(embeds1 * embeds2, dim=-1) / temp)
    deno = t.exp(embeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return -t.log(nume / deno).mean()