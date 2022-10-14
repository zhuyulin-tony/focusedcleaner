import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from gcn import GCN
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import utils
import math
import scipy as sp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.metrics.cluster import contingency_matrix
from sklearn.decomposition import PCA

class LinkPrediction(Module):
    def __init__(self, nfeat, nhidden, device):
        super().__init__()
        self.nfeat = nfeat
        self.nhidden = nhidden
        self.linear1 = nn.Linear(nfeat, nhidden)
        self.linear2 = nn.Linear(nhidden, nhidden)
        self.device = device
    
    def adj_to_triple(self, adj, node_idx):
        adj_ = adj[node_idx][:,node_idx].clone().detach()
        link_idx = torch.tensor(np.asarray(np.triu_indices(len(adj_), k=1)).T).to(self.device)
        link_weight = adj_[np.triu_indices(len(adj_), k=1)]
        link_idx = torch.cat((torch.tensor(np.sort(node_idx))[link_idx[:,0]].reshape(-1,1).to(self.device), \
                              torch.tensor(np.sort(node_idx))[link_idx[:,1]].reshape(-1,1).to(self.device)),1)
        return link_weight, link_idx
    
    def forward(self, features, link_idx):
        out = torch.relu(self.linear1(features))
        out = self.linear2(out)
        out = (out[link_idx[:,0]] * out[link_idx[:,1]]).sum(1).reshape(-1,1)
        return out
    
    def predict(self, features, u, v):
        with torch.no_grad():
            out = torch.relu(self.linear1(features))
            out = self.linear2(out)
            out = (out[u] * out[v]).sum(0)
        return out

class FocusedCleaner(Module):
    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, temp, eta,
                       device, with_relu=False, with_bias=False, lr=0.1, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.lr = lr
        self.temp = temp
        self.eta = eta
        self.weights = []
        self.device = device
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []
        previous_size = nfeat
        self.LP = LinkPrediction(2*nclass, 2*nclass, device)
        self.LP_optimizer = optim.Adam(self.LP.parameters(), lr=0.1)
        self.hidden_sizes = hidden_sizes
        self.nfeat = nfeat
        self.nclass = nclass
        self.nnodes = nnodes
        self.dropout = dropout
        self.train_iters = train_iters
        self.with_bias = with_bias
        self.with_relu = with_relu
        self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
        self.adj_changes.data.fill_(0)
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            b_velocity = torch.zeros(bias.shape).to(device)
            previous_size = nhid
            self.weights.append(weight)
            self.biases.append(bias)
            self.w_velocities.append(w_velocity)
            self.b_velocities.append(b_velocity)
        output_weight = Parameter(torch.FloatTensor(previous_size, nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        output_b_velocity = torch.zeros(output_bias.shape).to(device)
        self.weights.append(output_weight)
        self.biases.append(output_bias)
        self.w_velocities.append(output_w_velocity)
        self.b_velocities.append(output_b_velocity)
        self._initialize()
        
    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)
    
    def filter_potential_singletons(self, modified_adj):
        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask
    
    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj
        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv
        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat
    
    def train_LinkPrediction(self, features, ori_adj, node_idx):
        link_weight, link_idx = self.LP.adj_to_triple(ori_adj, node_idx)
        adj = ori_adj[node_idx][:,node_idx].clone().detach()
        pos_weight = ((len(node_idx) * len(node_idx) - adj.sum()) / adj.sum()).detach().clone()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        self.LP.train()
        for i in range(50):
            self.LP_optimizer.zero_grad()
            output = self.LP(features, link_idx)
            loss = norm * F.binary_cross_entropy_with_logits(output, link_weight.reshape(-1,1), pos_weight=pos_weight)
            loss.backward()
            self.LP_optimizer.step()
            
        self.LP.eval()
        with torch.no_grad():
            output = torch.sigmoid(self.LP(features, link_idx))
            fpr, tpr, thresholds = roc_curve(link_weight.cpu().data.numpy(), output.cpu().data.numpy())
            gmeans = np.sqrt(tpr * (1-fpr))
            ix = np.argmax(gmeans)
            auc = roc_auc_score(link_weight.cpu().data.numpy(), output.cpu().data.numpy())
            print('link pred auc', np.round(auc,2))
        return thresholds[ix]
    
    def AbnormalDetection(self, features, ori_adj, LP_thresh):
        self.LP.eval()
        with torch.no_grad():
            link_tuple = torch.where(torch.triu(ori_adj, diagonal=1)==1)
            link_idx = torch.cat((link_tuple[0].reshape(-1,1), link_tuple[1].reshape(-1,1)),1)
            probs = self.LP(features, link_idx)
            us = set(link_idx[torch.where(probs <= LP_thresh)[0]].cpu().data.numpy()[:,0])
            vs = set(link_idx[torch.where(probs <= LP_thresh)[0]].cpu().data.numpy()[:,1])
            abnormal_node_idx = list(us.union(vs))
        return abnormal_node_idx
    
    def inner_train(self, features, adj_norm, idx_train, idx_val, idx_test, labels):
        self._initialize()
        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True
            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)
                if ix < len(self.hidden_sizes):
                    hidden = F.dropout(hidden, self.dropout, training=True)

            output = F.log_softmax(hidden, dim=1)
            softmax = F.softmax(hidden/self.temp, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return hidden, softmax, labels_self_training
    
    def get_meta_grad(self, features, adj_norm, idx_train, idx_val, idx_test, labels, labels_self_training, smoother, normal_node_idx):
        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu:
                hidden = F.relu(hidden)
        output = F.log_softmax(hidden, dim=1)
        loss_test = F.nll_loss(output[list(set(idx_test).intersection(set(normal_node_idx)))],\
                 labels_self_training[list(set(idx_test).intersection(set(normal_node_idx)))])
        loss_val = F.nll_loss(output[list(set(idx_val).intersection(set(normal_node_idx)))],\
                              labels[list(set(idx_val).intersection(set(normal_node_idx)))])
        cleaner_loss = self.lambda_ * loss_val + (1-self.lambda_) * loss_test + self.eta * smoother
        'retain_graph=True or False has the same results.'
        adj_grad = torch.autograd.grad(cleaner_loss, self.adj_changes, retain_graph=False)[0]
        return adj_grad
    
    def forward(self, features, ori_adj, adj_clean, labels, idx_train, idx_val, idx_test, perturbations):
        self.sparse_features = sp.sparse.issparse(features)
        self.adversarial_detection_auc = []
        self.FNR = []
        self.gcn_acc = []
        self.smoother_lst = []
        all_node_set = set(np.arange(self.nnodes))
        self.lambda_lst = np.linspace(1., 0., perturbations)
        for i in tqdm(range(perturbations), desc="Sanitizing"):
            self.lambda_ = self.lambda_lst[i]
            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
            modified_adj = adj_changes_symm + ori_adj
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            'inner training.'
            gcn_feature, softmax, labels_self_training = self.inner_train(features, adj_norm, idx_train, idx_val, idx_test, labels)
            'get low-dimensional attributes.'
            attr_dr = torch.from_numpy(PCA(n_components=self.nclass, random_state=6).fit_transform(features.cpu().data.numpy())).to(self.device).float()
            feat_smoother = self.feature_smoothing(modified_adj, features)
            self.smoother_lst.append(feat_smoother.item())
            'train link prediction.'
            LP_thresh = self.train_LinkPrediction(torch.cat((gcn_feature.clone().detach(), attr_dr),1), modified_adj, list(all_node_set))
            'victim node detection.'
            abnormal_node_idx = self.AbnormalDetection(torch.cat((gcn_feature.clone().detach(), attr_dr),1), modified_adj, LP_thresh)
            normal_node_idx = list(all_node_set.difference(set(abnormal_node_idx)))
            
            ano_pred = np.ones((self.nnodes))
            ano_pred[normal_node_idx] = 0
            'anomaly detection true label for each sanitation step.'
            true_victim_nodes = list((set(np.where(np.triu(modified_adj.cpu().data.numpy()) != np.triu(np.array(adj_clean.todense())))[0]).union(\
                                      set(np.where(np.triu(modified_adj.cpu().data.numpy()) != np.triu(np.array(adj_clean.todense())))[1]))))
            ano_label = np.zeros((self.nnodes))
            ano_label[true_victim_nodes] = 1
            acc = accuracy_score(labels.cpu().data.numpy()[idx_test], labels_self_training.cpu().data.numpy()[idx_test])
            self.gcn_acc.append(acc)
            self.adversarial_detection_auc.append(roc_auc_score(ano_label, ano_pred))
            print('anomaly auc', np.round(roc_auc_score(ano_label, ano_pred),2))
            self.FNR.append((contingency_matrix(ano_label, ano_pred)[1,0])/self.nnodes)
            'only manipulate on links connected with adversarial nodes.'
            adj_grad = self.get_meta_grad(features, adj_norm, idx_train, idx_val, idx_test, labels, labels_self_training, feat_smoother, normal_node_idx)
            adj_grad[normal_node_idx,:] = 0.
            adj_grad[:,normal_node_idx] = 0.
            adj_grad.fill_diagonal_(0.)
            'only delete edges.'
            adj_grad = adj_grad * (adj_grad > 0)
            with torch.no_grad():
                'graph sanitization.'
                adj_meta_grad = -adj_grad * (-2 * modified_adj + 1)
                adj_meta_grad -= adj_meta_grad.min()
                adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
                singleton_mask = self.filter_potential_singletons(modified_adj)
                adj_meta_grad = adj_meta_grad * singleton_mask
                'Get argmax of the meta gradients.'
                adj_meta_argmax = torch.argmax(adj_meta_grad)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
        self.adj_changes.data.fill_diagonal_(0.)
        return self.adj_changes + ori_adj