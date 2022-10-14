import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.linalg.cholesky(a)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

class DGMM(nn.Module):
    def __init__(self, n_gmm, z_dim, hidden_dim, lambda_cov, dropout, device):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_gmm)
        self.n_gmm = n_gmm
        self.lambda_cov = lambda_cov
        self.device = device
        self.dropout = dropout
    
    def initialize_(self):
        stdv1 = 1. / math.sqrt(self.fc1.weight.size(1))
        stdv2 = 1. / math.sqrt(self.fc2.weight.size(1))
        self.fc1.weight.data.uniform_(-stdv1, stdv1)
        self.fc2.weight.data.uniform_(-stdv2, stdv2)
        self.fc1.bias.data.uniform_(-stdv1, stdv1)
        self.fc2.bias.data.uniform_(-stdv2, stdv2)
        
    def estimate(self, z):
        h = torch.relu(self.fc1(z))
        return F.softmax(self.fc2(h), dim=1)
    
    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)
        return phi, mu, cov
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        eps = 1e-20
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)
        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag
    
    def forward(self, z, gamma):
        """Computing the loss function for DAGMM."""
        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = sample_energy + self.lambda_cov * cov_diag
        return loss