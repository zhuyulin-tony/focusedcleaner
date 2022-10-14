import torch
import os
from utils import load_data, preprocess
import argparse
import numpy as np
from FocusedCleaner_CLD import FocusedCleaner
#from FocusedCleaner_LP import FocusedCleaner
import scipy as sp

for i in range(10):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=i, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--dropout', type=float, default=0., help='dropout.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--eta', type=float, default=1e-4, choices=[0, 1e-4], help='attribute smoother penalty.')
    parser.add_argument('--tau', type=float, default=0.6, choices=[0.6], help='tau.')
    parser.add_argument('--beta', type=float, default=0.3, choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], help='beta.')
    parser.add_argument('--temperature', type=float, default=2, help='soft class probability temperature.')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--atk', type=str, default='mettack', choices=['mettack', 'nettack'], help='attack.')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'polblogs'], help='dataset.')
    parser.add_argument('--ptb_rate', type=float, default=0.1, choices=[0.05,0.1,0.15,0.2,0.25],  help='ptb rate.')
    parser.add_argument('--n_ptb', type=float, default=3, choices=[1,2,3,4,5],  help='pertubation/node')
    parser.add_argument('--san_rate', type=float, default=0.1, choices=[0.01,0.05,0.1,0.15,0.2],  help='sanitation ratio.')
    
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    root_dir = '/'
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)
    
    'loading dataset'
    adj, features, labels = load_data(dataset=args.dataset)
    if args.atk == 'mettack':
        adj_mod = sp.sparse.csr_matrix(np.loadtxt(root_dir+'/'+args.dataset+'/adj_mettack_'+str(args.ptb_rate)+'.txt'))
    elif args.atk == 'nettack':
        adj_mod = sp.sparse.csr_matrix(np.loadtxt(root_dir+'/'+args.dataset+'/adj_nettack_'+str(args.n_ptb)+'.txt'))
        
    nclass = max(labels) + 1
    val_size = 0.1
    test_size = 0.8
    train_size = 1 - test_size - val_size
    
    idx = np.arange(adj_mod.shape[0])
    idx_train = np.loadtxt(root_dir+'/'+args.dataset+'/idx_train.txt', dtype='int')
    idx_val = np.loadtxt(root_dir+'/'+args.dataset+'/idx_val.txt', dtype='int')
    idx_test = np.loadtxt(root_dir+'/'+args.dataset+'/idx_test.txt', dtype='int')
    #idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
    idx_unlabeled = np.union1d(idx_val, idx_test)
    perturbations = int(args.san_rate * (adj_mod.sum()//2))
    
    adj_mod, features, labels = preprocess(adj_mod, features, labels, preprocess_adj=False)
    
    'set up attack model'
    'FocusedCleaner-CLD'
    model = FocusedCleaner(nfeat=features.shape[1], hidden_sizes=[args.hidden],
                           nnodes=adj_mod.shape[0], nclass=nclass, dropout=args.dropout, train_iters=args.epochs, beta=args.beta, tau = args.tau,
                           with_relu=False, with_bias=False, lr=args.lr, momentum=0.9, eta=args.eta, temp=args.temperature, device=device)
    
    'FocusedCleaner-LP'
    #model = FocusedCleaner(nfeat=features.shape[1], hidden_sizes=[args.hidden],
    #                       nnodes=adj_mod.shape[0], nclass=nclass, dropout=args.dropout, train_iters=args.epochs,
    #                       with_relu=False, with_bias=False, lr=args.lr, momentum=0.9, eta=args.eta, temp=args.temperature, device=device)
    
    if device != 'cpu':
        adj_mod = adj_mod.to(device)
        features = features.to(device)
        labels = labels.to(device)
        model = model.to(device)
    
    sanitize_adj = model(features, adj_mod, adj, labels, idx_train, idx_val, idx_test, perturbations)
    sanitize_adj = sanitize_adj.detach().cpu().data.numpy()
    
    if args.atk == 'mettack':
        store_dir = root_dir+'/'+args.dataset+'/sanitate_mettack/'+str(i)
    elif args.atk == 'nettack':
        store_dir = root_dir+'/'+args.dataset+'/sanitate_nettack/'+str(i)
        
    try:
        os.makedirs(store_dir)
    except:
        pass
    
    if args.atk == 'mettack':
        np.savetxt(store_dir+'/adj_FocusedCleaner_CLD_'+str(args.ptb_rate)+'_'+str(args.rec_rate)+'.txt', sanitize_adj, fmt='%d')
        #np.savetxt(store_dir+'/adj_FocusedCleaner_LP_'+str(args.ptb_rate)+'_'+str(args.rec_rate)+'.txt', sanitize_adj, fmt='%d')
    elif args.atk == 'nettack':
        np.savetxt(store_dir+'/adj_FocusedCleaner_CLD_'+str(args.n_ptb)+'_'+str(args.rec_rate)+'.txt', sanitize_adj, fmt='%d')
        #np.savetxt(store_dir+'/adj_Focusedcleaner_LP_'+str(args.n_ptb)+'_'+str(args.rec_rate)+'.txt', sanitize_adj, fmt='%d')