""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os
from networkx.algorithms.components.connected import connected_components
from tensorboardX import SummaryWriter

import sys
import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import torch.nn.functional as F
from torch import nn, optim
from gae.model import VGAE3MLP
from gae.optimizer import loss_function as gae_loss
from scipy.stats import zscore

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'gnnexp'))

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain


decimal_round = lambda x: round(x, 5)
color_map = ['gray', 'blue', 'purple', 'red', 'brown', 'green', 'orange', 'olive']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Mutagenicity', help='Name of dataset.')
parser.add_argument('--output', type=str, default=None, help='output path.')

parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--lr_sampler', type=float, default=0.003, help='Initial learning rate.')
parser.add_argument('-e', '--epoch', type=int, default=300, help='Number of training epochs.')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of samples in a minibatch.')
parser.add_argument('--seed', type=int, default=42, help='Number of training epochs.')
parser.add_argument('--max_grad_norm', type=float, default=1, help='max_grad_norm.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--encoder_hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--encoder_hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--encoder_output', type=int, default=16, help='Dim of output of VGAE encoder.')
parser.add_argument('--decoder_hidden1', type=int, default=16, help='Number of units in decoder hidden layer 1.')
parser.add_argument('--decoder_hidden2', type=int, default=16, help='Number of units in decoder  hidden layer 2.')
parser.add_argument('--mlp_hidden', type=int, default=32, help='Number of units in mlp hidden layer.')
parser.add_argument('--K', type=int, default=16, help='Number of casual factors.')
parser.add_argument('--coef_lambda', type=float, default=1.0, help='Coefficient of gae loss.')
parser.add_argument('--coef_kl', type=float, default=10.0, help='Coefficient of gae loss.')
parser.add_argument('--coef_size', type=float, default=0.1, help='Coefficient of size loss.')
parser.add_argument('--sparsity', type=float, default=0.6, help='Coefficient of size loss.')
parser.add_argument('--tau', type=float, default=1, help='Tau of gumbel softmax.')
parser.add_argument('--NX', type=int, default=1, help='Number of monte-carlo samples per causal factor.')
parser.add_argument('--NA', type=int, default=1, help='Number of monte-carlo samples per causal factor.')
parser.add_argument('--Nalpha', type=int, default=25, help='Number of monte-carlo samples per causal factor.')
parser.add_argument('--Nbeta', type=int, default=100, help='Number of monte-carlo samples per noncausal factor.')
parser.add_argument('--node_perm', action="store_true", help='Use node permutation as data augmentation for causal training.')
parser.add_argument('--load_ckpt', default=None, help='Load parameters from checkpoint.')
parser.add_argument('--gpu',default=0 , action='store_true')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--patient', type=int, default=100, help='Patient for early stopping.')
parser.add_argument('--plot_info_flow', action='store_true')

args = parser.parse_args()
args.retrain = True
args.load_ckpt = "explanation/%s/model.ckpt' % args.output"

if torch.cuda.is_available():
    print("Use cuda")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

def graph_labeling(G):
    for node in G:
        G.nodes[node]['string'] = 1
    old_strings = tuple([G.nodes[node]['string'] for node in G])
    for iter_num in range(100):
        for node in G:
            string = sorted([G.nodes[neigh]['string'] for neigh in G.neighbors(node)])
            G.nodes[node]['concat_string'] =  tuple([G.nodes[node]['string']] + string)
        d = nx.get_node_attributes(G,'concat_string')
        nodes,strings = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
        map_string = dict([[string, i+1] for i, string in enumerate(sorted(set(strings)))])
        for node in nodes:
            G.nodes[node]['string'] = map_string[G.nodes[node]['concat_string']]
        new_strings = tuple([G.nodes[node]['string'] for node in G])
        if old_strings == new_strings:
            break
        else:
            old_strings = new_strings
    return G

def gaeloss(x,mu,logvar,data):
    return gae_loss(preds=x, labels=data['adj_label'],
                    mu=mu, logvar=logvar, n_nodes=data['n_nodes'],
                    norm=data['norm'], pos_weight=data['pos_weight'])

softmax = torch.nn.Softmax(dim=1)
ce = torch.nn.CrossEntropyLoss(reduction='mean')

def main():
    # Load a model checkpoint
    ckpt = torch.load('ckpt/%s_base_h20_o20.pth.tar'%(args.dataset))
    cg_dict = ckpt["cg"] # get computation graph
    input_dim = cg_dict["feat"].shape[2] 
    num_classes = cg_dict["pred"].shape[2]
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # Explain Graph prediction
    classifier = models.GcnEncoderGraph(
        input_dim=input_dim,
        hidden_dim=20,
        embedding_dim=20,
        label_dim=num_classes,
        num_layers=3,
        bn=False,
        args=argparse.Namespace(gpu=args.gpu,bias=True,method=None),
    ).to(device)

    class NodeSampler(torch.nn.Module):
        def __init__(self,num_i,num_o):
            super(NodeSampler,self).__init__()
            self.linear1=torch.nn.Linear(num_i,num_o)
            #self.relu=torch.nn.Tanh()
            #self.linear2=torch.nn.Linear(num_h,num_o)
        def forward(self, x):
            x = self.linear1(x)
            #x = self.relu(x)
            #x = self.linear2(x)
            return x

    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    classifier.load_state_dict(ckpt["model_state"])
    classifier.eval()
    print("Number of graphs:", cg_dict["adj"].shape[0])
    if args.output is None:
        args.output = args.dataset

    K = args.K
    L = args.encoder_output - K
    ceparams = {
        'Nalpha': args.Nalpha,
        'Nbeta' : args.Nbeta,
        'K'     : K,
        'L'     : L,
        'z_dim' : args.encoder_output,
        'M'     : num_classes}

    model = VGAE3MLP(
        input_dim + 100, args.encoder_hidden1, args.encoder_hidden1,
        args.encoder_output, args.decoder_hidden1, args.decoder_hidden2,
        args.K, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model_sampler = NodeSampler(args.encoder_output,2).to(device)
    optimizer_sampler = optim.Adam(model_sampler.parameters(), lr=args.lr_sampler)
    criterion = gaeloss
    label_onehot = torch.eye(100, dtype=torch.float)
    class GraphSampler(torch.utils.data.Dataset):
        """ Sample graphs and nodes in graph
        """
        def __init__(
            self,
            graph_idxs
        ):
            self.graph_idxs = graph_idxs
            self.graph_data = []
            for graph_idx in graph_idxs:
                adj = cg_dict["adj"][graph_idx].float()
                label = cg_dict["label"][graph_idx].long()
                feat = cg_dict["feat"][graph_idx, :].float()
                G = graph_labeling(nx.from_numpy_array(cg_dict["adj"][graph_idx].numpy()))
                graph_label = np.array([G.nodes[node]['string'] for node in G])
                graph_label_onehot = label_onehot[graph_label]
                sub_feat = torch.cat((feat, graph_label_onehot), dim=1)
                preds_gcn = classifier(feat.unsqueeze(dim = 0).to(device), adj.unsqueeze(dim = 0).to(device))[0]
                adj_label = adj + np.eye(adj.shape[0])
                n_nodes = adj.shape[0]
                graph_size = torch.count_nonzero(adj.sum(-1))
                pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
                pos_weight = torch.from_numpy(np.array(pos_weight))
                norm = torch.tensor(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))
                self.graph_data += [{
                    "graph_idx": graph_idx,
                    "graph_size": graph_size, 
                    "sub_adj": adj.to(device), 
                    "feat": feat.to(device).float(), 
                    "sub_feat": sub_feat.to(device).float(), 
                    "sub_label": label.to(device).float(), 
                    "sub_preds": preds_gcn.squeeze(dim=0),
                    "adj_label": adj_label.to(device).float(),
                    "n_nodes": torch.Tensor([n_nodes])[0].to(device),
                    "pos_weight": pos_weight.to(device),
                    "norm": norm.to(device)
                }]

        def __len__(self):
            return len(self.graph_idxs)

        def __getitem__(self, idx):
            return self.graph_data[idx]

    train_idxs = np.array(cg_dict['train_idx'])
    val_idxs = np.array(cg_dict['val_idx'])
    test_idxs = np.array(cg_dict['test_idx'])
    train_graphs = GraphSampler(train_idxs)
    train_dataset = torch.utils.data.DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_graphs = GraphSampler(val_idxs)
    val_dataset = torch.utils.data.DataLoader(
        val_graphs,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )
    test_graphs = GraphSampler(test_idxs)
    test_dataset = torch.utils.data.DataLoader(
        test_graphs,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )

    def eval_model(dataset, prefix='eval'):
        model.eval()
        model_sampler.eval()
        with torch.no_grad():
            for data in dataset:
                labels = cg_dict['label'][data['graph_idx'].long()].long().to(device)
                org_probs = F.softmax(data['sub_preds'], dim=1)
                recovered, mu, logvar = model(data['sub_feat'], data['sub_adj'])
                recovered_adj = torch.sigmoid(recovered)
                masked_recovered_adj = recovered_adj * data['sub_adj']
                recovered_logits = classifier(data['feat'], masked_recovered_adj)[0]
                recovered_probs = F.softmax(recovered_logits, dim=1)

                #node sampling
                alpha_mu = torch.zeros_like(mu)
                std = torch.exp(logvar)
                eps = torch.randn_like(std)
                alpha_mu = eps.mul(std).add_(mu)
                z = model_sampler(alpha_mu)
                z = torch.softmax(z,dim = -1)
                z = F.gumbel_softmax(z,tau=args.tau,hard=True)
                masked_feat = data['feat']*(z[:, :, 1].unsqueeze(-1))
                masked_logit = classifier(masked_feat, data['sub_adj'])[0]
                sparsity_z = torch.sum(z[:, :, 1],dim=-1)/data['sub_adj'].shape[1]

                #loss
                nll_loss =  criterion(recovered, mu, logvar, data).mean()
                klloss = F.kl_div(F.log_softmax(masked_logit, dim=1), org_probs, reduction='mean')
                distance_sparsity = torch.abs(args.sparsity - sparsity_z)

            pred_labels = torch.argmax(org_probs,axis=1)
            recover_acc = (torch.argmax(org_probs,axis=1) == torch.argmax(recovered_probs,axis=1)).float().mean()
            pred_acc = (torch.argmax(F.softmax(masked_logit, dim=1),axis=1) == pred_labels).float().mean()

            loss = args.coef_lambda * nll_loss + \
                args.coef_kl * klloss + \
                args.coef_size * distance_sparsity.mean()
            writer.add_scalar("%s/total_loss"%prefix, loss, epoch)
            writer.add_scalar("%s/recover_acc"%prefix, recover_acc, epoch)
            writer.add_scalar("%s/pred_acc"%prefix, pred_acc, epoch)
            writer.add_scalar("%s/sparsity"%prefix, distance_sparsity.mean(), epoch)
        return loss.item()

    def save_checkpoint(filename):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_sampler': model_sampler.state_dict(),
            'optimizer_sampler': optimizer_sampler.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch
        }, filename)

    if args.load_ckpt:
        ckpt_path = args.load_ckpt
    else:
        ckpt_path = os.path.join('explanation', args.output, 'model.ckpt')
    if os.path.exists(ckpt_path) and not args.retrain:
        print("Load checkpoint from {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model_sampler.load_state_dict(checkpoint['model_sampler'])
        optimizer_sampler.load_state_dict(checkpoint['optimizer_sampler'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
    else:
        #args.retrain = True
        start_epoch = 1
        best_loss = 100
    if args.retrain:
        patient = args.patient
        model.train()
        model_sampler.train()
        start_time = time.time()
        writer = SummaryWriter(comment=args.output)
        os.makedirs('explanation/%s' % args.output, exist_ok=True)
        #train VGAE
        for epoch in tqdm(range(start_epoch, args.epoch+1)):
            # print("------- Epoch %2d ------" % epoch)
            model.train()
            train_losses = []
            for batch_idx, data in enumerate(train_dataset):
                optimizer.zero_grad()

                mu, logvar = model.encode(data['sub_feat'], data['sub_adj'])
                sample_mu = model.reparameterize(mu, logvar)
                recovered = model.dc(sample_mu)
                loss = args.coef_lambda * criterion(recovered, mu, logvar, data).mean()

                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_losses += [[loss]]
                sys.stdout.flush()
            
            # train_loss = (torch.cat(train_losses)).mean().item()
            nll_loss= torch.tensor(train_losses).mean(0)
            writer.add_scalar("train/nll", nll_loss, epoch)

        for name, parms in model.named_parameters():
            parms.requires_grad = False

        #train MLP
        for epoch in tqdm(range(start_epoch, args.epoch+1)):
            # print("------- Epoch %2d ------" % epoch)
            model_sampler.train()
            train_losses = []
            for batch_idx, data in enumerate(train_dataset):
                optimizer_sampler.zero_grad()
                org_probs = F.softmax(data['sub_preds'], dim=1)
                mu, logvar = model.encode(data['sub_feat'], data['sub_adj'])
                sample_mu = model.reparameterize(mu, logvar)
                #node sampling
                z = model_sampler(sample_mu)
                z = torch.softmax(z,dim = -1)
                z = F.gumbel_softmax(z,tau=args.tau,hard=True)
                masked_feat = data['feat']*(z[:, :, 1].unsqueeze(-1))
                masked_logit = classifier(masked_feat, data['sub_adj'])[0]
                sparsity_z = torch.sum(z[:, :, 1],dim=-1)/data['sub_adj'].shape[1]

                if args.coef_kl:
                    klloss = args.coef_kl * F.kl_div(F.log_softmax(masked_logit,dim=1), org_probs, reduction='batchmean')
                else:
                    klloss = 0
                if args.coef_size:
                    distance_sparsity = torch.abs(args.sparsity - sparsity_z)
                    size_loss = args.coef_size * distance_sparsity.mean()
                else:
                    size_loss = 0

                loss = klloss + size_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model_sampler.parameters(), args.max_grad_norm)
                optimizer_sampler.step()
                train_losses += [[klloss, size_loss,loss]]
                sys.stdout.flush()
            
            # train_loss = (torch.cat(train_losses)).mean().item()
            klloss, size_loss,loss = torch.tensor(train_losses).mean(0)
            writer.add_scalar("train/kl_loss", klloss, epoch)
            writer.add_scalar("train/size_loss", size_loss, epoch)
            writer.add_scalar("train/total_loss", loss, epoch)

            # val_loss = eval_model(val_dataset, 'val')
            # patient -= 1
            # if val_loss < best_loss:
            #     best_loss = val_loss
            #     save_checkpoint('explanation/%s/model.ckpt' % args.output)
            #     test_loss = eval_model(test_dataset, 'test')
            #     patient = 100
            # elif patient <= 0:
            #     print("Early stopping!")
            #     break
            # if epoch % 100 == 0:
        # save_checkpoint('explanation/%s/model-%depoch.ckpt' % (args.output,epoch))
        # print("Train time:", time.time() - start_time)
        # writer.close()
        # checkpoint = torch.load('explanation/%s/model.ckpt' % args.output)
        # model.load_state_dict(checkpoint['model'])
        # model_sampler.load_state_dict(checkpoint['model_sampler'])
        # optimizer_sampler.load_state_dict(checkpoint['optimizer_sampler'])

    print("Start evaluation.")

    model.eval()
    model_sampler.eval()
    results = []
    with torch.no_grad():
        for data in test_dataset:
            labels = cg_dict['label'][data['graph_idx'].long()].long().to(device)
            mu, logvar = model.encode(data['sub_feat'], data['sub_adj'])
            org_probs = F.softmax(data['sub_preds'], dim=1)
            pred_labels = torch.argmax(org_probs,axis=1)
            alpha_mu = torch.zeros_like(mu)
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            alpha_mu = eps.mul(std).add_(mu)
            #VGAE Recovery
            alpha_adj = torch.sigmoid(model.dc(alpha_mu))
            masked_alpha_adj = alpha_adj * data['sub_adj']
            recovered_logits = classifier(data['feat'], masked_alpha_adj)[0]
            recovered_probs = F.softmax(recovered_logits, dim=1)

            #node sampling
            z = model_sampler(alpha_mu)
            z = torch.softmax(z,dim = -1)
            z = F.gumbel_softmax(z,tau=args.tau,hard=True)
            masked_feat = data['feat']*(z[:, :, 1].unsqueeze(-1))
            masked_logit = classifier(masked_feat, data['sub_adj'])[0]
            sparsity_z = torch.sum(z[:, :, 1],dim=-1)/data['sub_adj'].shape[1]

        print("---Final Result---")
        print("VGAE Recovery Accuracy:",\
              (pred_labels == torch.argmax(recovered_probs,axis=1)).float().mean())
        print("Subgraph Prediction Accuracy:",\
              (pred_labels == torch.argmax(F.softmax(masked_logit, dim=1),axis=1)).float().mean())
        print("Subgraph Ground Truth Accuracy:",\
              (labels == torch.argmax(F.softmax(masked_logit, dim=1),axis=1)).float().mean())
        print("Sparsity of mask:",sparsity_z.mean())
        print("Node Sparsity:")

if __name__ == "__main__":
    main()

