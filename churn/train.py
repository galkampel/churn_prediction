import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, auc


class FocalLoss(nn.Module):
    def __init__(self, alpha_lst, gamma=2, device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha_vals = torch.tensor(alpha_lst).to(device)  # class weights
        self.gamma = gamma  # factor that makes model focus on hard examples
        self.device = device
        print('minimizing with focal loss')

    def forward(self, preds, targets):
        bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')  # targets.float() ?
        # alphas = (torch.zeros((preds.shape[0], self.alpha_vals.shape[0])) + self.alpha_vals).to(self.device)
        # print(f'alphas shape = {alphas.shape}')
        alphas = self.alpha_vals[targets.long()]  # targets.int() ?
        # print(f'alphas shape = {alphas.shape}\tpreds shape = {preds.shape}')
        # print(f'bce_loss = {bce_loss}')
        focal_loss = alphas * ((1 - preds) ** self.gamma) * bce_loss
        return focal_loss.mean()


class Trainer:
    def __init__(self, model, optimizer, trainer_name, device='cpu', checkpoint_folder='checkpoint',
                 **loss_params):
        self.model = model  # .to(device) is not needed because of optimizer
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.device = device
        self.criterion = self.set_loss_layer(device, loss_params)
        self.checkpoint_folder = checkpoint_folder
        self.trainer_name = trainer_name

    @staticmethod
    def set_loss_layer(device, params):
        # loss_type = params.get('type', 'bce')
        # print(f'loss type = {loss_type}')
        if params.get('type', 'bce') == 'focal':  # 'bce' or 'focal'
            alpha = params.get('alpha', [0.25, 0.75])  # adjust to true ratio
            gamma = params.get('gamma', 2)
            return FocalLoss(alpha, gamma, device)
        else:
            return nn.BCELoss()

    def train_epoch(self, dataloader):
        self.model.train()
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            # print(f'y shape = {y.shape}')
            preds_prob = self.model(X)
            # print(f'preds shape = {preds_prob.shape}\t y shape = {y.shape}')
            loss = self.criterion(preds_prob.view(-1), y.float())
            loss.backward()
            self.optimizer.step()
    # self.scheduler.step()

    def eval_epoch(self, dataloader, eval_loss=False, eval_f1=False, eval_cm=False, eval_pr_auc=False):
        self.model.eval()
        total_loss = 0.0
        f1, pr_auc = 0.0, 0.0
        cm = None
        num_examples = 0
        preds_prob_lst, targets_lst = [], []
        with torch.no_grad():
            for X, y in dataloader:
                N = X.shape[0]
                num_examples += N
                X, y = X.to(self.device), y.to(self.device)
                preds_prob = self.model(X)
                if eval_loss:
                    loss = self.criterion(preds_prob.view(-1), y.float())
                    total_loss += loss.item() * N
                preds_prob_lst.append(preds_prob.detach().cpu().numpy())
                targets_lst.append(y.detach().cpu().numpy())
                # if eval_metric:
                #     preds = (preds_prob > 0.5).float()
                #     acc = torch.eq(preds, y).sum().item()
                #     total_acc += acc
        total_loss /= num_examples
        preds_prob_arr = np.concatenate(preds_prob_lst)
        preds_arr = np.array((preds_prob_arr > 0.5), dtype=int)
        targets_arr = np.concatenate(targets_lst)
        if eval_f1:
            f1 = f1_score(targets_arr, preds_arr)
        if eval_pr_auc:
            precision, recall, _ = precision_recall_curve(targets_arr, preds_prob_arr)
            pr_auc = auc(recall, precision)
        if eval_cm:
            cm = confusion_matrix(targets_arr, preds_arr, normalize='all')
            # tn, fp, fn, tp   (rows- true label, cols- predicted labels)
        return total_loss, f1, pr_auc, cm

    def save_model(self, epoch):
        checkpoint_folder = os.path.join(os.getcwd(), self.checkpoint_folder)
        os.makedirs(checkpoint_folder, exist_ok=True)
        full_path = os.path.join(checkpoint_folder, f'{self.trainer_name}.pth')
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # 'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch}, full_path)

    def load_model(self, model_name):
        checkpoint_folder = os.path.join(os.getcwd(), self.checkpoint_folder)
        full_path = os.path.join(checkpoint_folder, f'{model_name}.pth')
        checkpoint = torch.load(full_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        return start_epoch
