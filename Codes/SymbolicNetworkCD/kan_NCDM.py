# coding: utf-8
# 2024/5/17 @ Yangzj
import sys 
import os

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score,mean_absolute_error,mean_squared_error
from cdm.CDM import CDM
 
sys.path.append(r'/home/dell/桌面/project/KANCD/cdm/CDM.py')

# from kan import KAN
# from kan import *
from cdm.efficient_kan import KANLinear,KAN

from cdm.ChebyKANLayer import ChebyKANLayer
##分别使用MLP网络和KAN网络
class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.prednet_len3, self.prednet_len4 = 128, 64

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        
        # self.prednet_full1 = KANLinear(self.prednet_input_len, self.prednet_len1)
        # self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.prednet_full1 = KANLinear(self.prednet_input_len, self.prednet_len2)
        # self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len2)
        self.drop_1 = nn.Dropout(p=0.5)
        # self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)

        self.prednet_full2 = KANLinear(self.prednet_len1, self.prednet_len2)        
        self.drop_2 = nn.Dropout(p=0.5)
        # self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        self.prednet_full3 = KANLinear(self.prednet_len2, self.prednet_len3)        
        # self.drop_3 = nn.Dropout(p=0.5)
        # self.prednet_full4 = KANLinear(self.prednet_len3, self.prednet_len4)        
        # self.drop_4 = nn.Dropout(p=0.5)
        # self.prednet_full3 = KANLinear(self.prednet_len4, 1)


        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        input_x = self.drop_3(torch.sigmoid(self.prednet_full3(input_x)))
        # input_x = self.drop_4(torch.sigmoid(self.prednet_full4(input_x)))
        # output_1 = torch.sigmoid(self.prednet_full4(input_x))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class kaNCDM(CDM):
    '''KAN Network Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n):
        super(kaNCDM, self).__init__()
        self.kancdm_net = Net(knowledge_n, exer_n, student_n)
        

    def train(self, train_data, test_data=None, epoch=20, device="cpu", lr=0.001, silence=False):
        self.kancdm_net = self.kancdm_net.to(device)
        self.kancdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.kancdm_net.parameters(), lr=lr)
        epoch_acc = []
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.kancdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            if test_data is not None:
                auc, accuracy,mae,rmse = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f , mae: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy,mae,rmse))
                
                epoch_acc.append(accuracy)
                
            self.save("output/snapshots/{}_{}.snapshot".format('kancdm',epoch_i))
        os.rename("output/snapshots/{}_{}.snapshot".format('kancdm',epoch_acc.index(max(epoch_acc))),'output/snapshots/best_eopch.snapshot')

    def eval(self, test_data, device="cpu"):
        self.kancdm_net = self.kancdm_net.to(device)
        self.kancdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.kancdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5),mean_absolute_error(y_true, np.array(y_pred)),mean_squared_error(y_true,np.array(y_pred))**0.5

    def save(self, filepath):
        torch.save(self.kancdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.kancdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)
