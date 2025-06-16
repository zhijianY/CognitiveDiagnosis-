# 学习者作答习题时的得分不仅仅受到知识点的影响，还受到学习者的其他技能的影响
# 组织形式：Y=(1-λ)*Y_A + λ*Y_B    
#加入学校特征后: Y = （1-λ）*Y_A +λ*Y_B

# A：学习者对知识点的熟练程度，大小=N*K
# B：学习者除知识点外的其他技能，大小=N*8
# C：学习者在知识簇上的属性，大小=N*K
# H：知识点的交互，大小=K*K
# W：习题与知识点的权重矩阵，大小=J*K，其中元素sigmoid函数后再除以行/列累加和归一化
# D：习题与其他技能的权重，大小=J*8，其中元素用行/列softmax函数归一化
# lambda：其他技能对答题记录的影响权重

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold
from torch import nn, Tensor
from typing import Union, Tuple, Optional
from torch.utils.data import TensorDataset, DataLoader
from initial_dataSet4 import DataSet
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def format_data(record, n_splits=5):
    train = [[], [], [],[]]  # 学生,习题，得分,学校
    label = [[], [], [],[]]  # 学生,习题，得分,学校
    stu_list = set(record.index)

    KF = KFold(n_splits=n_splits, shuffle=True)  # 5折交叉验证
    count = 0
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1
        stu_scohol = record.loc[[stu], 'school_id'].values - 1
        stu_score = record.loc[[stu], 'score'].values
        if len(stu_item) >= n_splits:

            for train_prob, label_prob in KF.split(stu_item):
                train[0].append(stu - 1)
                train[1].append(stu_item[train_prob])
                train[2].append(stu_score[train_prob])
                train[3].append(stu_scohol[train_prob])

                label[0].extend([count] * len(label_prob))
                label[1].extend(stu_item[label_prob])
                label[2].extend(stu_score[label_prob])
                label[3].extend(stu_scohol[label_prob])
                count += 1
    return train, label


def format_test_data(record, test_record):
    train = [[], [], [],[]]  # 学生,习题，得分
    test = [[], [], [],[]]  # 学生,习题，得分
    stu_list = set(record.index)

    count = 0
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1
        stu_score = record.loc[[stu], 'score'].values
        stu_school = record.loc[[stu], 'school_id'].values -1

        test_item = test_record.loc[[stu], 'item_id'].values - 1
        test_score = test_record.loc[[stu], 'score'].values
        test_school = test_record.loc[[stu], 'school_id'].values -1

        train[0].append(stu - 1)
        train[1].append(stu_item)
        train[2].append(stu_score)
        train[3].append(stu_school)

        test[0].extend([count] * len(test_item))
        test[1].extend(test_item)
        test[2].extend(test_score)
        test[3].extend(test_school)
        count += 1
    return train, test


def format_all_data(all_record):
    data = [[], [], [],[]]  # 学生,习题，得分，学校
    stu_list = set(all_record.index)

    for stu in stu_list:
        stu_item = all_record.loc[[stu], 'item_id'].values - 1
        stu_score = all_record.loc[[stu], 'score'].values
        stu_school = all_record.loc[[stu], 'school_id'].values -1

        data[0].append(stu - 1)
        data[1].append(stu_item)
        data[2].append(stu_score)
        data[3].append(stu_school)

    return data


def evaluate(pred, label):
    acc = metrics.accuracy_score(np.array(label).round(), np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(np.array(label).round(), np.array(pred))
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


class CICDM_Net(nn.Module):
    def __init__(self, concept_num: int, school_feature_dim: int,exercise_num: int, exer_conc_adj: Tensor,
                 conc_conc_adj: Tensor, potential_num: int = 32, conc_conc_ini_w: int = 5,
                 only_A: bool = False, device: str = 'cpu') -> None:
        super().__init__()

        #增加学校特征权重矩阵

        self.school_feature_dim = school_feature_dim
        self.school_feature_dim_w = nn.Parameter(torch.randn(school_feature_dim,potential_num))



        assert exer_conc_adj.size(0) == exercise_num and exer_conc_adj.size(1) == concept_num, 'exercise_concept adjacency matrix size wrong!'
        assert conc_conc_adj.size(0) == conc_conc_adj.size(1) == concept_num, 'concept_concept adjacency matrix size wrong!'
        self.device = device
        self.only_A = only_A

        self.concept_num = concept_num
        self.exercise_num = exercise_num
        self.potential_num = potential_num
        self.exer_conc_adj = exer_conc_adj
        self.exer_conc_w = nn.Parameter(torch.randn_like(exer_conc_adj))

        conc_conc_adj[torch.eye(concept_num, dtype=torch.bool)] = 1
        self.conc_conc_w = nn.Parameter(conc_conc_adj * conc_conc_ini_w)

        if not only_A:

            self.exer_pote_w = nn.Parameter(torch.randn((exercise_num, potential_num)))
            self.lambd = nn.Parameter(torch.ones((1, exercise_num)) * -2)

        self.guess = nn.Parameter(torch.ones((1, exercise_num)) * -2)
        self.slide = nn.Parameter(torch.ones((1, exercise_num)) * -2)

    def forward(self, exer_list, score_list,school_feature) -> Tuple[Tensor, Tensor]:
        A = torch.empty(len(score_list), self.concept_num).to(self.device)
        W = torch.sigmoid(self.exer_conc_w) * self.exer_conc_adj
        W2 = W / W.sum(dim=1).reshape(-1, 1)

        # print(school_feature.ndim,len(school_feature),school_feature.shape,school_feature)
        # print("a' dimension：",A.shape,A)     #A:192*110
        # print("school_feature:",school_feature.ndim,",",len(school_feature),",",school_feature.shape)  #192*1
        # print("school_feature_w' dimension：",school_feature_w.shape,school_feature_w)

        #加入学校特征
        if isinstance(school_feature, list):
            # Convert the list to a PyTorch tensor
            school_feature = torch.tensor(school_feature, dtype=torch.float).to(self.device)
        if school_feature.ndim == 1:
            # 如果 school_feature 是一维的，假设它表示单个学校的特征
            school_feature = school_feature.unsqueeze(0)  # 转换为 (1, 48)
        elif school_feature.size(1) != 48:
            # 如果 school_feature 不是 (batch_size, 48)，需要调整它
            # 这里的逻辑取决于 school_feature 的实际数据结构和意图
            # 例如，如果每个学校有一个特征值，可以尝试以下转换：
            school_feature = school_feature.expand(-1, 48)  # 扩展为 (batch_size, 48)
        # print("school_feature:",school_feature)

        school_feature_transformed = torch.matmul(school_feature,self.school_feature_dim_w)
        school_feature_transformed = school_feature_transformed.view(1, -1)  # Reshape to [1, 13]
        # Ensure it's the correct size
        school_feature_transformed = school_feature_transformed[:, :] 

        slide = torch.sigmoid(self.slide)
        guess = torch.sigmoid(self.guess)

        if not self.only_A:
            B = torch.empty(len(score_list), self.potential_num).to(self.device)
            D2 = torch.softmax(self.exer_pote_w, dim=1)
            lambd = torch.sigmoid(self.lambd)

        for i, X_i in enumerate(score_list):

            if i < school_feature_transformed.size(0):
        # Proceed with the operation if the index is within bounds
                transformed_feature = school_feature_transformed[i]
                # ... rest of the code ...
            else:
                # Handle the case where i exceeds the size of school_feature_transformed
                # This could be breaking the loop, continuing with a default value, etc.
                break
            X_i = torch.tensor(X_i).float().to(self.device).reshape(1, -1)

            # print("X_i",X_i.shape,"   school_feature_transformed",school_feature_transformed[i].shape)
            transformed_feature = school_feature_transformed[i]
            if transformed_feature.size(0) != X_i.size(1):
                if transformed_feature.size(0) < X_i.size(1):
                    # Pad if transformed_feature is smaller
                    padding_size = X_i.size(1) - transformed_feature.size(0)
                    transformed_feature = nn.pad(transformed_feature, (0, padding_size), "constant", 0)
                else:
                    # Truncate if transformed_feature is larger
                    transformed_feature = transformed_feature[:X_i.size(1)]
            
            # print("X_i.shape:",X_i.shape)
            # print("school_feature_transformed[i]",i,":",school_feature_transformed[i])

            # X_i = X_i + school_feature_transformed[i]

            # --------Knowledge concept start---------------
            W1_i_ = W[exer_list[i]]
            W1_i_sum = W1_i_.sum(dim=0)  # The cumulative sum of concepts not involved is 0
            W1_i = W1_i_[:, W1_i_sum != 0] / W1_i_sum[W1_i_sum != 0].reshape(1, -1)
            A1_i = X_i @ W1_i
            H1_i = torch.softmax(self.conc_conc_w[W1_i_sum != 0], dim=0)
            A[i] = A1_i @ H1_i
            # --------Knowledge concept end---------------

            if not self.only_A:
                # --------Skill start---------------
                D1_i_ = self.exer_pote_w[exer_list[i]]
                D1_i = torch.softmax(D1_i_, dim=0)
                B[i] = X_i @ D1_i
                # --------Skill end-----------------

        Y_A = A @ W2.T
        if not self.only_A:
            Y_B = B @ D2.T
            Y_ = (1 - lambd) * Y_A + lambd * Y_B
        else:
            Y_ = Y_A
        Y_ = Y_.clamp(1e-8, 1 - 1e-8)
        Y = (1 - slide) * Y_ + guess * (1 - Y_)

        return A, Y


class CICDM():
    def __init__(self, student_num: int, school_num: int,concept_num: int, exercise_num: int, exer_conc_adj: Tensor,
                 conc_conc_adj: Tensor, potential_num: int = 32, lr: float = 0.001,
                 only_A: bool = False, device: str = 'cpu') -> None:
        self.cd_net = CICDM_Net(concept_num,school_num, exercise_num, exer_conc_adj.to(device),
                                conc_conc_adj.to(device), potential_num, only_A=only_A, device=device).to(device)
        self.device = device
        self.student_num = student_num
        self.concept_num = concept_num
        self.school_num = school_num
        self.exercise_num = exercise_num
        self.optimizer = torch.optim.Adam(self.cd_net.parameters(), lr=lr)
        self.loss = torch.nn.BCELoss(reduction='mean')

    def fit(self, index_loader: DataLoader, train_df: pd.DataFrame, epochs: int = 5,
            n_splits: int = 5, test_df: pd.DataFrame = None) -> None:
        for epoch in range(epochs):
            epoch_loss = []
            for betch_data in tqdm(index_loader, "[Epoch:%s]" % (epoch + 1)):
                stu_list = np.array([x.numpy() for x in betch_data], dtype='int').reshape(-1)
                train_data, label_data = format_data(train_df.loc[stu_list, :], n_splits=n_splits)

                # -----start training-------------------

                ##修改
                max_length = max(len(f) for f in train_data[3])
                padded_school_features = [f + [0]*(max_length - len(f)) for f in train_data[3]]
                train_data[3] = torch.tensor(padded_school_features, dtype=torch.float).to(self.device)

                # 确定最大长度
                max_length = max(len(f) for f in train_data[3])

                # # 填充每个学校特征列表以匹配最大长度
                padded_school_features = []
                for f in train_data[3]:
                    # 确保 f 是列表
                    if not isinstance(f, list):
                        f = [f]
                    # 填充操作
                    padded_feature = f + [0] * (max_length - len(f))
                    padded_school_features.append(padded_feature)
                train_data[3] = torch.tensor(padded_school_features, dtype=torch.float).to(self.device)

                max_length = max(len(f) if isinstance(f, list) else 1 for f in train_data[3])

                # 填充每个特征列表以匹配最大长度
                padded_school_features = []
                for f in train_data[3]:
                    # 确保 f 是一个列表
                    if not isinstance(f, list):
                        f = [f]
                    # 确保列表中的所有元素都是浮点数
                    new_f = []
                    for element in f:
                        # 如果元素是单一数值，则直接转换
                        if isinstance(element, (int, float)):
                            new_f.append(float(element))
                        # 如果元素是数组或列表，处理数组或列表（例如，取第一个元素或计算平均值）
                        elif isinstance(element, (list, np.ndarray)) and len(element) > 0:
                            new_f.append(float(element[0]))  # 这里只是一个示例，具体取决于您的需求
                        else:
                            new_f.append(0.0)  # 如果无法处理，使用默认值
                    # 进行填充
                    padded_feature = new_f + [0.0] * (max_length - len(new_f))
                    padded_school_features.append(padded_feature)

                # 使用 NumPy 创建一个规整的二维数组
                padded_school_features_np = np.array(padded_school_features, dtype=float)

                # 转换为张量
                train_data[3] = torch.tensor(padded_school_features_np, dtype=torch.float).to(self.device)


                # predictions = self.model(input_data)
                # # Ensure predictions are in the range [0, 1]
                # predictions = torch.clamp(predictions, min=0, max=1)

                # _, all_pred = self.cd_net(train_data[1], train_data[2])
                _, all_pred = self.cd_net(train_data[1], train_data[2], train_data[3])

                all_pred = torch.clamp(all_pred, min=0, max=1)
                pred = all_pred[label_data[0], label_data[1]]
                label = torch.FloatTensor(label_data[2]).to(self.device)
                loss: Tensor = self.loss(pred, label)
                # ------end training--------------------
                epoch_loss.append(loss.item())

                # ------start update parameters----------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # ------ end update parameters-----------
            print('\t{}th epoch loss = {}'.format(epoch + 1, np.mean(epoch_loss)))
            if test_df is not None:
                self.test(index_loader, train_df, test_df)

    def test(self, index_loader: DataLoader, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[float, float, float, float]:
        test_pred_list, test_label_list = [], []
        for betch_data in tqdm(index_loader, "[Testing:]"):
            stu_list = np.array([x.numpy() for x in betch_data], dtype='int').reshape(-1)
            train, test = format_test_data(train_df.loc[stu_list, :],
                                           test_df.loc[stu_list, :])
            with torch.no_grad():
                # _, all_pred = self.cd_net(train[1], train[2])
                _, all_pred = self.cd_net(train[1], train[2],train[3])
                test_pred = all_pred[test[0], test[1]].clone().to('cpu').detach()
                test_pred_list.extend(test_pred.tolist())
                test_label_list.extend(test[2])
        acc, auc, rmse, mae = evaluate(test_pred_list, test_label_list)
        print("\ttest_result: \tacc:%.6f, auc:%.6f, rmse:%.6f, mae:%.6f" % (acc, auc, rmse, mae))
        return acc, auc, rmse, mae

    def get_A_and_Y(self, index_loader: DataLoader, all_record: pd.DataFrame):
        A = torch.empty((self.student_num, self.concept_num))
        Y = torch.empty((self.student_num, self.exercise_num))
        for betch_data in tqdm(index_loader, "[get_A_and_Y:]"):
            stu_list = np.array([x.numpy() for x in betch_data], dtype='int').reshape(-1)
            data = format_all_data(all_record.loc[stu_list, :])
            with torch.no_grad():
                cogn_state, all_pred = self.cd_net(data[1], data[2])
                A[data[0], :] = cogn_state.cpu().detach()
                Y[data[0], :] = all_pred.cpu().detach()
        return A, Y
