import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from util import *


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = False


def plot_tsne(train_dataloader, val_dataloader, test_dataloader, model, epoch, name,
              model_type, task_name, device, root_path, adv_weight_str, rse_weight_str):
    # gtnet_features = FeatureExtractor(model, layers=["end_conv_1"])
    output_lst = []
    edema_lst = []
    input_lst = []
    user_label_lst = []
    num_data_lst = [0, 0, 0]
    for c, dataloader in enumerate([train_dataloader, val_dataloader, test_dataloader]):
        for batch_data in dataloader:
            X = batch_data['X'].to(device)
            Y = batch_data['Y'].to(device)
            user_label = batch_data['user_label']
            user_label_lst.append(user_label)
            feature_size = X.shape[-1]

            if model_type == 'GNN':
                X = torch.unsqueeze(X, dim=1)
                X = X.transpose(2, 3)
            edema_label = Y[:, -1].unsqueeze(-1).cpu().detach().numpy()

            edema_lst.append(edema_label)

            num_data_lst[c] += X.shape[0]

            with torch.no_grad():

                output = model(X, CLS=True)['cls_emb']
                # print(output.shape)
                output_lst.append(output.squeeze(-1).view(output.shape[0], -1).cpu().detach().numpy())
                # input_lst.append(X.squeeze(-1).reshape(X.shape[0], -1).cpu().detach().numpy())
    edema_lst = np.concatenate(edema_lst)
    all_embeddings = np.concatenate(output_lst, axis=0)
    all_user_label = np.concatenate(user_label_lst)
    for perplexity in [40, 60]:
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300, random_state=42)
        # tsne_results = tsne.fit_transform(pca_result)
        tsne_results = tsne.fit_transform(all_embeddings)
        df_tsne = pd.DataFrame(tsne_results, columns=["X", "Y"])

        df_tsne["User_labels"] = all_user_label
        df_tsne["User_labels"] = df_tsne["User_labels"].apply(lambda i: str(i))

        df_tsne["Training_set"] = [True] * num_data_lst[0] + [False] * (num_data_lst[1] + num_data_lst[2])
        markers_dict = {
            True: 'o',
            False: 'X',
        }
        df_tsne["Symptom label"] = edema_lst
        fig, axs = plt.subplots(ncols=2, figsize=(12, 12))
        # plt.figure(figsize=(16,16))
        axs[0].set(ylim=(-12, 12))
        axs[0].set(xlim=(-25, 25))
        axs[1].set(ylim=(-12, 12))
        axs[1].set(xlim=(-25, 25))
        sns.scatterplot(
            x="X", y="Y",
            hue="User_labels",
            style="Training_set",
            data=df_tsne,
            legend="full", s=70,
            alpha=0.9,
            markers=markers_dict,
            ax=axs[0]
        )
        sns.scatterplot(
            x="X", y="Y",
            hue="Symptom label",
            style="Training_set",
            data=df_tsne,
            legend="full", s=70,
            alpha=0.9,
            markers=markers_dict,
            ax=axs[1]
        )
        print("save as ", os.path.join(root_path, 'results', 'plots',
                                       f'tsne_epoch{epoch}_adv{adv_weight_str}_l1{rse_weight_str}_{name}_{model_type}_{task_name}_perplex{perplexity}'))
        plt.savefig(os.path.join(root_path, 'results', 'plots',
                                 f'tsne_epoch{epoch}_adv{adv_weight_str}_l1{rse_weight_str}_{name}_{model_type}_{task_name}_perplex{perplexity}'))
        plt.clf()


def extract_data_for_users(user_ids, predict_lst, test_lst, graph_lst, target_user_id):
    indices = [i for i, user_id in enumerate(user_ids) if user_id == target_user_id]
    # print(indices)
    extracted_predict_lst = predict_lst[indices]
    extracted_test_lst = test_lst[indices]

    extracted_graph_lst = graph_lst[indices]

    return extracted_predict_lst, extracted_test_lst, extracted_graph_lst


# generate error per participants, and graph per participants
def error_analysis(loss_type, dataloader, model, device, model_type, list_users, time_step=0):
    model.eval()
    predict = None
    test = None
    user_label_aggregate = None
    graphs = None
    for batch_data in dataloader:
        X = batch_data['X'].to(device)
        Y = batch_data['Y'].to(device)
        user_label = batch_data['user_label'].to(device)
        feature_size = X.shape[-1]
        if 'GNN' in model_type:
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            with torch.no_grad():
                output_dict = model(X)
                if loss_type == "CE":

                    output = model(X)['classification_output']

                    curr_graph = model.gc(X.squeeze().permute(0, 2, 1), model.idx)
                    if len(curr_graph.shape) == 2:
                        curr_graph = curr_graph.repeat(X.shape[0], 1, 1)
                    # output = torch.argmax(output, dim=1, keepdim=True)
                    # Y = Y[:, -1].unsqueeze(1)
                    output = torch.argmax(output, dim=2, keepdim=True)[:, time_step, :].squeeze()
                    output = output.reshape(-1)
                    # print('output.shape', output.shape)
                    Y = Y[:, time_step, -1]
                    Y = Y.reshape(-1)
                else:
                    # raise NotImplementedError
                    # 

                    output = model(X)['output']
                    curr_graph = model.gc(X.squeeze().permute(0, 2, 1), model.idx)
                    if len(curr_graph.shape) == 2:
                        curr_graph = curr_graph.repeat(X.shape[0], 1, 1)
                    output = output[:, -1].reshape(-1)
                    Y = Y[:, time_step, -1]
                    Y = Y.reshape(-1)

        else:
            if loss_type == "CE":
                output = model(X)['classification_output']
                curr_graph = torch.zeros((X.shape[0], feature_size, feature_size))
                output = torch.argmax(output, dim=1, keepdim=True)
                Y = Y[:, -1].unsqueeze(1)
            else:
                raise NotImplementedError

        if predict is None:
            predict = output
            test = Y
            user_label_aggregate = user_label
            graphs = curr_graph
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            user_label_aggregate = torch.cat((user_label_aggregate, user_label))
            graphs = torch.cat((graphs, curr_graph))

    predict = predict.data.cpu().numpy()
    test = test.data.cpu().numpy()

    graphs = graphs.data.cpu().numpy()
    user_label_aggregate = user_label_aggregate.data.cpu().numpy()
    # print(predict.shape)
    # print(test.shape)
    # print(graphs.shape)
    # print(user_label_aggregate.shape)

    unique_users = list(set(user_label_aggregate))
    result_dict = {}

    for u in unique_users:
        curr_pred, curr_test, curr_graph = extract_data_for_users(user_label_aggregate, predict, test, graphs, u)

        result_dict[list_users[u]] = (curr_pred, curr_test, curr_graph)
    return result_dict


def evaluate(args, dataloader, model, evaluateL2, evaluateL1, device, model_type, root_path, adv_weight_str, scaler,
             plot=False, plot_name='', time_step=0):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    user_label_aggregate = None
    for batch_data in dataloader:
        X = batch_data['X'].to(device)
        Y = batch_data['Y'].to(device)
        user_label = batch_data['user_label'].to(device)
        feature_size = X.shape[-1]
        # if model_type == 'GNN' or model_type == "GNN_only":
        if 'GNN' in model_type:
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            with torch.no_grad():
                output_dict = model(X)
                if args.loss_type == "CE":
                    output = model(X)['classification_output']

                    output = torch.argmax(output, dim=2, keepdim=True)[:, time_step, :].squeeze()
                    output = output.reshape(-1)
                    # print('output.shape', output.shape)
                    Y = Y[:, time_step, -1]
                    Y = Y.reshape(-1)
                    # print('Y.shape', Y.shape)
                else:
                    output = model(X)['output']
        elif model_type == 'LSTM' or model_type == 'GRU' or model_type == 'NLinear' or model_type == 'DLinear':
            with torch.no_grad():
                if args.loss_type == 'CE':
                    output = model(X)['classification_output']

                    output = torch.argmax(output, dim=2, keepdim=True)[:, time_step, :].squeeze()
                    output = output.reshape(-1)
                    # print('output.shape', output.shape)
                    Y = Y[:, time_step, -1]
                    Y = Y.reshape(-1)
                    # print('Y.shape', Y.shape)
                else:
                    output = model(X)
        elif model_type == 'Autoformer':
            output = model(X)
            print('Autoformer', output.shape)
            exit(0)
        elif model_type == "Linear":
            with torch.no_grad():
                output = model(X)
        # elif model_type == "NLinear":
        #     output = model(X)
        #     print('NLinear', output.shape)
        #     exit(0)
        if args.loss_type != 'CE':
            output = torch.squeeze(output)
            if len(output.shape) == 1:
                output = output.unsqueeze(dim=0)

        if predict is None:
            predict = output
            test = Y
            user_label_aggregate = user_label
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            user_label_aggregate = torch.cat((user_label_aggregate, user_label))

        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * feature_size)

    rse = math.sqrt(total_loss)  # / data.rse
    rae = (total_loss_l1)  # / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()

    user_label_aggregate = user_label_aggregate.data.cpu().numpy()
    if plot:
        # print(predict[:, -1])

        target_values = [0, 0.2, 0.4, 0.8, 1, 1.2]
        rounded_predict = np.array([min(target_values, key=lambda x: abs(x - num)) for num in predict[:, -1]])
        # print(rounded_predict)
        # print(Ytest[:, -1])
        # print(user_label_aggregate)
        unique_users = np.unique(user_label_aggregate)
        indices = [np.where(user_label_aggregate == value)[0] for value in unique_users]

        user_specific_pred = [predict[:, -1][idx] for idx in indices]
        user_specific_pred_rounded = [rounded_predict[idx] for idx in indices]
        user_specific_Ytest = [Ytest[:, -1][idx] for idx in indices]
        for i in range(len(unique_users)):
            curr_user = unique_users[i]
            x = list(range(len(user_specific_pred[i])))
            plt.plot(x, user_specific_Ytest[i] * 5, label='True label')
            plt.plot(x, user_specific_pred[i] * 5, label='Predicted label')
            plt.plot(x, user_specific_pred_rounded[i] * 5, label='Rounded predicted label')
            plt.legend()
            plt.ylabel('Survey answer')
            plt.ylim(ymax=10, ymin=0)
            plt.savefig(
                os.path.join(root_path, 'results', 'result_plot', f'user{curr_user}adv_{adv_weight_str}_{plot_name}'))
            plt.clf()

    def get_correlation(predict, Ytest):
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        # correlation = ((predict) * (Ytest)).mean(axis=0)

        correlation = (correlation[index]).mean()
        return correlation

    if args.loss_type == 'CE':
        corr_lst = [0] * args.num_nodes
        mae_lst = [0] * args.num_nodes
        normalized_mae_lst = [0] * args.num_nodes
        roc_score, accuracy, specificity, sensitivity = get_symptom_metric(Ytest, predict)
        correlation = 0
    else:
        # edema_correlation = get_correlation(predict[:, -1], Ytest[:, -1])
        correlation = get_correlation(predict, Ytest)
        corr_lst = []
        roc_score, accuracy, specificity, sensitivity = get_symptom_metric(Ytest[:, -1], predict[:, -1])

        mae_lst = []
        normalized_mae_lst = []

        original_pred = scaler.inverse_transform([predict])
        original_test = scaler.inverse_transform([Ytest])

        for feat_idx in range(predict.shape[1]):
            corr_lst.append(get_correlation(predict[:, feat_idx], Ytest[:, feat_idx]))
            mae_lst.append(np.mean(np.absolute(original_pred[:, feat_idx] - original_test[:, feat_idx])))
            normalized_mae_lst.append(np.mean(np.absolute(predict[:, feat_idx] - Ytest[:, feat_idx])))
    return rse, rae, correlation, corr_lst, roc_score, accuracy, specificity, sensitivity, mae_lst, normalized_mae_lst


def get_symptom_metric(symptom_true, symptom_pred):
    target_values = [0, 0.2, 0.4, 0.8, 1, 1.2]
    print('symptom_pred.shape', symptom_pred.shape)
    rounded_predict = np.array([min(target_values, key=lambda x: abs(x - num)) for num in symptom_pred])
    rounded_true = np.array([min(target_values, key=lambda x: abs(x - num)) for num in symptom_true])
    rounded_predict = rounded_predict > 0
    rounded_true = rounded_true > 0
    roc_score = roc_auc_score(rounded_true, rounded_predict)
    tn, fp, fn, tp = confusion_matrix(rounded_true, rounded_predict).ravel()
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(rounded_true, rounded_predict)
    sensitivity = tp / (tp + fn)
    return roc_score, accuracy, specificity, sensitivity


def train(args, dataloader, model, criterion, optim, device, model_type, epoch, rse_weight, adv_weight,
          adv_D_delay_epochs, num_epoch_discriminator, adv_E_delay_epochs,
          optim_D=None, optim_E=None, discriminator=None,
          criterion_adv=None, symptom_weight=1, use_fake_label=True):
    train_loss_E_class = 0
    adv_loss_E = 0
    train_adv_loss_D = 0
    n_samples = 0
    iter = 0
    train_adv_correct_num = 0
    for batch_data in dataloader:
        model.train()
        if args.adv:
            discriminator.train()
        X = batch_data['X'].to(device)
        Y = batch_data['Y'].to(device)
        user_label = batch_data['user_label'].to(device)
        change_mask = batch_data['change_mask'].to(device)
        feature_size = X.shape[-1]

        # if model_type == 'GNN' or model_type == "GNN_only":
        if 'GNN' in model_type:
            optim_E.zero_grad()
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            if iter % args.step_size == 0:
                perm = np.random.permutation(range(args.num_nodes))
            # num_sub = int(args.num_nodes / args.num_split)
            # for j in range(args.num_split):
            #     if j != args.num_split - 1:
            #         id = perm[j * num_sub:(j + 1) * num_sub]
            #     else:
            #         id = perm[j * num_sub:]
            id = perm
            id = torch.tensor(id).to(device)
            tx = X
            ty = Y
            output_dict = model(tx, CLS=args.adv)
            output = output_dict['output']
            if args.adv:
                cls_emb = output_dict['cls_emb']
            if args.loss_type != 'CE':
                output = torch.squeeze(output)
                weight_mask = torch.ones(output.shape)
                weight_mask[:, -1] = 1
                weight_mask = weight_mask.to(device)

                loss_CLS = criterion(output * weight_mask, ty * weight_mask)
            else:
                classification_output = output_dict['classification_output']
                classification_y = ty[:, :, -1]
                classification_y = (classification_y > 0).long()

                weight = torch.ones(classification_y.shape).to(device)

                weight[change_mask == 1] = symptom_weight
                weight[change_mask == 0] = 1
                # print(weight)

                if 'edema' in args.task_name:
                    class_weights = torch.Tensor([0.15, 0.85]).to(device)
                else:
                    class_weights = torch.Tensor([0.5, 0.5]).to(device)
                loss_CLS = criterion(classification_output, classification_y, data_point_weights=weight,
                                     class_weights=class_weights)
            train_loss_E_class += loss_CLS.item()
            if args.adv and epoch >= adv_D_delay_epochs:
                if epoch % 5 == 0:
                    discriminator.apply(weight_reset)
                for i in range(num_epoch_discriminator):
                    discriminator.train()
                    optim_D.zero_grad()

                    output_dict = model(tx, CLS=args.adv)
                    # print('output_dict["cls_emb"]', output_dict["cls_emb"].shape)
                    pred = discriminator(output_dict["cls_emb"].detach())
                    loss_discriminator = criterion_adv(pred, user_label)
                    loss_discriminator.backward()
                    optim_D.step()
                    if i == num_epoch_discriminator - 1:
                        train_adv_loss_D += loss_discriminator.item()
                        pred_class = torch.squeeze(pred.max(1)[1])
                        train_adv_correct_num += torch.sum(pred_class == user_label)
                        n_samples += pred_class.shape[0]

            if args.adv and epoch >= adv_E_delay_epochs:

                output_dict = model(tx, CLS=args.adv)
                fake_idx = torch.randperm(user_label.nelement())
                fake_user_label = user_label.view(-1)[fake_idx].view(user_label.size())
                if use_fake_label:
                    loss_adv_E = criterion_adv(discriminator(output_dict["cls_emb"]), fake_user_label)
                else:
                    loss_adv_E = -criterion_adv(discriminator(output_dict["cls_emb"]), user_label)
            else:
                loss_adv_E = 0

            loss_total = rse_weight * loss_CLS + adv_weight * loss_adv_E
            loss_total.backward()
            optim_E.step()

        elif model_type == "LSTM" or model_type == "GRU" or model_type == "NLinear" or model_type == 'DLinear':
            loss_adv_E = 0
            if args.loss_type == 'CE':
                output_dict = model(X)
                # output = output_dict['classification_output']
                # classification_y = Y[:, -1]
                # classification_y = (classification_y > 0).long()
                # loss = criterion(output, classification_y)

                classification_output = output_dict['classification_output']
                ty = Y
                classification_y = ty[:, :, -1]
                classification_y = (classification_y > 0).long()
                output = classification_y

                weight = torch.ones(classification_y.shape).to(device)

                weight[change_mask == 1] = symptom_weight
                weight[change_mask == 0] = 1
                # print(weight)

                if 'edema' in args.task_name:
                    class_weights = torch.Tensor([0.15, 0.85]).to(device)
                else:
                    class_weights = torch.Tensor([0.5, 0.5]).to(device)
                loss = criterion(classification_output, classification_y, data_point_weights=weight,
                                 class_weights=class_weights)
            else:
                output = model(X)
                output = torch.squeeze(output)
                weight_mask = torch.ones(output.shape)
                weight_mask[:, -1] = 1
                weight_mask = weight_mask.to(device)
                # loss = criterion(output[:, -1], Y[:, -1])
                loss = criterion(weight_mask * output, weight_mask * Y)  # loss = criterion(output, Y)
            loss.backward()
            train_loss_E_class += loss.item()
            n_samples += (output.size(0) * feature_size)
            grad_norm = optim.step()
        elif model_type == 'Autoformer':
            output = model(X)
            print('Autoformer', output.shape)
            exit(0)
        elif model_type == "Linear":
            loss_adv_E = 0
            output = model(X)
            output = torch.squeeze(output)
            weight_mask = torch.zeros(output.shape)
            weight_mask[:, -1] = 1
            weight_mask = weight_mask.to(device)
            # loss = criterion(output[:, -1], Y[:, -1])
            loss = criterion(weight_mask * output, weight_mask * Y)
            loss.backward()
            train_loss_E_class += loss.item()
            n_samples += (output.size(0) * feature_size)
            grad_norm = optim.step()
        # elif model_type == "NLinear":
        #     output = model(X)
        #     print("nlinear", output.shape)
        #     print(Y.shape)
        #     exit(0)

    loss_dict = {'training_loss': train_loss_E_class, 'adv_loss_D': adv_weight * loss_adv_E,
                 'train_acc_D': 0 if n_samples == 0 else train_adv_correct_num / n_samples}
    return loss_dict
