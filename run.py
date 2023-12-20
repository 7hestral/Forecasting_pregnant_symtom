
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import DataLoader

from dataset.sequence_dataset import MultiStepDataset
from model.Autoformer import Model as Autoformer
from model.DLinear import Model as DLinear
from model.GNN_Conv import gtnet as GNN_conv
from model.GNN_EMB import gtnet as GNN_emb
from model.GNN_LSTM import gtnet as GNN_lstm
from model.LSTM import LSTMClassifier, AdversarialDiscriminator
from model.Linear import Model as Linear
from model.MTGNN import gtnet as MTGNN
from model.MTGNN_Conv import gtnet as MTGNN_conv
from model.MTGNN_LSTM import gtnet as MTGNN_lstm
from model.MTGNN_two_graph import gtnet as MTGNN_two_graph
from model.NLinear import Model as NLinear
from model.Simple_GNN import gtnet as GNN_simple
from train_eval import *
from training_utils import *

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

parser.add_argument('--model_type', type=str, default="GNN")
parser.add_argument('--task_name', type=str, default="edema_analysis_3rd_missingness")
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')

parser.add_argument('--optim', type=str, default='adam')
# parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--loss_type', type=str, default='L1', help='L1, L2, or CE')
parser.add_argument('--normalize', type=int, default=0) # already normalized
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=14,help='number of nodes/variables') 
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=5,help='k') # used to be 20 by default
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=8,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=5,help='input sequence length') 
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
# parser.add_argument('--horizon', type=int, default=2) 
parser.add_argument('--layers',type=int,default=2,help='number of layers')

parser.add_argument('--batch_size',type=int,default=200,help='batch size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=5,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=50,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=30,help='step_size')
parser.add_argument('--adv', type=bool, default=True, help='whether to add adverserial loss')
parser.add_argument('--schedule_interval',type=int,default=1,help='scheduler interval')
parser.add_argument('--schedule_ratio',type=float,default=0.001,help='multiplicative factor of learning rate decay')
parser.add_argument('--adv_E_delay_epochs',type=int,default=3,help='number of epochs start training encoder for adversarial task')
parser.add_argument('--adv_D_delay_epochs',type=int,default=0,help='mnumber of epochs start training decoder for adversarial task')
parser.add_argument('--num_epoch_discriminator',type=int,default=10,help='number of epochs for discriminator')
parser.add_argument('--adv_weight',type=float,default=0,help='adversarial weight')
parser.add_argument('--rse_weight',type=float,default=1,help='mse loss weight')
parser.add_argument('--symptom_weight',type=float,default=1,help='symptom loss weight')
parser.add_argument('--with_demographic',type=bool,default=True,help='whether or not adding demographic data')


args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)
adv_E_delay_epochs = args.adv_E_delay_epochs
adv_D_delay_epochs = args.adv_D_delay_epochs
num_epoch_discriminator = args.num_epoch_discriminator
# adv_weight = args.adv_weight
rse_weight = args.rse_weight
# adv_weight_str = str(adv_weight).replace('.', 'dot')
rse_weight_str = str(rse_weight).replace('.', 'dot')
use_fake_label_adv = False
root_path = os.path.join('/storage/home/ruizhu')
norm_population = True
# symptom_weight = args.symptom_weight
# symptom_weight_str = str(symptom_weight).replace('.', 'dot')
symptom_weight_lst = [args.symptom_weight, args.symptom_weight, args.symptom_weight, args.symptom_weight]
symptom_weight_str_lst = [str(s).replace('.', 'dot') for s in symptom_weight_lst]
adv_weight_lst = [args.adv_weight, args.adv_weight, args.adv_weight, args.adv_weight]
adv_weight_str_lst = [str(adv_weight).replace('.', 'dot') for adv_weight in adv_weight_lst]

def weighted_multi_step_cross_entropy(outputs, targets, data_point_weights=None, class_weights=None):
    """
    Custom loss function for multi-step classification with different weights for each data point.
    'outputs' is expected to be a tensor of shape (batch_size, num_steps, num_classes),
    'targets' is expected to be a tensor of shape (batch_size, num_steps) with class indices.
    'data_point_weights' can be used to apply different weights to different data points.
    """
    batch_size, num_steps, _ = outputs.shape
    loss = 0
    for step in range(num_steps):
        step_loss = F.cross_entropy(outputs[:, step, :], targets[:, step], reduction='none', weight=class_weights)
        if data_point_weights is not None:
            # print(step_loss.shape)
            # print(data_point_weights.shape)
            step_loss *= data_point_weights[:, step]
        loss += step_loss.sum()
    return loss / (batch_size * num_steps)

if args.with_demographic:
    args.num_nodes += 3
def main(user_lst, curr_slice, name, feature_lst, model_type="GNN", task_name='edema_pred', print_feature_corr=True, model_path=None, demo_df=None):

    # # remove temperature delta
    # idx_to_remove = feature_lst.index('temperature_delta')
    # feature_lst = feature_lst.copy()
    # feature_lst.pop(idx_to_remove)
    
    train_df_lst = []
    val_df_lst = []
    test_df_lst = []

    train_dataset_lst = []
    val_dataset_lst = []
    test_dataset_lst = []

    list_users = []
    for u in user_lst:
        # if curr_slice >= user_dict[u]: # not available for this user
        #     continue
        
        file_name = os.path.join(root_path,
                                 f'dataset/{task_name}/user_{u}_{task_name}_hyperimpute_slice{curr_slice}.csv')
        # print(file_name)
       

        try:
            curr_all_data = np.loadtxt(file_name, delimiter=',')# [:, :-1] # exclude edema coarse label for now
            list_users.append(u)
        except OSError:
            continue
        
        # print(u)
        num_all_data, _ = curr_all_data.shape

        # # remove temperature delta
        # curr_all_data = np.delete(curr_all_data, idx_to_remove, axis=1)
        
        # val_split_idx = int(num_all_data * 0.6)
        # test_split_idx = int(num_all_data * 0.8)
        # TODO: remove this, it is useless, the part which split is when creating sequence dataset
        val_split_idx = 17
        test_split_idx = 19

        # val_split_idx -= args.seq_in_len
        # test_split_idx -= args.seq_in_len

        if args.with_demographic:
            demo_data = demo_df[demo_df['user_id'] == u].drop('user_id', axis=1).to_numpy()
            demo_data = np.repeat(demo_data, num_all_data, axis=0)

            # demographic data column must come before the time series data, 
            # since symptom data should be at -1
            curr_all_data = np.concatenate((demo_data, curr_all_data), axis=1)
            
        
        # test is the same as val for now
        curr_train_data = curr_all_data[:val_split_idx, :].copy()
        curr_val_data = curr_all_data[val_split_idx:test_split_idx, :].copy()
        # curr_val_data = curr_all_data[val_split_idx:test_split_idx, :]
        curr_test_data = curr_all_data[test_split_idx:, :].copy()
        data_is_padded = False
        # print(curr_all_data.shape)
        previous_data_file_name = os.path.join(root_path,
                                               f'dataset/{task_name}/user_{u}_{task_name}_hyperimpute_slice{curr_slice - 1}.csv')
        if os.path.exists(previous_data_file_name):
            data_is_padded = True

            
            # print(curr_train_data.shape)
            # prev_data = np.loadtxt(previous_data_file_name, delimiter=',')[-args.seq_in_len:, :-1]
            prev_data = np.loadtxt(previous_data_file_name, delimiter=',')
            if args.with_demographic:
                prev_data = np.concatenate((demo_data, prev_data), axis=1)
            curr_train_data = np.concatenate((prev_data, curr_train_data), axis=0)
            print("padded", curr_train_data.shape)
        


        curr_train_data[:, -1] = curr_train_data[:, -1].astype(int)
        curr_val_data[:, -1] = curr_val_data[:, -1].astype(int)
        curr_test_data[:, -1] = curr_test_data[:, -1].astype(int)
        
        # print(curr_train_data.shape)
        # print(curr_val_data.shape)
        # print(curr_test_data.shape)

        train_df_lst.append(curr_train_data)
        val_df_lst.append(curr_val_data)
        test_df_lst.append(curr_test_data)

    # normalization

    # over all population

    if norm_population:
        print('len(train_df_lst)', len(train_df_lst))
        
        # normalized_train_df_lst, min_value_lst, max_value_lst = min_max_normalization(train_df_lst)
        # normalized_val_df_lst, _, _ = min_max_normalization(val_df_lst, min_value_lst=min_value_lst, max_value_lst=max_value_lst)
        # normalized_test_df_lst, _, _ = min_max_normalization(test_df_lst, min_value_lst=min_value_lst, max_value_lst=max_value_lst)
        # feature_size = (train_df_lst[0].shape[1])
        my_scaler = MinMaxScaler(args.num_nodes)
        # print('test_df_lst[0]', test_df_lst[0][-1])

        normalized_train_df_lst = my_scaler.fit_transform(train_df_lst)
        normalized_val_df_lst = my_scaler.transform(val_df_lst)
        normalized_test_df_lst = my_scaler.transform(test_df_lst)
        # print('normalized_test_df_lst[0]', normalized_test_df_lst[0][-1])
        # original_test_df_lst = my_scaler.inverse_transform(normalized_test_df_lst)
        
        normalized_all_df_lst = []
    
        for i in range(len(list_users)):
            curr_train_data = normalized_train_df_lst[i]
            curr_val_data = normalized_val_df_lst[i]
            curr_test_data = normalized_test_df_lst[i]
            normalized_all_df = np.concatenate((curr_train_data, curr_val_data, curr_test_data), axis=0)
            
            normalized_all_df_lst.append(normalized_all_df)
    
    else:
        # over each individual, use only first 1 week data
        normalized_train_df_lst = []
        normalized_val_df_lst = []
        normalized_test_df_lst = []
        normalized_all_df_lst = []
        for i in range(len(list_users)):
            curr_train_data = train_df_lst[i]
            curr_val_data = val_df_lst[i]
            curr_test_data = test_df_lst[i]
    
            # print(curr_train_data.shape)
    
            first_two_week = curr_train_data[:8, :]
            rest = curr_train_data[8:, :]
            curr_train_data[:, -1] = curr_train_data[:, -1].astype(int)
            curr_val_data[:, -1] = curr_val_data[:, -1].astype(int)
            curr_test_data[:, -1] = curr_test_data[:, -1].astype(int)
            
    
            normalized_first_two_week, min_value_lst, max_value_lst = min_max_normalization([first_two_week])
            normalized_rest, _, _ = min_max_normalization([rest], min_value_lst=min_value_lst, max_value_lst=max_value_lst)
    
            normalized_val, _, _ = min_max_normalization([curr_val_data], min_value_lst=min_value_lst, max_value_lst=max_value_lst)
            normalized_test, _, _ = min_max_normalization([curr_test_data], min_value_lst=min_value_lst, max_value_lst=max_value_lst)
    
            normalized_train = np.concatenate((normalized_first_two_week, normalized_rest), axis=0)
            
            normalized_train_df_lst.append(normalized_train)
            # print(normalized_train[:, -1])
            
            if np.sum(np.isnan(normalized_train)) > 0:
                print(list_users[i])
    
            normalized_val_df_lst.append(normalized_val)
            
            normalized_test_df_lst.append(normalized_test)
            normalized_all_df = np.concatenate((normalized_train, normalized_val, normalized_test), axis=0)
            print(normalized_all_df.shape)
            normalized_all_df_lst.append(normalized_all_df)
            
    
    


    # create sequential datasets
    # 22 in total, 13/2/7
    num_test_data = 7
    num_val_data = 2
    for count, curr_data in enumerate(normalized_all_df_lst):
        all_df_dataset = MultiStepDataset(curr_data, args.seq_out_len, args.seq_in_len, device, user_id=count)

        all_idx = list(range(len(all_df_dataset)))
        test_idx = all_idx[-num_test_data:]
        val_idx = all_idx[-(num_test_data + num_val_data):-num_test_data]
        train_idx = all_idx[:-(num_test_data + num_val_data)]
        

        train_subset = Subset(all_df_dataset, train_idx)
        val_subset = Subset(all_df_dataset, val_idx)
        test_subset = Subset(all_df_dataset, test_idx)

        
        test_dataset_lst.append(test_subset)
        val_dataset_lst.append(val_subset)
        train_dataset_lst.append(train_subset)
    # for count, curr_train_data in enumerate(normalized_train_df_lst):
    #     curr_train_dataset = SequenceDataset(curr_train_data, args.horizon, args.seq_in_len, device, user_id=count)
    #     train_dataset_lst.append(curr_train_dataset)
    # for count, curr_val_data in enumerate(normalized_val_df_lst):
    #     curr_val_dataset = SequenceDataset(curr_val_data, args.horizon, args.seq_in_len, device, user_id=count)
    #     val_dataset_lst.append(curr_val_dataset)
    # for count, curr_test_data in enumerate(normalized_test_df_lst):
    #     curr_test_dataset = SequenceDataset(curr_test_data, args.horizon, args.seq_in_len, device, user_id=count)
    #     test_dataset_lst.append(curr_test_dataset)
    
    # aggregate them
    aggregated_train_dataset = ConcatDataset(train_dataset_lst)
    aggregated_val_dataset = ConcatDataset(val_dataset_lst)
    aggregated_test_dataset = ConcatDataset(test_dataset_lst)

    train_dataloader = DataLoader(aggregated_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(aggregated_val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(aggregated_test_dataset, batch_size=args.batch_size, shuffle=False)
    # args.data = f'/mnt/results/user_{selected_user}_activity_bodyport_hyperimpute.csv'
    args.save = os.path.join(root_path, f'results/model/model_{name}_{model_type}_adv{args.adv}_slice{curr_slice}_{task_name}.pt')
    print(vars(args))

    
    # all_y = None
    # for batch_data in train_dataloader:
    #     Y = batch_data['Y'][:, :, -1]
    #     Y.reshape(-1)
    #     if all_y is None:
    #         all_y = Y
    #     else:
    #         all_y = torch.concat([all_y, Y])
    # all_y[all_y>0] = 1
    # print(task_name)
    # print(curr_slice)
    # print('with symptom', torch.sum(all_y))
    # print('all', all_y.shape)
    # exit(0)

    
    if args.loss_type == 'CE':
        is_classification = True
    if model_type == "MTGNN":
        model = MTGNN(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification)
        if model_path is None:
            model.apply(weight_reset)
        else:
            stored_model = torch.load(model_path)
            model.load_state_dict(stored_model.state_dict())
    elif model_type == "LSTM" or model_type == 'GRU':
        if is_classification:
            final_states = 2
        else:
            final_states = args.num_nodes
        model = LSTMClassifier(feature_size=args.num_nodes, n_state=final_states, hidden_size=30, rnn=model_type, is_classifier=is_classification)
    elif model_type == "NLinear":
        model_config = Config(seq_len=args.seq_in_len, pred_len=args.seq_out_len, feature_size=args.num_nodes, is_classifier=is_classification)
        model = NLinear(model_config)
    elif model_type == "Linear":
        model_config = Config(seq_len=args.seq_in_len, pred_len=args.seq_out_len)
        model = Linear(model_config)
    # model = model.to(device)
    elif model_type == "DLinear":
        model_config = Config(seq_len=args.seq_in_len, pred_len=args.seq_out_len, feature_size=args.num_nodes, is_classifier=is_classification)
        model = DLinear(model_config)
    elif model_type == "Autoformer":
        model_config = Config(seq_len=args.seq_in_len, pred_len=args.seq_out_len)
        model = Autoformer(model_config)
    elif model_type == "GNN_lstm":
        model = GNN_lstm(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification)
    elif model_type == 'GNN_emb':
        model = GNN_emb(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification)
    elif model_type == "MTGNN_lstm":
        model = MTGNN_lstm(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification)
    elif model_type == "MTGNN_gru":
        model = MTGNN_lstm(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification, cell_name="GRU")
    elif model_type == "GNN_gru":
        model = GNN_lstm(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification, cell_name="GRU")
    elif model_type == 'GNN_conv':
        model = GNN_conv(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification, cell_name="GRU")
    elif model_type == 'MTGNN_conv':
        model = MTGNN_conv(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification, cell_name="GRU")
    elif model_type == 'MTGNN_two_graph_lstm':
        model = MTGNN_two_graph(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification, cell_name="LSTM")
    elif model_type == 'MTGNN_two_graph_gru':
        model = MTGNN_two_graph(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification, cell_name="GRU")
    elif model_type == 'GNN_simple':
        model = GNN_simple(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False, is_classifier=is_classification)
    model = model.to(device)

    print(args)
    if model_type == 'GNN' or model_type == "GNN_only":
        print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)
    optimizer_D = None
    optimizer_E = None
    discriminator = None
    criterion_adv = None
    decoder = nn.Sequential(
        nn.Linear(in_features=args.end_channels * args.num_nodes, out_features=128, bias=True),
        nn.ReLU(),
        nn.LayerNorm(128),
        nn.Linear(in_features=128, out_features=6, bias=True)
    )
    if args.adv:
        lr_ADV = 0.001  # learning rate for discriminator, used when ADV is True
        discriminator = AdversarialDiscriminator(
            d_model=args.end_channels * args.num_nodes,
            n_cls=len(list_users)).to(device)

        criterion_adv = nn.CrossEntropyLoss().to(device)  # consider using label smoothing
        optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
        scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, args.schedule_interval, gamma=args.schedule_ratio)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D, args.schedule_interval, gamma=args.schedule_ratio)

    if args.loss_type == 'L1':
        criterion = nn.L1Loss().to(device)
    elif args.loss_type == 'L2':
        criterion = nn.MSELoss().to(device)
    elif args.loss_type == "CE":
        # if 'edema' in args.task_name:
        #     weight = torch.Tensor([0.15, 0.85])
        # else:
        #     weight = torch.Tensor([0.5, 0.5])
        # criterion = nn.CrossEntropyLoss(weight=weight).to(device)
        criterion = weighted_multi_step_cross_entropy
    evaluateL2 = nn.MSELoss().to(device)
    evaluateL1 = nn.L1Loss().to(device)


    best_val = 0
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            loss_dict = train(args, train_dataloader, model, criterion, optim, device, model_type, epoch=epoch,
                              rse_weight=rse_weight, adv_weight=adv_weight_lst[curr_slice], adv_D_delay_epochs=adv_D_delay_epochs,
                              num_epoch_discriminator=num_epoch_discriminator, adv_E_delay_epochs=adv_E_delay_epochs,
                              optim_D=optimizer_D, optim_E=optimizer_E, 
                discriminator=discriminator, criterion_adv=criterion_adv, symptom_weight=symptom_weight_lst[curr_slice], use_fake_label=use_fake_label_adv)
            train_loss = loss_dict['training_loss']
            adv_loss = loss_dict['adv_loss_D']
            train_acc_D = loss_dict['train_acc_D']
            # train_loss = train(train_dataloader, model, criterion, optim, device, model_type, epoch=epoch)
            val_loss, val_rae, val_corr, feat_corr_lst, val_roc_score, val_accuracy, val_specificity, val_sensitivity, val_mae_lst, val_norm_mae_lst = evaluate(args, val_dataloader, model, evaluateL2, evaluateL1, device, model_type, root_path, adv_weight_str_lst[curr_slice], my_scaler)
            # print('edema_corr', edema_corr)
            edema_corr = feat_corr_lst[-1]
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | adv loss {:5.4f} | adv acc {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, adv_loss, train_acc_D, val_loss, val_rae, val_corr, edema_corr), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            feature_corr_output_str = ''
            if print_feature_corr:
                for idx, feat in enumerate(feature_lst):
                    feature_corr_output_str += f'{feat}_corr {feat_corr_lst[idx]} | '
                print(feature_corr_output_str)

            if best_val < val_roc_score:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_roc_score
            if epoch % 5 == 0:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
            # if epoch % 20 == 0 and model_type != 'NLinear' and model_type != 'DLinear':
            #     plot_tsne(train_dataloader, val_dataloader, test_dataloader, model, 
            #               epoch, name, model_type, task_name, device, root_path, adv_weight_str_lst[curr_slice], rse_weight_str)

        evaluate(args, val_dataloader, model, evaluateL2, evaluateL1, device, model_type, root_path, adv_weight_str_lst[curr_slice], my_scaler, plot=False, plot_name=f'{model_type}_slice{curr_slice}.png')
            # if epoch % 5 == 0:
            #     test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
            #                                          args.batch_size)
            #     print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr), flush=True)
            # if args.adv:
            #     scheduler_D.step()
            #     scheduler_E.step()
        # with open(args.save, 'wb') as f:
        #     torch.save(model, f)



    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    vtest_acc, vtest_rae, vtest_corr, vfeat_corr_lst, vroc_score, vaccuracy, vspecificity, vsensitivity, vmae_lst, _ = evaluate(args, val_dataloader, model, evaluateL2, evaluateL1, device, model_type, root_path, adv_weight_str_lst[curr_slice], my_scaler)
    # test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
    #                                      args.batch_size)
    # print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
    test_acc, test_rae, test_corr, tfeat_corr_lst, troc_score, taccuracy, tspecificity, tsensitivity, tmae_lst, tnorm_mae_lst = evaluate(args, test_dataloader, model, evaluateL2, evaluateL1, device, model_type, root_path, adv_weight_str_lst[curr_slice], my_scaler)
    # first row is val then test
    symptom_metric_dict = {
        "validation_roc_score": vroc_score,
        "validation_accuracy": vaccuracy,
        "validation_specificity": vspecificity,
        "validation_sensitivity": vsensitivity,
        "test_roc_score": troc_score,
        "test_accuracy": taccuracy,
        "test_specificity": tspecificity,
        "test_sensitivity": tsensitivity,

    }
    print(symptom_metric_dict)
    return vtest_acc, vtest_rae, vtest_corr, vfeat_corr_lst, test_acc, test_rae, test_corr, tfeat_corr_lst, symptom_metric_dict, tmae_lst, tnorm_mae_lst

if __name__ == "__main__":

    
    vacc = []
    vrae = []
    vcorr = []
    acc = []
    rae = []
    corr = []
    feat_corr_lst = []
    test_feat_corr_lst = []
    num_runs = 5
    
    task_name = args.task_name
    if 'fatigue' in task_name:
        feature_name_lst = ['high','inactive','low','medium','non_wear','rest','steps',
    'breath_average','hr_average','rmssd','score','restless', 'hr_lowest', 'fatigue']
        
        user_lst = [39, 84, 122, 137, 142, 174, 214, 225, 431, 581, 962, 966, 1002, 1032, 1363, 1373, 1387, 1437, 1440, 1697, 1715, 1719, 1730, 1744, 1753, 2015, 2018, 2058, 2062, 2068, 2109, 2150, 2153, 2159, 2167, 2214, 2265, 2312, 2318, 2339, 2340, 2347, 2370, 2386, 2482, 2484, 2488, 2500, 2514, 2516, 2572, 2578, 2583, 2598, 2603, 2607, 2610, 2612, 2664]
    elif 'edema' in task_name:
        feature_name_lst = ['high','inactive','low','medium','non_wear','rest','steps',
    'breath_average','hr_average','rmssd','score','restless', 'hr_lowest', 'edema']
        user_lst = [28, 29, 30, 35, 37, 38, 39, 42, 44, 45, 47, 53, 54, 64, 65, 66, 67, 68, 74, 75, 79, 80, 94, 95, 1431, 1393, 99, 114, 118, 119, 2093, 122, 124, 127, 135, 137, 155, 1014, 156, 158, 1373, 168, 1000, 173, 174, 185, 186, 1001, 190, 192, 193, 199, 200, 212, 1021, 972, 1724, 1004, 1429, 234, 975, 280, 289, 290, 291, 292, 404, 405, 407, 408, 409, 410, 977, 1047, 428, 581, 594, 595, 600, 603, 615, 1658, 1045, 983, 966, 969, 985, 987, 989, 991, 992, 1005, 1455, 1038, 1044, 2117, 2225, 2226, 2120, 1363, 1366, 1367, 1378, 1719, 2151, 1383, 1386, 1387, 1721, 1425, 1426, 1427, 1432, 1436, 1439, 1440, 1443, 1697, 1452, 1700, 1703, 1706, 1707, 1708, 1709, 1710, 1712, 1715, 1716, 1728, 1731, 2299, 1743, 1745, 1749, 1750, 2187, 2091, 2197, 1976, 1988, 1989, 1994, 1995, 1996, 1997, 1999, 2000, 2001, 2004, 2014, 2016, 2018, 2019, 2038, 2032, 2056, 2058, 2060, 2061, 2062, 2064, 2068, 2076, 2202, 2645, 2654, 2084, 2223, 2311, 2100, 2102, 2109, 2158, 2159, 2160, 2166, 2169, 2174, 2176, 2178, 2212, 2235, 2664, 2650, 2265, 2339, 2340, 2341, 2342, 2260, 2510, 2615, 2147, 2145, 2139, 2133, 2164, 2182, 2183, 2167, 2201, 2204, 2203, 2261, 2259, 2257, 2313, 2312, 2351, 2350, 2379, 2386, 2381, 2516, 2541, 2518, 2530, 2487, 2490, 2496, 2489, 2485, 2500, 2483, 2502, 2503, 2547, 2592, 2593, 2549, 2580, 2581, 2536, 2599, 2602, 2572, 2603, 2571, 2583, 2574, 2584, 2575, 2628, 2576, 2611, 2562, 2631, 2613, 2635, 2636, 2564, 2565, 2551, 2637, 2656]

    # for curr_slice in range(4):
    
    # feat_name_lst_copy.remove('temperature_delta')
    demo_df = None
    if args.with_demographic:
        demo_df = pd.read_csv(os.path.join(root_path, 'dataset', 'demographic.csv'))
        feature_name_lst = ['age', 'prior_pregnant', 'prior_birth'] + feature_name_lst
    feat_name_lst_copy = feature_name_lst.copy()
    for curr_slice in [0,1,2]:
    # for curr_slice in range(4):
        # curr_slice = 2
        vacc = []
        vrae = []
        vcorr = []
        acc = []
        rae = []
        corr = []
        feat_corr_lst = []
        test_feat_corr_lst = []

        test_feat_mae_lst = []
        test_feat_norm_mae_lst = []
        
        model_type = args.model_type
        
        symptom_metric_lst = []
        # if curr_slice != 0 and model_type == 'GNN':
        #     previous_model = os.path.join('/', 'mnt', 'results', 'model', f'model_all_feat_0_advweight_{adv_weight_str}_indinorm_slices{curr_slice-1}_{model_type}_advTrue_slice{curr_slice-1}_{task_name}.pt')
        # else:
        #     previous_model = None
        
        previous_model = None
        # 42
        # torch.manual_seed(2139)
        # seed_everything(2139)
        seed_everything(2140)
        
        for i in range(num_runs):
            model_name = f'single_{i}_advweight_{adv_weight_str_lst[curr_slice]}_slices{curr_slice}_symptom_weight{symptom_weight_str_lst[curr_slice]}_demo{args.with_demographic}_k{args.subgraph_size}_dim{args.node_dim}_xydiffw'
            val_acc, val_rae, val_corr, vfeat_corr, test_acc, test_rae, test_corr, test_feat_corr, symptom_metric_dict, tmae_lst, tnorm_mae_lst = main(user_lst, curr_slice, model_name, 
            feature_lst=feature_name_lst, model_type=model_type, task_name=task_name, model_path=previous_model, demo_df=demo_df)
            symptom_metric_lst.append(symptom_metric_dict)
            vacc.append(val_acc)
            vrae.append(val_rae)
            vcorr.append(val_corr)
            feat_corr_lst.append(vfeat_corr)
            acc.append(test_acc)
            rae.append(test_rae)
            corr.append(test_corr)
            test_feat_corr_lst.append(test_feat_corr)
            test_feat_mae_lst.append(tmae_lst)
            test_feat_norm_mae_lst.append(tnorm_mae_lst)
        test_feat_mae_lst = np.vstack(test_feat_mae_lst)
        mean_feat_mae_lst = np.mean(test_feat_mae_lst, axis=0)
        std_feat_mae_lst = np.std(test_feat_mae_lst, axis=0)

        test_feat_norm_mae_lst = np.vstack(test_feat_norm_mae_lst)
        mean_norm_feat_mae_lst = np.mean(test_feat_norm_mae_lst, axis=0)
        std_norm_feat_mae_lst = np.std(test_feat_norm_mae_lst, axis=0)

        
        print('\n\n')
        print(f'{num_runs} runs average')
        print('\n\n')
        print("valid\trse\trae\tcorr")
        print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
        print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))


        # feat_str = 'valid\t'
        # mean_str = 'mean\t'
        # std_str = 'std\t'
        # for feat in feature_name_lst:
        #     feat_str += feat + '\t'
        # for i in range(mean_feat_corr_lst.shape[0]):
        #     mean_str += ("{:5.4f}\t".format(mean_feat_corr_lst[i]))
        #     std_str += ("{:5.4f}\t".format(std_feat_corr_lst[i]))

        feat_corr_lst = np.vstack(feat_corr_lst)
        mean_feat_corr_lst = np.mean(feat_corr_lst, axis=0)
        std_feat_corr_lst = np.std(feat_corr_lst, axis=0)
        feat_str = 'valid&'
        mean_str = 'mean&'
        std_str = 'std&'
        for feat in feat_name_lst_copy:
            feat_str += feat + '&'
        for i in range(mean_feat_corr_lst.shape[0]):
            mean_str += ("${:5.4f}\pm{:5.4f}$&".format(mean_feat_corr_lst[i], std_feat_corr_lst[i]))
            # std_str += ("{:5.4f}&".format(std_feat_corr_lst[i]))
        print('\n\n')
        print(feat_str)
        print(mean_str)
        # print(std_str)
        print('-'*10, 'test', '-' * 10)
        feat_corr_lst = np.vstack(test_feat_corr_lst)
        mean_feat_corr_lst = np.mean(test_feat_corr_lst, axis=0)
        std_feat_corr_lst = np.std(test_feat_corr_lst, axis=0)
        feat_str = 'test&'
        mean_str = 'mean&'
        std_str = 'std&'
        for feat in feat_name_lst_copy:
            feat_str += feat + '&'
        for i in range(mean_feat_corr_lst.shape[0]):
            mean_str += ("${:5.4f}\pm{:5.4f}$&".format(mean_feat_corr_lst[i], std_feat_corr_lst[i]))
            # std_str += ("{:5.4f}&".format(std_feat_corr_lst[i]))
        print('\n\n')
        print(feat_str)
        print(mean_str)
        with open(os.path.join(root_path, 'results', 'result_txts', f'{model_name}_{model_type}_{task_name}.txt'), 'w', newline='') as f:
            f.write(feat_str)
            f.write('\n\n')
            f.write(mean_str)

            f.write('\n\n')
            f.write(f'{num_runs} runs average')
            f.write('\n\n')
            f.write("valid\trse\trae\tcorr")
            f.write("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
            f.write("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))

            f.write('\n\n')
            f.write(f'{num_runs} runs average')
            f.write('\n\n')
            f.write("test\trse\trae\tcorr")
            f.write("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
            f.write("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))


        result_data = [feat_name_lst_copy, mean_feat_corr_lst, std_feat_corr_lst, mean_feat_mae_lst, std_feat_mae_lst, mean_norm_feat_mae_lst, std_norm_feat_mae_lst]
        data_t = list(map(list, zip(*result_data)))
        headers = ['feature name', 'corr_mean', 'corr_std', 'mae_mean', 'mae_std', 'norm_mae_mean', 'norm_mae_std']
        with open(os.path.join(root_path, 'results', 'result_csvs', f'{model_name}_{model_type}_{task_name}.csv'), 'w', newline='') as csvfile:
            # create a CSV writer object
            writer = csv.writer(csvfile)
            
            # write the header row to the CSV file
            writer.writerow(headers)
            
            # write the data to the CSV file
            writer.writerows(data_t)
    
        symptom_metric_df = pd.DataFrame(symptom_metric_lst)
        
        
        print(symptom_metric_df)
        symptom_metric_df.to_csv(os.path.join(root_path, 'results', 'result_csvs', f'{model_name}_{model_type}_{task_name}_slice{curr_slice}_symptom_metric.csv'))
        
        # Compute mean and standard deviation for each column
        column_means = symptom_metric_df.mean()
        column_std_devs = symptom_metric_df.std()
        combined_df = pd.concat([column_means, column_std_devs], axis=1)
        combined_df.columns = ['Mean', 'StandardDeviation']  # Rename columns
        combined_df.to_csv(os.path.join(root_path, 'results', 'result_csvs', f'{model_name}_{model_type}_{task_name}_slice{curr_slice}_symptom_metric_summary.csv'))
