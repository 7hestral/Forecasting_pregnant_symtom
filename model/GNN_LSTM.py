from layer import *

class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=10, node_dim=16, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=1, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, per_batch_adj=True, is_classifier=False, cell_name="LSTM"):
        super(gtnet, self).__init__()
        self.is_classifier = is_classifier
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.per_batch_adj = per_batch_adj
     
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        # self.start_conv = nn.Conv2d(in_channels=in_dim,
        #                             out_channels=residual_channels,
        #                             kernel_size=(1, 1))
        self.gconv_channel_size_lst = [1] + [conv_channels] * layers
        
        self.gc = graph_constructor_lstm(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat, cell_name=cell_name)
        self.cls_emb = None

        

        self.seq_length = seq_length
        kernel_size = 5
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1
        # self.avg_pool = nn.AvgPool2d(kernel_size=(1,self.receptive_field))
        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)
                if self.gcn_true:
                    self.gconv1.append(mixprop_batch(self.gconv_channel_size_lst[j-1], self.gconv_channel_size_lst[j], gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop_batch(self.gconv_channel_size_lst[j-1], self.gconv_channel_size_lst[j], gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=conv_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,self.receptive_field),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.is_classifier:
            self.linear = nn.Linear(in_features=num_nodes, out_features=2)


        self.idx = torch.arange(self.num_nodes).to(device)

    def _get_cell_emb(self, layer_output):
            """
            Args:
                layer_output(:obj:`Tensor`): shape (batch, end_channels, feature_size, 1)

            Returns:
                :obj:`Tensor`: shape (batch, end_channels*feature_size)
            """

            cell_emb = layer_output.squeeze(-1).reshape(layer_output.shape[0], -1)
            return cell_emb

    def forward(self, input, idx=None, CLS=False):
        # print(input.shape)
        
        # input = input.permute(0,2,1).unsqueeze(1)
        
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        # print("this is gnn only model")
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(input.squeeze().permute(0,2,1), self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A
        x = input
        # x = self.start_conv(input)
        # skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            # residual = x
            # filter = self.filter_convs[i](x)
            # filter = torch.tanh(filter)
            # gate = self.gate_convs[i](x)
            # gate = torch.sigmoid(gate)
            # x = filter * gate
            # x = F.dropout(x, self.dropout, training=self.training)
            # s = x
            # s = self.skip_convs[i](s)
            # skip = s + skip
            if self.per_batch_adj:
                adp_transpose = adp.permute(0, 2, 1)
            else:
                adp_transpose = adp.transpose(1,0)
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp_transpose)
            else:
                x = self.residual_convs[i](x)

            # x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        # skip = self.skipE(x) + skip
        x = F.relu(x)
        temp = self.end_conv_1(x)
        self.cls_emb = self._get_cell_emb(temp)
        temp = F.relu(temp)
        x = self.end_conv_2(temp)
        # x = self.avg_pool(x)

        output_dict = {}
        output_dict['output'] = x
        if CLS:
            output_dict['cls_emb'] = self.cls_emb
        if self.is_classifier:
            output_dict['classification_output'] = self.linear(x.squeeze())

        return output_dict
