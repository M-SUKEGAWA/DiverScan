import numpy as np
import torch
import torch.nn as nn


class Graph():
    """ 
    The graph to model the skeletons. 
    
    Originally from 
    https://github.com/yysijie/st-gcn/blob/master/net/utils/graph.py
    https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/03_action_recognition_ST_GCN.ipynb
    
    """

    def __init__(
        self,
        edge_index,  # (num_edges, 2), zero-indexed node ids
        num_node,
        max_hop=1,  # hop0: self-loop. hop1: next 1 node.
        dilation=1,
    ):

        self.get_edge(edge_index, num_node)
        self.hop_dis = self.get_hop_distance(num_node, self.edge, max_hop)
        self.get_adjacency(num_node, max_hop, dilation)

    def __str__(self):
        return self.A

    def get_edge(self, edge_index, num_node):
        """Collect the self-loops and the given edges.

        edge_index holds zero-indexed node ids, which is how the pipeline
        builds a complete graph and how userset_edges are converted. The
        reference implementation this class came from subtracted one here,
        because its skeleton definitions were one-indexed.
        """
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(int(i), int(j)) for (i, j) in edge_index]
        self.edge = self_link + neighbor_link

    def get_adjacency(self, num_node, max_hop, dilation):
        valid_hop = range(0, max_hop + 1, dilation)
        adjacency = np.zeros((num_node, num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        # elif strategy == 'distance':
        A = np.zeros((len(valid_hop), num_node, num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A

    def get_hop_distance(self, num_node, edge, max_hop):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
    

class ConvTemporalGraphical(nn.Module):

    """
    The basic module for applying a graph convolution.
    
    Originally from
    https://github.com/yysijie/st-gcn/blob/master/net/utils/tgcn.py
    https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/03_action_recognition_ST_GCN.ipynb
    
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        s_kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()

        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * s_kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):

        # A: (B, max_hop+1, N, N, T)
        # x: (B, C, T, N) 
        #   -conv> (B, s_kernel_size*C, T, N) 
        #   -view> (B, s_kernel_size, C, T, N) 
        #   -einsum> (B, C, T, N)

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        # x = torch.einsum('nkctv,kvw->nctw', (x, A))  # nct, w without k and v (will be contracted)
        x = torch.einsum('nkctv,nkvwt->nctw', (x, A))  # nct, nwt without k and v (will be contracted)

        return x.contiguous()
    

class MaskedBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d whose batch statistics ignore padded timestamps.

    Without the mask the padded zeros pull the mean towards zero and shrink the
    variance, by an amount that depends on how much padding the other sequences
    in the batch happen to carry. In eval mode the running statistics are used
    and the mask makes no difference, so the parent implementation is called.
    """

    def forward(self, x, mask=None):
        # x: (B, C, T, N), mask: (B, 1, T, 1)
        # A mask with nothing masked is the same as no mask; defer to the
        # parent so that unpadded batches take exactly the standard path.
        if mask is None or not self.training or bool(mask.all()):
            return super().forward(x)

        valid = mask.to(x.dtype).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        count = valid.sum()
        if count < 2:
            return super().forward(x)

        mean = (x * valid).sum(dim=(0, 2, 3)) / count
        centered = x - mean.view(1, -1, 1, 1)
        var = ((centered ** 2) * valid).sum(dim=(0, 2, 3)) / count

        if self.track_running_stats:
            with torch.no_grad():
                unbiased_var = var * count / (count - 1)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased_var)
                self.num_batches_tracked += 1

        normalized = centered / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        if self.affine:
            normalized = normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return normalized


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d whose batch statistics ignore padded timestamps.

    See MaskedBatchNorm2d; this one takes (B, F, T) with a (B, 1, T) mask.
    """

    def forward(self, x, mask=None):
        if mask is None or not self.training or bool(mask.all()):
            return super().forward(x)

        valid = mask.to(x.dtype).expand(x.shape[0], 1, x.shape[2])
        count = valid.sum()
        if count < 2:
            return super().forward(x)

        mean = (x * valid).sum(dim=(0, 2)) / count
        centered = x - mean.view(1, -1, 1)
        var = ((centered ** 2) * valid).sum(dim=(0, 2)) / count

        if self.track_running_stats:
            with torch.no_grad():
                unbiased_var = var * count / (count - 1)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased_var)
                self.num_batches_tracked += 1

        normalized = centered / torch.sqrt(var.view(1, -1, 1) + self.eps)
        if self.affine:
            normalized = normalized * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

        return normalized


class STGCNBlock(nn.Module):
    """
    Applies a spatial temporal graph convolution over an input graph sequence.

    Originally from 
    https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/03_action_recognition_ST_GCN.ipynb
    
    """

    def __init__(
        self,
        A_size,
        in_channels_edge_weight,
        in_channels,
        out_channels,
        t_kernel_size,
        s_kernel_size,
        dropout_probability,
        residual=True,
        stride=1,
    ):
        super().__init__()

        assert t_kernel_size % 2 == 1
        padding = ((t_kernel_size - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(
            in_channels, 
            out_channels,
            s_kernel_size,
        )

        # Edge importance weighting (learnable weight)
        if in_channels_edge_weight > 0:
            self.linear_edge_weight = nn.Linear(
                in_features=in_channels_edge_weight,
                out_features=s_kernel_size,  # s_kernel_size: max_hop+1
                bias=False,
            )
        # else:
        #     self.M = nn.Parameter(torch.ones(A_size))

        # A ModuleList rather than a Sequential, so that the mask can be handed
        # to the normalisation layers. The children keep their index names, so
        # the state_dict keys are unchanged.
        self.tcn = nn.ModuleList([
            MaskedBatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (t_kernel_size, 1),
                (stride, 1),
                padding,
            ),
            MaskedBatchNorm2d(out_channels),
            nn.Dropout(p=dropout_probability),
        ])

        if not residual:
            self.residual_mode = "zero"
            self.residual = None

        elif (in_channels == out_channels) and (stride == 1):
            self.residual_mode = "identity"
            self.residual = None

        else:
            self.residual_mode = "project"
            self.residual = nn.ModuleList([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                MaskedBatchNorm2d(out_channels),
            ])

        self.relu = nn.ReLU()
        
    def forward(self, x, A, edge_weight, mask=None):
        # x: (B, C, T, N)
        # mask: (B, 1, T, 1), False on padded timestamps. None means no padding.
        # A: (max_hop+1, N, N), consistent across all data.
        #   -> (1, max_hop+1, N, N, 1)
        # edge weight: (B, N, N, T, in_channels_edge_weight)
        #   -linear> (B, N, N, T, max_hop+1) 
        #   -transpose> (B, max_hop+1, N, N, T)
        #   -*(1, max_hop+1, N, N, 1)> (B, max_hop+1, N, N, T)

        B, *_ = x.shape

        A = A.unsqueeze(0).unsqueeze(-1)

        edge_weight = self.linear_edge_weight(edge_weight).permute(0, 4, 1, 2, 3)
        edge_weight = self.relu(edge_weight)

        edge_weight = A * edge_weight

        batch_has_padding = mask is not None and torch.logical_not(mask).any().item()
        block_mask = mask if batch_has_padding else None

        def apply_layers(layers, value):
            for layer in layers:
                if isinstance(layer, MaskedBatchNorm2d):
                    value = layer(value, block_mask)
                else:
                    value = layer(value)
                if block_mask is not None:
                    value = value * block_mask
            return value

        if batch_has_padding:
            # Keep the padded timestamps at zero so that they cannot leak into
            # the temporal convolution of their neighbours.
            x = x * mask

        if self.residual_mode == "zero":
            res = 0
        elif self.residual_mode == "identity":
            res = x * mask if batch_has_padding else x
        else:
            res = apply_layers(self.residual, x)

        # x: (B, C, T, N)
        # edge_weight: (B, max_hop+1, N, N, T)
        x = self.gcn(x, edge_weight)
        if batch_has_padding:
            x = x * mask

        x = apply_layers(self.tcn, x) + res
        if batch_has_padding:
            x = x * mask
        x = self.relu(x)
        if batch_has_padding:
            x = x * mask

        return x
        
    
class AttentionBlock(nn.Module):
    """
    Attention mask.
    kernel_size=(1, 1).
    """
    
    def __init__(
        self, 
        in_channels,
    ):
        
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

    def forward(self, x, mask=None):
        attention = self.conv2d(x)  # (B, C, T, N) -> (B, 1, T, N)
        batch_size, _, num_timestamp, num_node = attention.shape

        batch_has_padding = mask is not None and torch.logical_not(mask).any().item()

        if batch_has_padding:
            # Drive the padded timestamps to zero probability by giving them the
            # lowest representable score before the softmax.
            padded = torch.logical_not(mask.expand(-1, 1, -1, num_node))
            attention = attention.masked_fill(padded, torch.finfo(attention.dtype).min)

        attention = attention.reshape(batch_size, -1)  # (B, T*N)
        attention = torch.softmax(attention, dim=1)  # softmax over (T*N). (B, T*N)
        attention = attention.reshape(batch_size, 1, num_timestamp, num_node)  # (B, 1, T, N)

        if batch_has_padding:
            attention = attention * mask

        return attention  # (B, 1, T, N)


class NoCalculationBlock(nn.Module):
    def __init__(
        self, 
    ):

        super().__init__()    

    def forward(self, x, *args):

        return x
    

class OutputBlock(nn.Module):

    def __init__(
        self,
        in_features,
        output_size,
        node_shuffle,
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=output_size,
        )

        self.node_shuffle = node_shuffle

    def forward(self, x, mask=None):

        batch_size, _, _, num_node = x.shape

        if mask is not None:
            # The flatten below is over a fixed T, so the padded timestamps must
            # contribute exactly zero.
            x = x * mask

        if self.training == True and self.node_shuffle == True:
            # In training mode, to prevent learning that depends on the order of N, elements are randomly shuffled along N
            device = x.device  # Retrieve the device information of the x tensor
            shuffled_index = torch.randperm(num_node).to(device)  # Generate a random permutation of integers from 0 to N - 1 -> Place on the same device as the x tensor
            x = torch.index_select(x, 3, shuffled_index)  # Rearrange the second dimension according to the index. (B, C, T, N_shuffled)
        else:
            x = x

        x = x.reshape(batch_size, -1)
        output = self.linear(x)

        # Do not use softmax function here.
        # nn.CrossEntropyLoss() will be used later.
        # https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch

        return output  # (B, output_size)
    

class FlexibleBranch(nn.Module):
    def __init__(
        self, 
        layers, 
        output_size,
        A_size,
        in_channels_edge_weight,
        s_kernel_size,
        node_shuffle,
    ):
        """
        Layers after branching.
        """

        super().__init__()

        self.attention = None  # keep the attention of the branch
        self.layers = nn.ModuleList()
        self.layertypes = []
        for layer_info in layers:
            layer_type = layer_info['layertype']
            if layer_type == 'STGCNBlock':
                layer = STGCNBlock(   
                    A_size=A_size,
                    in_channels_edge_weight=in_channels_edge_weight,  
                    in_channels=layer_info['in_channels'],
                    out_channels=layer_info['out_channels'],
                    t_kernel_size=layer_info['temporal_kernel_size'],
                    s_kernel_size=s_kernel_size,
                    dropout_probability=layer_info['dropout_probability'],
                )
            elif layer_type == 'AttentionBlock':
                layer = AttentionBlock(
                    in_channels=layer_info['in_channels'],
                )
            elif layer_type == 'OutputBlock':
                layer = OutputBlock(
                    in_features=layer_info['in_features'],  # in_features: (out_channels*num_node*num_timestamp)
                    output_size=output_size,
                    node_shuffle=node_shuffle,
                )
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            self.layers.append(layer)
            self.layertypes.append(layer_type)

    def forward(self, x, A, edge_weight, mask=None):
        self.attention = None  # initialize attention
        for layer_type, layer in zip(self.layertypes, self.layers):
            if layer_type == 'STGCNBlock':
                # x: (B, C, T, N)
                x = layer(x, A, edge_weight, mask)
            elif layer_type == 'AttentionBlock':
                # x: (B, C, T, N) * (B, 1, T, N) -> (B, C, T, N)
                attention = layer(x, mask)  # get the attention mask (B, 1, T, N)
                x = x * attention
                self.attention = attention  # keep the attention mask
            elif layer_type == 'OutputBlock':
                # x: (B, C, T, N) -> (B, output_size)
                output = layer(x, mask)
            else:
                raise ValueError("Unsupported layer type for branch")
        return output, self.attention


class Model(nn.Module):
    """
    Spatial temporal graph convolutional multi models.

    Originally from 
    https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/03_action_recognition_ST_GCN.ipynb
    
    """

    def __init__(
        self, 
        before_branch_config, 
        branch_config,
        output_size,
        in_channels,   # eg., Set in_channels = 3 when your data have speed, distance_from_initial_position, and pose data (head, back, tail). Set in_channels = 2 when your data have speed and distance_from_initial_position. 
        in_channels_pose,  # 0 or a natural number. If you do not have pose data, you can set 0.
        in_channels_edge_weight,
        edge_index,
        num_node,
        node_shuffle,
        max_hop=1,
    ):
        super().__init__()

        # load graph
        self.graph = Graph(
            edge_index=edge_index, 
            num_node=num_node, 
            max_hop=max_hop,
        )
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)  # A: (max_hop+1, N, N)
        self.register_buffer('A', A)

        A_size = A.size()
        spatial_kernel_size = A.size(0)

        # Linear for pose
        self.linear_for_pose = nn.Linear(
            in_features=in_channels_pose,
            out_features=1,
            bias=False,
        )
        self.relu = nn.ReLU()

        # Batch normalization
        self.batchnorn = MaskedBatchNorm1d(in_channels * num_node)

        # Before branching
        self.before_branch = nn.ModuleList()
        if before_branch_config:
            for layer_info in before_branch_config:
                layer_type = layer_info['layertype']
                if layer_type == 'STGCNBlock':
                    self.before_branch.append(
                        STGCNBlock(
                            A_size=A_size,
                            in_channels_edge_weight=in_channels_edge_weight,
                            in_channels=layer_info['in_channels'],
                            out_channels=layer_info['out_channels'],
                            t_kernel_size=layer_info['temporal_kernel_size'],
                            s_kernel_size=spatial_kernel_size,
                            dropout_probability=layer_info['dropout_probability'],
                        )
                    )
                else:
                    raise ValueError("Unsupported layer type for before_branch")
        else:
            self.before_branch.append(NoCalculationBlock())

        # After branching
        self.branches = nn.ModuleList()
        for config in branch_config:
            branch = FlexibleBranch(
                layers=config, 
                output_size=output_size,
                A_size=A_size,
                in_channels_edge_weight = in_channels_edge_weight,
                s_kernel_size=spatial_kernel_size,
                node_shuffle=node_shuffle,
            )
            self.branches.append(branch)

    def forward(self, x, edge_weight, x_pose, has_pose):

        # x: (B, C, T, N)
        # A: (max_hop+1, N, N), consistent across all data.
        # edge weight: (B, N, N, T, in_channels_edge_weight)
        # x_pose: 
            # if has_pose == True: (B, N, T, in_channels_pose).
            # if has_pose == False: (B)
        # has_pose: bool

        # Make a mask for the input x. False marks a padded timestamp.
        # Padding is written as zeros across every channel and every node by
        # standardize_and_replace_nan_to_zero(), so an all-zero slice is padding.
        all_zeros = torch.all(torch.all(x == 0, dim=1), dim=-1)  # (B, T)
        mask = torch.logical_not(all_zeros)[:, None, :, None]    # (B, 1, T, 1)

        # Make pose to one channel and concat it to x
        if has_pose:
            x_pose = self.linear_for_pose(x_pose)  # (B, N, T, 1)
            x_pose = x_pose.permute(0, 3, 2, 1)  # (B, 1, T, N)
            x_pose = self.relu(x_pose)
            x = torch.cat((x, x_pose), dim=1)  # (B, in_channels-1, T, N) -cat- (B, 1, T, N) -> (B, in_channels, N, T)
        else:
            x = x  # (B, in_channels, N, T)
        
        # Batch normalization
        B, C, T, N = x.size() # batch, channel, timestamp, node
        x = x.permute(0, 3, 1, 2).contiguous().view(B, N * C, T)
        # mask is (B, 1, T, 1) here but x has been folded to (B, N*C, T).
        x = self.batchnorn(x, mask[:, :, :, 0])
        x = x.view(B, N, C, T).permute(0, 2, 3, 1).contiguous()
        x = x * mask

        # Before branching
        for layer in self.before_branch:
            x = layer(x, self.A, edge_weight, mask)
        
        # After branching
        outputs = []
        attentions = []
        for branch in self.branches:
            x_branch = x.clone()
            # output: (B, output_size)
            # attention: (B, T, N)
            output, attention = branch(x_branch, self.A, edge_weight, mask)
            outputs.append(output)  # outputs: length number_of_attention_branches
            attentions.append(attention)

        # attentions: 
        #   -cat> (B, number_of_attention_branches, T, N) 
        #   -permute> (B, number_of_attention_branches, N, T)
        attentions = torch.cat(attentions, dim=1).permute(0, 1, 3, 2)
        
        return outputs, attentions