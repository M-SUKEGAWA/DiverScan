import torch
import torch.nn as nn


class MaskedConv1dBlock(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size,
        padding,
        dropout_probability,
        stride=1,
        residual=True,
    ):

        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if not residual:
            self.residual = lambda x: x.new_zeros(x.size(0), out_channels, x.size(2))

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout_probability)           

    def forward(
        self,
        x,
        mask,
    ):
        
        batch_has_padding = torch.logical_not(mask).any().item()  # bool

        if batch_has_padding:
            # Masking
            x = x * mask  # (B, in_channels, T) * (B, 1, T)
            # Residual
            res = self.residual(x)
            res = res * mask
            # Conv1D
            x = self.conv1d(x)  # (B, out_channels, T)
            # Masking
            x = x * mask # (B, out_channels, T) * (B, 1, T)

            x = (x + res) * mask
            x = self.relu(x)
            x = self.dropout(x)
            x = x * mask

        else:
            # Residual
            res = self.residual(x)
            # Conv1D
            x = self.conv1d(x)  # (B, out_channels, T)

            x = x + res
            x = self.relu(x)
            x = self.dropout(x)

        return x  # (B, out_channels, T)


class MaskedLSTMBlock(nn.Module):
    def __init__(
        self, 
        input_size,
        hidden_size,
        dropout_probability,
        bidirectional=True,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout_probability)           

    def forward(
        self,
        x,
        lengths,
    ):
        
        # x: (B, T, input_size)
        _, num_timestamp, _ = x.shape

        # Pack
        # The lengths argument contains the actual lengths of each sequence in the batch, allowing the model to ignore the padded parts.
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            x, 
            lengths.cpu().long(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=num_timestamp,
            padding_value=0.0,
        )

        # dropout
        output = self.dropout(output)

        return output  # (B, T, hidden_size*2)
    
    
class MaskedAttentionBlock(nn.Module):
    
    def __init__(
        self, 
        in_channels,
    ):
        
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self, 
        x,
        mask,
    ):
        
        batch_has_padding = torch.logical_not(mask).any().item()  # bool

        if batch_has_padding:
            # Masking
            x = x * mask  # (B, in_channels, T) * (B, 1, T)
            # Conv1D for attention
            attention = self.conv1d(x)  # (B, in_channels, T) -> (B, 1, T)
            # Masking
            attention = attention * mask # (B, 1, T) * (B, 1, T)

            # Reshape
            batch_size, _, _ = attention.shape
            attention = attention.reshape(batch_size, -1)  # (B, T)

            # Softmax with a mask
            # In theory it remains a tiny non-zero value, so when the softmax is normalized to sum to 1 a very slight rounding error may be introduced. In practice, however, this is negligible.
            bool_mask = mask.squeeze(1).to(torch.bool)
            neg_val = torch.finfo(attention.dtype).min
            attention = attention.masked_fill(~bool_mask, neg_val)
            attention = torch.softmax(attention, dim=1)  # softmax over T. (B, T)
            attention = attention.reshape(batch_size, 1, -1)  # (B, 1, T)
            attention = attention * mask # (B, 1, T) * (B, 1, T)

        else:
            attention = self.conv1d(x)  # (B, in_channels, T) -> (B, 1, T)
            batch_size, _, _ = attention.shape
            attention = attention.reshape(batch_size, -1)  # (B, T)
            attention = torch.softmax(attention, dim=1)  # softmax over T. (B, T)
            attention = attention.reshape(batch_size, 1, -1)  # (B, 1, T)

        return attention  # (B, 1, T)


class NoCalculationBlock(nn.Module):
    def __init__(
        self, 
    ):

        super().__init__()    

    def forward(
        self,
        x,
        *args,
    ):

        return x


class OutputBlock(nn.Module):
    def __init__(
        self,
        in_features,
        output_size,
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=output_size,
        )

    def forward(
        self,
        x,
    ):

        batch_size, _,  _ = x.shape  # (B, C, T)
        x = x.reshape(batch_size, -1)  # (B, C, T) -> (B, C*T)
        label = self.linear(x)  # (B, C*T) -> (B, output_size)
        
        # Do not use softmax function here.
        # nn.CrossEntropyLoss() will be used later.
        # https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch

        return label  # (B, output_size)
    

class FlexibleBranch(nn.Module):
    def __init__(
        self, 
        layers, 
        output_size,
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
            if layer_type == 'MaskedConv1dBlock':
                layer = MaskedConv1dBlock(   
                    in_channels=layer_info['in_channels'],
                    out_channels=layer_info['out_channels'],
                    kernel_size=layer_info['temporal_kernel_size'],
                    padding=int((layer_info['temporal_kernel_size'] - 1) / 2),
                    dropout_probability=layer_info['dropout_probability'],
                )
            elif layer_type == 'MaskedLSTMBlock':
                layer = MaskedLSTMBlock(
                    input_size=layer_info['input_size'],
                    hidden_size=layer_info['hidden_size'],
                    dropout_probability=layer_info['dropout_probability'],
                )
            elif layer_type == 'MaskedAttentionBlock':
                layer = MaskedAttentionBlock(
                    in_channels=layer_info['in_channels'],
                )
            elif layer_type == 'OutputBlock':
                layer = OutputBlock(
                    in_features=layer_info['in_features'],  # in_features: (out_channels*num_node*num_timestamp)
                    output_size=output_size,
                )
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            self.layers.append(layer)
            self.layertypes.append(layer_type)

    def forward(
        self, 
        x,
        mask,
        lengths,
    ):
        self.attention = None  # initialize attention
        for layer_type, layer in zip(self.layertypes, self.layers):
            if layer_type == 'MaskedConv1dBlock':
                # x: (B, C, T)
                x = layer(x, mask)
            elif layer_type == 'MaskedLSTMBlock':
                x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
                x = layer(x, lengths)
                x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
            elif layer_type == 'MaskedAttentionBlock':
                # x: (B, C, T) * (B, 1, T) -> (B, C, T)
                attention = layer(x, mask)  # get the attention mask (B, 1, T)
                x = x * attention
                self.attention = attention  # keep the attention mask
            elif layer_type == 'OutputBlock':
                # x: (B, C, T) -> (B, output_size)
                output = layer(x)
            else:
                raise ValueError("Unsupported layer type for branch")
        return output, self.attention


class Model(nn.Module):

    def __init__(
        self, 
        before_branch_config, 
        branch_config,
        output_size,
        in_channels_pose,  # 0 or a natural number. If you do not have pose data, you can set 0. eg., Set in_channels = 3 when your data have speed, distance_from_initial_position, and pose data (head, back, tail). Set in_channels = 2 when your data have speed and distance_from_initial_position. 
    ):

        super().__init__()

        # Linear for pose
        self.linear_for_pose = nn.Linear(
            in_features=in_channels_pose,
            out_features=1,
            bias=False,
        )
        self.relu = nn.ReLU()

        # Before branching
        self.before_branch = nn.ModuleList()
        if before_branch_config:
            for layer_info in before_branch_config:
                layer_type = layer_info['layertype']
                if layer_type == 'MaskedConv1dBlock':
                    self.before_branch.append(
                        MaskedConv1dBlock(   
                            in_channels=layer_info['in_channels'],
                            out_channels=layer_info['out_channels'],
                            kernel_size=layer_info['temporal_kernel_size'],
                            padding=int((layer_info['temporal_kernel_size'] - 1) / 2),
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
            )
            self.branches.append(branch)

    def forward(
        self,
        x,
        x_pose,
        has_pose,
    ):
        
        # x: (B, C, T)
        # x_pose: 
            # if has_pose == True: (B, T, in_channels_pose).
            # if has_pose == False: (B)
        # has_pose: bool

        # Make a mask for the input x. False is a padded-part.
        all_zeros_along_F = torch.all(x == 0, dim=1)  # Check if all values along the F axis are 0
        mask = torch.logical_not(all_zeros_along_F)   # (B, T). False if all values in F are 0, True otherwise
            # ex. (2, 3, 4)
                # [
                # [[0, 1, 1, 0],
                #  [1, 1, 1, 1],
                #  [1, 1, 1, 0]],
                # [[1, 0, 0, 0],
                #  [0, 0, 1, 0],
                #  [1, 1, 0, 0]]
                # ]

                # [
                #  [False, False, False, False],
                #  [False, False, False, True]
                # ]

                # [
                #  [ True,  True,  True,  True],
                #  [ True,  True,  True, False]
                # ]

        # Calculate actual length of each sequence for LSTM.
        lengths = mask.sum(dim=1)  # B. eg. [4, 3].

        # Reshape
        mask = mask.unsqueeze(1)  # (B, 1, T)

        # Make pose to one channel and concat it to x
        if has_pose:
            x_pose = self.linear_for_pose(x_pose)  # (B, T, 1)
            x_pose = x_pose.permute(0, 2, 1)  # (B, 1, T)
            x_pose = self.relu(x_pose)
            x = torch.cat((x, x_pose), dim=1)  # (B, in_channels-1, T) -cat- (B, 1, T) -> (B, in_channels, T)
        else:
            x = x  # (B, in_channels, T)


        # Before branching
        for layer in self.before_branch:
            x = layer(x, mask)
        
        # After branching
        outputs = []
        attentions = []
        for branch in self.branches:
            x_branch = x.clone()
            # output: (B, output_size)
            # attention: (B, 1, T)
            output, attention = branch(x_branch, mask, lengths)
            outputs.append(output)  # outputs: length number_of_attention_branches
            attentions.append(attention)

        # attentions: (B, number_of_attention_branches, T)
        attentions = torch.cat(attentions, dim=1)
        
        return outputs, attentions