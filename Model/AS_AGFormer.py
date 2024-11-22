import torch
import numpy as np
import math
from torch import nn
from timm.models.layers import DropPath
from collections import OrderedDict
from Model.S_AGCN_modules import TCN_GCN_unit, Graph, TemporalModelBase
from Model.AGFormer_modules import Attention, GCN, MLP, MultiScaleTCN

class S_AGCN(TemporalModelBase):
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=256, connections=None):

        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        layers_tcngcn = []
        num_person = 1
        in_channels = in_features
        num_point = num_joints_in
        self.graph = Graph(num_joints_in, connections=connections)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        A = self.graph.A
        self.expand_gcn = TCN_GCN_unit(in_features, channels, A)
        self.in_features = in_features
        self.causal_shift = []
        next_dilation = filter_widths[0]
        for i in range(0, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            layers_tcngcn.append(TCN_GCN_unit(channels, channels, A))
            layers_tcngcn.append(TCN_GCN_unit(channels, channels, A, stride=filter_widths[i], residual=False))
            next_dilation *= filter_widths[i]

        self.layers_tcngcn = nn.ModuleList(layers_tcngcn)
        self.fc = nn.Conv1d(channels, 3, 1)

    def set_bn_momentum(self, momentum):
        self.data_bn.momentum = momentum
        self.expand_gcn.gcn1.bn.momentum = momentum
        self.expand_gcn.tcn1.bn.momentum = momentum
        for layer in self.layers_tcngcn:
            layer.gcn1.bn.momentum = momentum
            layer.tcn1.bn.momentum = momentum

    def _forward_blocks(self, x):
        B, J, T = x.size()
        j = J//self.in_features # number of 2D pose joints
        x = self.data_bn(x)
        x = x.view(B, 1, j, self.in_features, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B, self.in_features, T, j)

        x = self.expand_gcn(x)
        for i in range( len(self.pad) -1):
            res = x[:, :, self.causal_shift[i] + self.filter_widths[i]//2 :: self.filter_widths[i], :]

            x = self.drop(self.layers_tcngcn[2*i](x))
            x = self.drop(self.layers_tcngcn[2*i+1](x))
            x = res + x
        pose_3d_ = x
        device = x.device if x.is_cuda else 'cpu'
        pose_3d = torch.from_numpy(np.full((B, 3, j),0).astype('float32')).to(device)
        for i in range(0,j):
            pose_joint_3d = pose_3d_[:,:,:,i].mean(2)
            pose_joint_3d = self.fc(pose_joint_3d.view(B,-1,1))
            pose_3d[:,:,i] = pose_joint_3d.view(B,-1)

        return pose_3d


class AGFormerBlock(nn.Module):
    """
    Implementation of AGFormer block.
    """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                     num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                     mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                     temporal_connection_len=1, neighbour_num=4, n_frames=81, connections=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixer_type == 'attention':
            self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                    proj_drop=drop, mode=mode)
        elif mixer_type == 'graph':
            self.mixer = GCN(dim, dim,
                                num_nodes=17 if mode == 'spatial' else n_frames,
                                neighbour_num=neighbour_num,
                                mode=mode,
                                use_temporal_similarity=use_temporal_similarity,
                                temporal_connection_len=temporal_connection_len,
                                connections=connections)
        elif mixer_type == "ms-tcn":
            self.mixer = MultiScaleTCN(in_channels=dim, out_channels=dim)
        else:
            raise NotImplementedError("AGFormer mixer_type is either attention or graph")
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                        act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MotionAGFormerBlock(nn.Module):
    """
    Implementation of MotionAGFormer block. It has two ST and TS branches followed by adaptive fusion.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=2, n_frames=81, connections=None):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim

        # ST Attention branch
        self.att_spatial = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                         qk_scale, use_layer_scale, layer_scale_init_value,
                                         mode='spatial', mixer_type="attention",
                                         use_temporal_similarity=use_temporal_similarity,
                                         neighbour_num=neighbour_num,
                                         n_frames=n_frames,
                                         connections=connections)
        self.att_temporal = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qk_scale, use_layer_scale, layer_scale_init_value,
                                          mode='temporal', mixer_type="attention",
                                          use_temporal_similarity=use_temporal_similarity,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,
                                          connections=connections)

        # ST Graph branch
        if graph_only:
            self.graph_spatial = GCN(dim, dim,
                                     num_nodes=17,
                                     mode='spatial',
                                     connections=connections)
            if use_tcn:
                self.graph_temporal = MultiScaleTCN(in_channels=dim, out_channels=dim)
            else:
                self.graph_temporal = GCN(dim, dim,
                                          num_nodes=n_frames,
                                          neighbour_num=neighbour_num,
                                          mode='temporal',
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                          connections=connections)
        else:
            self.graph_spatial = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                               qkv_bias,
                                               qk_scale, use_layer_scale, layer_scale_init_value,
                                               mode='spatial', mixer_type="graph",
                                               use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len,
                                               neighbour_num=neighbour_num,
                                               n_frames=n_frames,
                                               connections=connections)
            self.graph_temporal = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                                qkv_bias,
                                                qk_scale, use_layer_scale, layer_scale_init_value,
                                                mode='temporal', mixer_type="ms-tcn" if use_tcn else 'graph',
                                                use_temporal_similarity=use_temporal_similarity,
                                                temporal_connection_len=temporal_connection_len,
                                                neighbour_num=neighbour_num,
                                                n_frames=n_frames,
                                                connections=connections)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.hierarchical:
            B, T, J, C = x.shape
            x_attn, x_graph = x[..., :C // 2], x[..., C // 2:]

            x_attn = self.att_temporal(self.att_spatial(x_attn))
            x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_attn))
        else:
            x_attn = self.att_temporal(self.att_spatial(x))
            x_graph = self.graph_temporal(self.graph_spatial(x))

        if self.hierarchical:
            x = torch.cat((x_attn, x_graph), dim=-1)
        elif self.use_adaptive_fusion:
            alpha = torch.cat((x_attn, x_graph), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]
        else:
            x = (x_attn + x_graph) * 0.5

        return x


class AS_AGFormer(nn.Module):
    """
    AS-MotionAGFormer model
    """

    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4., act_layer=nn.GELU,
                 attn_drop=0., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, num_heads=8, qkv_bias=False, qkv_scale=None, hierarchical=False,
                 num_joints=17, use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False,
                 graph_only=False, neighbour_num=2, n_frames=81, filter_widths=[3, 3, 3, 3], channels=256,
                 dropout=0.2, connections=None):
        super().__init__()

        # Embedding input
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))

        self.layers = nn.Sequential(
            *[
                MotionAGFormerBlock(
                    dim=dim_feat,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path * i / max(n_layers - 1, 1),
                    num_heads=num_heads,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    qkv_bias=qkv_bias,
                    qk_scale=qkv_scale,
                    use_adaptive_fusion=use_adaptive_fusion,
                    hierarchical=hierarchical,
                    use_temporal_similarity=use_temporal_similarity,
                    temporal_connection_len=temporal_connection_len,
                    use_tcn=use_tcn,
                    graph_only=graph_only,
                    neighbour_num=neighbour_num,
                    n_frames=n_frames,
                    connections=connections
                )
                for i in range(n_layers)
            ]
        )

        # S_AGCN module
        self.s_agcn = S_AGCN(
            num_joints_in=num_joints,
            in_features=dim_feat,
            num_joints_out=num_joints,
            filter_widths=filter_widths,
            causal=False,
            dropout=dropout,
            channels=channels,
            connections=connections
        )

        # Head
        self.norm = nn.LayerNorm(dim_feat)
        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))
        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x, return_rep=False):
        """
        Forward pass.
        :param x: Tensor input [B, T, J, C]
        :param return_rep: representation output instead of 3D output [B, T, J, C_rep].
        """
        x = self.joints_embed(x)
        x = x + self.pos_embed
        x = self.layers(x)
        x = self.norm(x)
        x_local = self.s_agcn(x)
        x = self.rep_logit(x)

        if return_rep:
            return x
        x = self.head(x)
        return x, x_local

