import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GINConv,  
    GATConv,  
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
    SAGPooling,
    BatchNorm,
    LayerNorm,
    InstanceNorm,
    GraphNorm
)
from torch.nn import GroupNorm
from torch_geometric.nn.models import DimeNetPlusPlus

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x

class DimeNetpp(nn.Module):
    def __init__(self, hidden_channels = 64, num_blocks = 6):
        super(DimeNetpp, self).__init__()
        self.dimenet = DimeNetPlusPlus(hidden_channels=hidden_channels, 
                                        out_channels=1, 
                                        num_spherical=7,
                                        num_radial=6, 
                                        num_blocks=num_blocks,
                                        int_emb_size=64,
                                        basis_emb_size=8,
                                        out_emb_channels=256)
    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch

        x = self.dimenet(z, pos, batch)

        return x

class GIN3LayerGraphNormTrainEps(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__()

        mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        mlp3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )

        self.conv1 = GINConv(mlp1, train_eps=True)
        self.gn1 = GraphNorm(hidden_channels)

        self.conv2 = GINConv(mlp2, train_eps=True)
        self.gn2 = GraphNorm(hidden_channels)

        self.conv3 = GINConv(mlp3, train_eps=True)
        self.gn3 = GraphNorm(hidden_channels)

        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.gn1(x, batch)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.gn2(x, batch)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.gn3(x, batch)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

class GAT3LayerGraphNorm(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 hidden_channels: int = 64,
                 heads: int = 4):
        super().__init__()

        out_per_head = hidden_channels // heads  # 64 / 4 = 16


        self.conv1 = GATConv(
            in_channels=num_node_features,
            out_channels=out_per_head,
            heads=heads,
            concat=True,         
        )
        self.gn1 = GraphNorm(hidden_channels)
        self.conv2 = GATConv(
            in_channels=hidden_channels,
            out_channels=out_per_head,
            heads=heads,
            concat=True,          
        )
        self.gn2 = GraphNorm(hidden_channels)
        self.conv3 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=1,
            concat=False,         
        )
        self.gn3 = GraphNorm(hidden_channels)

        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.gn1(x, batch)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.gn2(x, batch)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.gn3(x, batch)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

class ImprovedGCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        hidden_channels=128,
        num_layers=3,
        dropout=0.2,
        edge_dropout=0.0,
        normalization='batch',
        activation='relu',
        pooling='mean',
        use_residual=True,
        aggregation='mean',
        num_groups=8
    ):
        super(ImprovedGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.use_residual = use_residual
        self.normalization_type = normalization
        self.activation_type = activation
        self.pooling_type = pooling
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels, aggr=aggregation))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, aggr=aggregation))
        
        # Normalization layers
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if normalization == 'batch':
                self.norms.append(BatchNorm(hidden_channels))
            elif normalization == 'layer':
                self.norms.append(LayerNorm(hidden_channels))
            elif normalization == 'instance':
                self.norms.append(InstanceNorm(hidden_channels))
            elif normalization == 'graph':
                self.norms.append(GraphNorm(hidden_channels))
            elif normalization == 'group':
                self.norms.append(GroupNorm(num_groups, hidden_channels))
            elif normalization == 'none':
                self.norms.append(nn.Identity())
            else:
                raise ValueError(f"Unknown normalization: {normalization}")
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Pooling layer
        if pooling == 'set2set':
            self.pool = Set2Set(hidden_channels, processing_steps=3)
            pool_out_dim = 2 * hidden_channels
        elif pooling == 'attention':
            gate_nn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
            self.pool = GlobalAttention(gate_nn)
            pool_out_dim = hidden_channels
        elif pooling == 'sag':
            self.pool = SAGPooling(hidden_channels, ratio=0.5)
            pool_out_dim = hidden_channels
        else:  # mean, add, max
            self.pool = None
            pool_out_dim = hidden_channels
        
        # Output layer
        self.lin = nn.Linear(pool_out_dim, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # DropEdge
        if self.training and self.edge_dropout > 0:
            edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.edge_dropout
            edge_index = edge_index[:, edge_mask]
        
        # GCN layers
        for i in range(self.num_layers):
            x_prev = x if i > 0 else None
            
            # Convolution
            x = self.convs[i](x, edge_index)
            
            # Normalization
            x = self.norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if self.use_residual and x_prev is not None:
                x = x + x_prev
        
        # Graph-level pooling
        if self.pooling_type == 'set2set':
            x = self.pool(x, batch)
        elif self.pooling_type == 'attention':
            x = self.pool(x, batch)
        elif self.pooling_type == 'sag':
            x, _, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
            x = global_mean_pool(x, batch)
        elif self.pooling_type == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling_type == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling_type == 'max':
            x = global_max_pool(x, batch)
        
        x = self.lin(x)
        
        return x
