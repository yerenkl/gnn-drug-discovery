import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.models import DimeNetPlusPlus
import torch.nn as nn

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
