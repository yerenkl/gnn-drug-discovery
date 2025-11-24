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