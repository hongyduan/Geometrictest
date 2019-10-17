import torch
from layer_specific_to_common import *
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class Net(torch.nn.Module):
    def __init__(self, in_dim, out_dim, x_g1, data_G2_x, data_G2_num_nodes, data_G2_num_edges):
        super(Net, self).__init__()
        self.x_g1 = Parameter(x_g1)
        self.layer_s_to_c = STCConv(in_dim, out_dim)

        self.x_g2 = Parameter(data_G2_x)
        self.layer_g2_rgcn = RGCNConv(data_G2_num_nodes, data_G2_num_nodes, data_G2_num_edges, num_bases=30)

    def forward(self, data_G1, data_G2):
        edge_index = data_G1.edge_index
        x_g1 = self.layer_s_to_c(self.x_g1, edge_index)
        x_g1 = F.relu(x_g1)

        x_g2 = self.layer_g2_rgcn(self.x_g2, data_G2.edge_index, data_G2.edge_type)
        x_g2 = F.relu(x_g2)

        # return F.log_softmax(x_g1, dim=1)

        return F.log_softmax(x_g1, dim=1), F.log_softmax(x_g2, dim=1)



