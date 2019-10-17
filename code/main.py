from layer_specific_to_common import STCConv
from torch_geometric.data import Data
from load_data import *
from pre_data import *
from net import *
import argparse



# x: has shape [N, in_channels]
# edge_index: has shape [2, E]
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--type_path', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/type_embedding/node_embedding.npy")
    parser.add_argument('--entity_path', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/entity_embedding/node_embedding.npy")
    parser.add_argument('--entity_relation_path', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/entity_embedding/node_re_embedding.npy")
    parser.add_argument('--data_path', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/yago_result")
    parser.add_argument('--data_path_bef', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/yago")
    parser.add_argument('--G2_val_file_name', type=str, default="val_entity_Graph.txt")
    parser.add_argument('--G2_test_file_name', type=str, default="test_entity_Graph.txt")

    parser.add_argument('--dim', type=int, default=500)
    parser.add_argument('--in_dim', type=int, default=500)
    parser.add_argument('--out_dim', type=int, default=500)
    parser.add_argument('--leaf_node_entity', action='store_true', default=True)

    return parser.parse_args(args)

def pre_load_data(args):

    # load_data
    # node2id_G2_re: 26078;   relation2id_G2_re: 34;    type_node2id_G1_re: 911;   G1_graph_sub3: 106;     G1_graph_sub2: 8948;     G1_graph_sub1: 894;     G2_links: 228619
    node2id_G2_re, relation2id_G2_re, type_node2id_G1_re, G1_graph_sub3, G1_graph_sub2, G1_graph_sub1, val_data, test_data = load_da(args.data_path, args.data_path_bef)

    # pre_data
    # all_node_embedding_: 26989*500;       # G1_node_embedding_: [0]:911*500     [1]:106*500    [2]:9962*500       # G2_node_embedding_: 26078*500;
    all_node_embedding_ = all_node_embedding(args.entity_path, args.type_path)
    G1_node_embedding_ = G1_node_embedding(args.type_path, all_node_embedding_, G1_graph_sub3)
    G1_node_embedding_type_, G1_node_embedding_type_small_, G1_ndoe_embedding_entity_ = G1_node_embedding_
    G2_node_embedding_ = G2_node_embedding(args.entity_path)


    # load train data: G1
    # x:[N, in_channels];    edge_index:[2,E];    G1_graph_sub2(en is_instance_of ty)  G1_graph_sub1(ty1 is_a ty2)
    edge_index_G1_, edge_index_G1_sub2_, edge_index_G1_sub1_ = edge_index_G1(G1_graph_sub2, G1_graph_sub1)
    if args.leaf_node_entity:
        # G1_x = torch.cat((G1_node_embedding_type_, G1_ndoe_embedding_entity_), 0)  # (911+8948)*500
        G1_x = all_node_embedding_
        G1_edge_index = edge_index_G1_ # (8962+8467) edges
    else:
        G1_x = G1_node_embedding_type_  # 911*500
        G1_edge_index = edge_index_G1_sub1_  # 8962 edges
    data_G1 = Data(x = G1_x, edge_index = G1_edge_index.t().contiguous())


    # load train data: G2
    # G2_graph (en1 relaiton en2) undirected
    G2_x = G2_node_embedding_  # 26078*500
    G2_edge_index, G2_edge_attr = edge_index_attr_G2(args.data_path, args.entity_relation_path, node2id_G2_re, relation2id_G2_re, args.dim)  # 332127 edges
    data_G2 = Data(x = G2_x, edge_index = G2_edge_index.t().contiguous(), edge_attr = G2_edge_attr)


    #    val_data = (triples_val_G1, triples_val_G2, G2_links_val)
    #    test_data = (triples_test_G1, triples_test_G2, G2_links_test)
    # load val data: G1     # load test data: G1
    G1_edge_index_val, G1_edge_index_test = edge_index_G1_val_test(val_data[0],test_data[0])
    data_G1_val = Data(edge_index = G1_edge_index_val.t().contiguous())
    data_G1_test = Data(edge_index = G1_edge_index_test.t().contiguous())

    # load val data: G2

    G2_edge_index_val, G2_edge_attr_val = edge_index_attr_G2_val_test(args.data_path,args.entity_relation_path,node2id_G2_re, relation2id_G2_re, args.dim, args.G2_val_file_name)
    data_G2_val = Data(edge_index = G2_edge_index_val.t().contiguous(), edge_attr = G2_edge_attr_val)

    # load test data: G2
    G2_edge_index_test, G2_edge_attr_test = edge_index_attr_G2_val_test(args.data_path,args.entity_relation_path,node2id_G2_re, relation2id_G2_re, args.dim, args.G2_test_file_name)
    data_G2_test = Data(edge_index = G2_edge_index_test.t().contiguous(), edge_attr = G2_edge_attr_test)


    return all_node_embedding_, G1_node_embedding_, G2_node_embedding_, data_G1, data_G2, data_G1_val, data_G1_test, data_G2_val, data_G2_test


def loss(out):
    return 1

def score():
    return 1


def main(args):

    # all_node_embedding_: 26989*500;  # G1_node_embedding_: [0]:911*500     [1]:106*500    [2]:8948*500 # G2_node_embedding_: 26078*500;
    all_node_embedding_, G1_node_embedding_, G2_node_embedding_, data_G1, data_G2, data_G1_val, data_G1_test, data_G2_val, data_G2_test = pre_load_data(args)

    print("for debug")

    # device = torch.device('cuda' if torch.cuda.is_available() else'cpu')

    model = Net(args.in_dim, args.out_dim, data_G1.x, data_G2.x, data_G2.num_nodes, data_G2.num_edges)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(2000):
        optimizer.zero_grad()

        # out_G1 = model(data_G1, data_G2)
        out_G1, out_G2 = model(data_G1, data_G2)
        loss = torch.mean(out_G2**2)
        loss.backward()
        optimizer.step()
        em_check = torch.mean(out_G2)
        print('loss: {}'.format(loss))
        print('em_mean: {}'.format(em_check))
        # loss = torch.mean(out_G1**2)
        # # loss = loss(out_G1, out_G2)
        # loss.backward()
        # optimizer.step()
        # em_check = torch.mean(out_G1)
        # print('loss: {}'.format(loss))
        # print('em_mean: {}'.format(em_check))

    print("for debug...")


    model.eval()

    # test

if __name__ == '__main__':
    main(parse_args())
