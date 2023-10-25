import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx
import os.path as osp

   
def load_redditbinary(data_path):
    #print('loading data')
    # root = 'benchmarks/xgraph/datasets'
    # data_name = 'REDDITBINARY'
    g_list = []
    label_dict = {}
    feat_dict = {}
    
    with open(data_path, 'r') as f:
        n_g = int(f.readline().strip()) # how many graph
        for i in range(n_g):
            node_tags = []
            g = nx.Graph()
            n_edges = 0
            row = f.readline().strip().split()
            n, l = [int(w) for w in row] # n is num of nodes, l is graph_level label
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                row = [int(w) for w in row]
                node_tags.append(1)
                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
                    #print(j, row[k])
            
            x = torch.ones((g.number_of_nodes(),1), dtype=torch.float)
            edge_index = [[a,b] for a,b in g.edges()]       
            edge_index = torch.LongTensor(edge_index).transpose(-1,-2)
            edge_index_rev = edge_index[[1,0],:]
            edge_index = torch.cat([edge_index,edge_index_rev], dim=-1)  
            data_g = Data(x=x,y=l,edge_index=edge_index)
            g_list.append(data_g)
    return g_list
    # for g in g_list:
    #     print(g.number_of_nodes())
    #     print(sorted(g.edges()))
    #     assert 0


class redditbinary(InMemoryDataset):
    def __init__(self, root, num_per_class=1000, transform=None, pre_transform=None):
        self.name = 'redditbinary'
        self.num_per_class = num_per_class
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        return f"redditbinary.txt"
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data_path = osp.join(self.root, 'redditbinary/raw/redditbinary.txt')
        data_list = load_redditbinary(data_path)
        
        torch.save(self.collate(data_list), self.processed_paths[0])
        

    
if __name__ == '__main__':
    load_redditbinary()
        
        
    
            
                    
                    
                
                
                
                
                
            
        
    
        
    