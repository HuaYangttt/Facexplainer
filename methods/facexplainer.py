import os
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.optim import Adam
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn.conv import MessagePassing

from .shapley import gnn_score, GnnNetsGC2valueFunc, sparsity




class FACExplainer(object):
    def __init__(self, model, layer, device, subgraph_building_method):
        self.model = model
        self.layer = layer
        self.device = device
        #self.subgraph_building_func = get_graph_build_func(subgraph_building_method)
        self.subgraph_building_method = subgraph_building_method
        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        self.model_num_hops = k
        
        self.gradients = dict()
        self.activations = dict()
        
        def backward_hook(module, grad_input, grad_output):
            if torch.cuda.is_available():
              self.gradients['value'] = grad_output[0].cuda()
            else:
              self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            if torch.cuda.is_available():
              self.activations['value'] = output.cuda()
            else:
              self.activations['value'] = output
            return None
        
        self.layer.register_forward_hook(forward_hook)
        self.layer.register_backward_hook(backward_hook)
    
    def explain(self, data, sparsity):
        data = data.clone().to(self.device)
        exp_size = max(int((1 - sparsity)*int(data.x.shape[0])),1)
        ori_logit, ori_prob, ori_prediction, ori_score = self.forward_pass(data, require_grad=True)
        activations = self.activations['value']
        grad = torch.abs(self.gradients['value']).mean(dim=0)
        
        weighted_act = (activations * grad).sum(dim=-1)
        #print(weighted_act)
        
        weighted_act_sorted, node_indexes_sorted = torch.sort(weighted_act, descending=True)
        
        
        nx_graph = self.construct_graph(data)
        node_indexes_sorted_list = node_indexes_sorted.cpu().tolist()
        search_list = node_indexes_sorted_list[0:5]
        # vis_saving_dir = '/home/hua.yang/Faceplainer/GStarX-main/imgs'
        # pos = nx.kamada_kawai_layout(nx_graph)
        # nodelist = [int(node) for node in search_list]
        # nx.draw_networkx_nodes(nx_graph, pos,
        #                        nodelist=list(nx_graph.nodes()),
        #                        node_color='#FFA500',
        #                        node_size=300)
        # nx.draw_networkx_edges(nx_graph, pos, width=3, edge_color='gray', arrows=False)
        # nx.draw_networkx_labels(nx_graph, pos, { i:str(i) for i in list(nx_graph.nodes())})
        # figname = os.path.join(vis_saving_dir, f'example_.png')
        # plt.savefig(figname)
        count = 0
        for tar_idx in range(exp_size):
            tar_node = node_indexes_sorted_list[tar_idx]
            
            alters = nx.single_source_shortest_path_length(nx_graph, int(tar_node), 3).keys()
            alters = [node for node in alters if node not in node_indexes_sorted_list[:exp_size]]
            #alters = node_set
            # print('neighbors:', node_set)
            
            search_map = [node_indexes_sorted_list[:]]
            for alter in alters:
                basic_subgraph = node_indexes_sorted_list[:]
                alter_idx = basic_subgraph.index(alter)
                
                basic_subgraph[alter_idx] = tar_node
                basic_subgraph[tar_idx] = alter
                search_map.append(basic_subgraph)
                
            #debug_n_v('search_map:', search_map)
            search_result_dict = []
        
            for nodes in search_map:
                count+=1
                masked_score, maskout_score, sparsity_score = self.Score_fuc_F_FI(nodes[:exp_size], data, prediction=ori_prediction)
                #print(nodes[:exp_size], masked_score, maskout_score)
                search_result_dict.append(( masked_score-maskout_score , masked_score, maskout_score, nodes, sparsity_score)) 
                
                #search_result_dict.append(( masked_score , masked_score, maskout_score, nodes, sparsity_score)) #ablation study
            
            best_score, masked_score, maskout_score, best_nodes, sparsity_score= self.find_best_result(search_result_dict, max= True)
            node_indexes_sorted_list = best_nodes
        #assert 0   
        result_info = {
            'explanation': node_indexes_sorted_list[:exp_size],
            "masked": masked_score,
            "maskout": maskout_score,
            "origin": ori_prob[:, ori_prediction].item(),
            "sparsity": sparsity_score,
        }
        
        return result_info        
    
    def find_best_result(self, target, max=True):
        if max:
            d_order=sorted(target, key=lambda x:x[0], reverse=True)
        else:
            d_order=sorted(target, key=lambda x:x[0], reverse=False)
        
        return d_order[0]
            
           
    
    
    def Score_fuc_F_FI(self, tager_nodes_list, data, prediction):
        # important structure to keep
        masked_node_list = tager_nodes_list 
        # spurious part to keep
        maskout_node_list = [node for node in range(data.x.shape[0]) if node not in masked_node_list]
        
        value_func = GnnNetsGC2valueFunc(self.model, target_class=prediction)
        
        masked_score = gnn_score(
            masked_node_list,
            data,
            value_func=value_func,
            subgraph_building_method=self.subgraph_building_method,
        )

        maskout_score = gnn_score(
            maskout_node_list,
            data,
            value_func=value_func,
            subgraph_building_method=self.subgraph_building_method,
        )

        sparsity_score = sparsity(
            masked_node_list,
            data,
            subgraph_building_method=self.subgraph_building_method,
        )
        
        
        return masked_score, maskout_score, sparsity_score
            
            
        
            
        
        
        
    
    def construct_graph(self, data):
        graph = nx.Graph()
        assert isinstance(data,Data)
        for i in range(data.x.shape[0]):
            graph.add_node(i)
        for i in range(data.edge_index.shape[1]):
            graph.add_edge(int(data.edge_index[0][i]),int(data.edge_index[1][i]))
        return graph       
        
        
    
    def forward_pass(self, data, target_class=None, require_grad =False):
        if not require_grad:
            with torch.no_grad():
                batch_data = Batch.from_data_list([data])
                logits = self.model(batch_data)
                probs = F.softmax(logits, dim=-1)
                if target_class == None:
                    prediction = probs.squeeze().argmax(-1).item()
                    score = probs[:,prediction]
                else:
                    score = probs[:,target_class]
                
                return logits, probs, target_class, score
        elif require_grad:
            batch_data = Batch.from_data_list([data])
            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(self.model.parameters(), lr=0.01, weight_decay=0.0)
            logits = self.model(batch_data)
            probs = F.softmax(logits, dim=-1)
            loss = criterion(logits, batch_data.y)
            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            if target_class == None:
                prediction = probs.squeeze().argmax(-1).item()
                score = probs[:,prediction]
            else:
                score = probs[:,target_class]
            
            return logits, probs, prediction, score  
        
         
        
        



















