import os
import torch
import hydra
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from tqdm import tqdm
from dataset import get_dataset, get_dataloader
from gnnNets import get_gnnNets
from utils import check_dir, get_logger, evaluate_scores_list, PlotUtils, fidelity_normalize_and_harmonic_mean
from pipeline.baselines_utils import evaluate_related_preds_list
from torch_geometric.utils import add_self_loops, add_remaining_self_loops
import time

from methods.facexplainer import FACExplainer

IS_FRESH = False

@hydra.main(config_path="../config", config_name="config")
def main(config):
    # Set config
    cwd = os.getcwd()
    config.datasets.dataset_root = os.path.join(cwd, "datasets")
    config.models.gnn_saving_path = os.path.join(cwd, "checkpoints")
    config.explainers.explanation_result_path = os.path.join(cwd, "results")

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    
    explainer_name = config.explainers.explainer_name
    
    log_file = (
        f"{explainer_name}_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.debug(OmegaConf.to_yaml(config))
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda", index=config.device_id)
    else:
        device = torch.device("cpu")

    # Load dataset
    dataset = get_dataset(
        dataset_root=config.datasets.dataset_root,
        dataset_name=config.datasets.dataset_name,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    dataloader_params = {
        "batch_size": config.models.param.batch_size,
        "random_split_flag": config.datasets.random_split_flag,
        "data_split_ratio": config.datasets.data_split_ratio,
        "seed": config.datasets.seed,
    }
    dataloader = get_dataloader(dataset, **dataloader_params)
    test_indices = dataloader["test"].dataset.indices
    if config.datasets.data_explain_cutoff > 0:
        test_indices = test_indices[: config.datasets.data_explain_cutoff]
        
    if config.explainers.subgraph_building_method == "split":
        config.models.param.add_self_loop = False   
    # Load model
    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        model_config=config.models,
    )

    state_dict = torch.load(
        os.path.join(
            config.models.gnn_saving_path,
            config.datasets.dataset_name,
            f"{config.models.gnn_name}_"
            f"{len(config.models.param.gnn_latent_dim)}l_best.pth",
        )
    )["net"]

    model.load_state_dict(state_dict)

    model.to(device)

    explanation_saving_path = os.path.join(
        config.explainers.explanation_result_path,
        config.datasets.dataset_name,
        config.models.gnn_name,
        explainer_name,
    )

    check_dir(explanation_saving_path)
    
    explainer = FACExplainer(
        model, 
        model.hook_layer, 
        device, 
        subgraph_building_method=config.explainers.subgraph_building_method 
    )
    plot_utils = PlotUtils(config.datasets.dataset_name, is_show=False)
    related_preds_list = []
    s_time = time.time() 
    for i, data in enumerate(tqdm(dataset[test_indices])):
        
        idx = test_indices[i]
        data.to(device)
        data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        related_preds = explainer.explain(
                data,
                config.explainers.sparsity
            )
        
        #print(idx, int(data.y), related_preds['explanation'], data.x.shape[0])
        
        # from utils import to_networkx
        # explained_example_plot_path = os.path.join(
        #         explanation_saving_path, f"fig/example_{idx}.png"
        #     )
        # plot_utils.plot_ba2motifs(
        #     to_networkx(data), related_preds['explanation'], title_sentence='0', figname=explained_example_plot_path
        # )
        
        related_preds_list += [related_preds]
        
    cost_time = time.time() - s_time
    metrics = evaluate_related_preds_list(related_preds_list, logger)
    metrics_str = ",".join([f"{m: .4f}" for m in metrics])
    
    metrics_str += f'{cost_time: .4f}, {len(test_indices)}'
    print(metrics_str)    
        
        
        

if __name__ == "__main__":
    import sys

    sys.argv.append("explainers=facexplainer")
    main()
    

    








