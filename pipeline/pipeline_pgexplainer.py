import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from dataset import get_dataset, get_dataloader
from gnnNets import get_gnnNets
from utils import check_dir, get_logger, PlotUtils
from methods import PGExplainer, PGExplainer_edges
from pipeline.baselines_utils import evaluate_related_preds_list
import time

IS_FRESH = False


@hydra.main(config_path="../config", config_name="config")
def pipeline(config):
    # cwd = os.path.dirname(os.path.abspath(__file__))
    # pwd = os.path.dirname(cwd)
    
    cwd = os.getcwd()

    config.datasets.dataset_root = os.path.join(cwd, "datasets")
    config.models.gnn_saving_path = os.path.join(cwd, "checkpoints")
    config.explainers.explanation_result_path = os.path.join(cwd, "results")
    config.log_path = os.path.join(cwd, "log")

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]

    explainer_name = config.explainers.explainer_name
    log_file = (
        f"{explainer_name}_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.debug(OmegaConf.to_yaml(config))

    if torch.cuda.is_available():
        device = torch.device("cuda", index=config.device_id)
    else:
        device = torch.device("cpu")

    dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {
            "batch_size": config.models.param.batch_size,
            "random_split_flag": config.datasets.random_split_flag,
            "data_split_ratio": config.datasets.data_split_ratio,
            "seed": config.datasets.seed,
        }
        loader = get_dataloader(dataset, **dataloader_params)
        train_indices = loader["train"].dataset.indices
        test_indices = loader["test"].dataset.indices

        if config.datasets.data_explain_cutoff > 0:
            test_indices = test_indices[: config.datasets.data_explain_cutoff]

    else:
        train_indices = range(len(dataset))

    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=dataset.num_classes,
        model_config=config.models,
    )
    eval_model = get_gnnNets(
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
    eval_model.load_state_dict(state_dict)

    model.to(device)
    eval_model.to(device)
    if config.models.param.graph_classification:
        input_dim = config.models.param.gnn_latent_dim[-1] * 2
    else:
        input_dim = config.models.param.gnn_latent_dim[-1] * 3

    pgexplainer = PGExplainer(
        model,
        in_channels=input_dim,
        device=device,
        explain_graph=config.models.param.graph_classification,
        epochs=config.explainers.param.ex_epochs,
        lr=config.explainers.param.ex_learing_rate,
        coff_size=config.explainers.param.coff_size,
        coff_ent=config.explainers.param.coff_ent,
        t0=config.explainers.param.t0,
        t1=config.explainers.param.t1,
    )
    explanation_saving_path = os.path.join(
        config.explainers.explanation_result_path,
        config.datasets.dataset_name,
        config.models.gnn_name,
        explainer_name,
    )

    check_dir(explanation_saving_path)

    pgexplainer_saving_path = os.path.join(
        explanation_saving_path, f"{explainer_name}.pth"
    )
    s_time = time.time()
    if not IS_FRESH and os.path.isfile(pgexplainer_saving_path):
        logger.info("Load saved PGExplainer model...")
        state_dict = torch.load(
            pgexplainer_saving_path, map_location=torch.device("cpu")
        )
        pgexplainer.load_state_dict(state_dict)
    else:
        if config.models.param.graph_classification:
            pgexplainer.train_explanation_network(dataset[train_indices[60:]])
        else:
            pgexplainer.train_explanation_network(dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    train_end_time = time.time() 
    train_using_time = train_end_time - s_time
    pgexplainer_edges = PGExplainer_edges(
        pgexplainer=pgexplainer, model=eval_model, molecule=True
    )
    pgexplainer_edges.device = pgexplainer.device

    plot_utils = PlotUtils(config.datasets.dataset_name, is_show=False)
    related_preds_list = []
    for i, data in enumerate(tqdm(dataset[test_indices])):
        idx = test_indices[i]
        data.to(device)
        prediction = model(data).softmax(dim=-1).argmax().item()
        edge_masks, hard_edge_masks, related_preds = pgexplainer_edges(
            data.x,
            data.edge_index,
            num_classes=dataset.num_classes,
            sparsity=config.explainers.sparsity,
        )

        related_preds = related_preds[prediction]
        hard_edge_masks = hard_edge_masks[prediction]
        related_preds_list += [related_preds]
        #print(idx, related_preds['explanation'])
        

        if config.save_plot:
            logger.debug(f"Plotting example {idx}.")
            from utils import fidelity_normalize_and_harmonic_mean, to_networkx
            from pipeline.baselines_utils import hard_edge_masks2coalition

            coalition = hard_edge_masks2coalition(
                data, hard_edge_masks, config.models.param.add_self_loop
            )
            f = related_preds["origin"] - related_preds["maskout"]
            inv_f = related_preds["origin"] - related_preds["masked"]
            sp = related_preds["sparsity"]
            n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)
            title_sentence = f"fide: {f:.3f}, inv-fide: {inv_f:.3f}, h-fide: {h_f:.3f}"

            if hasattr(dataset, "supplement"):
                words = dataset.supplement["sentence_tokens"][str(idx)]
            else:
                words = None

            explained_example_plot_path = os.path.join(
                explanation_saving_path, f"example_{idx}.png"
            )
            plot_utils.plot(
                to_networkx(data),
                coalition,
                x=data.x,
                words=words,
                title_sentence=title_sentence,
                figname=explained_example_plot_path,
            )
    cost_time = time.time() - train_end_time
    metrics = evaluate_related_preds_list(related_preds_list, logger)
    metrics_str = ",".join([f"{m}" for m in metrics])
    metrics_str += f'{cost_time: .4f}, {len(test_indices): .1f}, {train_using_time: .4f}'
    print(metrics_str)


if __name__ == "__main__":
    import sys

    sys.argv.append("explainers=pgexplainer")
    pipeline()
