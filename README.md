<!-- #region -->
# Facexplainer
This is the official implement of Paper [FACExplainer: Generating Model-faithful Explanations for Graph Neural Networks Guided by Spatial Information]

## Prerequisites
```shell script
$ cd ./Facexplainer
$ source install.sh
``` 

For used datasets in this paper. We suggest to download them from their source code or any exsiting work that has it (we recommend DIG).

## Usage
For experiment in the paper, please see `run_explainers.sh`




### Remark
Our baseline code is heavily based on the DIG library with small changes. We summarize the difference in `difference_from_dig.md`

### Reference
The `gnn_explainer`, `pgexplainer`, and `subgraphx` implementations are based on the DIG library.

https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph

The `gstarx` implementaiton is based on the official Gstarx code.

https://github.com/ShichangZh/GStarX

The `orphicx` implementaiton is based on the official OrphicX code, where datasets processing and GNNs are changed to align with DIG.

https://github.com/WanyuGroup/CVPR2022-OrphicX
<!-- #endregion -->
