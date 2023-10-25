conda create -y -n facexplainer python=3.7.11
source activate facexplainer
conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install scipy
CUDA="cu101" #specify your own version
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-geometric==1.6.0
pip install cilog typed-argument-parser==1.5.4 captum==0.2.0 shap IPython tqdm
pip install networkx
conda install -y -c conda-forge rdkit