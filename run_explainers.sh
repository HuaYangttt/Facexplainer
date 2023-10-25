# 7 methods: GradCAM, GNNExplainer, PGExplainer, SubgraphX, OrphicX, GStarX, FACExplainer
# 
# Sparsity: 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85

# FACExplainer
method=facexplainer
output_file="output/$method.txt" 
model='gcn'
cuda_id=4
for ds in   "bace" "bbbp" "graph_sst2" "mutag" "twitter" 'redditbinary'      #'collab' #'redditbinary' #"ba_2motifs" "bace" "bbbp" "graph_sst2" "mutag" "twitter"
do
    for sp in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done


# GNNExplainer
method=gnnexplainer
output_file="output/$method.txt" 
model='gcn'
cuda_id=2
for ds in  "ba_2motifs" # 'collab' 'redditbinary' "ba_2motifs" "bace" "bbbp" "graph_sst2" "mutag" "twitter"
do
    for sp in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done


# PGExplainer
method=pgexplainer
output_file="output/$method.txt" 
model='gcn'
cuda_id=5
for ds in  'collab' 'redditbinary' "ba_2motifs" "bace" "bbbp" "graph_sst2" "mutag" "twitter"
do
    for sp in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done

# GradCAM
method=gradcam
output_file="output/$method.txt" 
model='gcn'
cuda_id=3
for ds in  "ba_2motifs" #'collab' 'redditbinary' "ba_2motifs" "bace" "bbbp" "graph_sst2" "mutag" "twitter"
do
    for sp in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done

# OrphicX
method=orphicx
output_file="output/$method.txt" 
model='gcn'
cuda_id=3
for ds in      "ba_2motifs" 'redditbinary' #"bace" "bbbp" "graph_sst2" "mutag" "twitter" 'collab'
do
    for sp in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done

# GStarX
method=gstarx
output_file="output/$method.txt" 
model='gcn'
cuda_id=1
for ds in      "graph_sst2" #'collab' 'redditbinary' #"bace" "bbbp" "graph_sst2" "mutag" "twitter" "ba_2motifs" 
do
    for sp in 0.7 # 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done

# SubgraphX
method=subgraphx
output_file="output/test_$method.txt" 
model='gcn'
cuda_id=1
for ds in     'redditbinary' #"bace" #'collab' #'redditbinary' #"bace" "bbbp" "graph_sst2" "mutag" "twitter" "ba_2motifs" 
do
    for sp in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done




# GStarX
method=gstarx
output_file="output/fig_$method.txt" 
model='gcn'
cuda_id=1
for ds in      "graph_sst2" 
do
    for sp in  0.5 #0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp save_plot=True >> $output_file        
    done
done

# facexplainer
method=facexplainer
output_file="output/fig_$method.txt" 
model='gcn'
cuda_id=1
for ds in      "graph_sst2" 
do
    for sp in  0.55 #0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp save_plot=True >> $output_file        
    done
done

# SubgraphX
method=subgraphx
output_file="output/fig_$method.txt" 
model='gcn'
cuda_id=1
for ds in     "graph_sst2" #"bace" #'collab' #'redditbinary' #"bace" "bbbp" "graph_sst2" "mutag" "twitter" "ba_2motifs" 
do
    for sp in 0.5 #0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done

# OrphicX
method=orphicx
output_file="output/fig_$method.txt" 
model='gcn'
cuda_id=3
for ds in      "graph_sst2" #"ba_2motifs" 'redditbinary' #"bace" "bbbp" "graph_sst2" "mutag" "twitter" 'collab'
do
    for sp in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done


method=gradcam
output_file="output/fig_$method.txt" 
model='gcn'
cuda_id=3
for ds in  "graph_sst2" #"ba_2motifs" #'collab' 'redditbinary' "ba_2motifs" "bace" "bbbp" "graph_sst2" "mutag" "twitter"
do
    for sp in 0.5 #0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done

# GNNExplainer
method=gnnexplainer
output_file="output/fig_$method.txt" 
model='gcn'
cuda_id=2
for ds in  "graph_sst2" #"ba_2motifs" # 'collab' 'redditbinary' "ba_2motifs" "bace" "bbbp" "graph_sst2" "mutag" "twitter"
do
    for sp in 0.55 #0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done


# PGExplainer
method=pgexplainer
output_file="output/fig_$method.txt" 
model='gcn'
cuda_id=5
for ds in  "graph_sst2" #'collab' 'redditbinary' "ba_2motifs" "bace" "bbbp" "graph_sst2" "mutag" "twitter"
do
    for sp in 0.55 #0.55 0.6 0.65 0.7 0.75 0.8 0.85
    do
        printf "method: %s, " $method >> $output_file
        printf "dataset: %s, " $ds >> $output_file
        printf "sparsity: %s, " $sp >> $output_file
        CUDA_VISIBLE_DEVICES=$cuda_id python -m pipeline.pipeline_${method} models=$model datasets=$ds\
                                explainers.sparsity=$sp >> $output_file        
    done
done




