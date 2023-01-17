# Graph-Neural-Network
Title | Original medium link | Implementation | 
--- | --- | --- | 
Gnn_Social_Network | [graph neural networks for social networks using pytorch](https://dev.to/awadelrahman/tutorial-graph-neural-networks-for-social-networks-using-pytorch-2kf) | [My implementation](Gnn_for_social_network/SocialGNN.ipynb) |
Uni of Ams Tutorial | [Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html) | [My implementation](uva_tutorial/Tutorial.ipynb) | 

# Install 
Build docker image
```
cd dockerfiles && docker build -t gnn_image -f gnn_dockerfile . && cd -
```

Build docker container
```
docker run -it --rm --name gnn_container --gpus=all --shm-size 8G --volume="$PWD:/workspace/" gnn_image:latest /bin/bash
```