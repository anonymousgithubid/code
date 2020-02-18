# Multi-Graph Transformer for Free-Hand Sketch Recognition

![](https://img.shields.io/badge/language-Python-{green}.svg)
![](https://img.shields.io/npm/l/express.svg)

<div align=center><img src="https://github.com/anonymousgithubid/code/blob/master/figures/cat.gif" width = 30% height = 30% /></div>

<div align=center><img src="https://github.com/anonymousgithubid/code/blob/master/figures/cat_graph.png"/></div>


This code repository is the source code of the paper "Multi-Graph Transformer for Free-Hand Sketch Recognition".


<div align=center><img src="https://github.com/anonymousgithubid/code/blob/master/figures/MGT_pipeline_details.png"/></div>


## Requirements
Ubuntu 16.04.10

Anaconda 4.7.10

Python 3.7

PyTorch 1.2.0

How to install (clone) our environment will be detailed in the following.
First of all, please install Anaconda.

Our hardware environment: 2 Intel(R) Xeon(R) CPUs (E5-2690  v4  @ 2.60GHz), 128 GB RAM, 4 GTX 1080 Ti GPUs.

All the following codes can run on single GTX 1080 Ti GPU.

## Usage (How to Train Our MGT)

```
# 1. Choose your workspace and download our repository.
cd ${CUSTOMIZED_WORKSPACE}
git clone https://github.com/anonymousgithubid/code

# 2. Enter the directory.
cd code

# 3. Clone our environment, and activate it.
conda-env create --name ${CUSTOMIZED_ENVIRONMENT_NAME} --file ./MGT_environment.yml
conda activate ${CUSTOMIZED_ENVIRONMENT_NAME}

# 4. Download our training/evaluation/testing datasets and the associated URL lists from our Google Drive folder. Then extract them into './dataloader' folder. data_anonymous.tar.gz is 116MB, and its MD5 checksum is a7509d9470019b4c6b80f5936b10acaf.
cd ./dataloader
chmod +x download_anonymous.sh
./download_anonymous.sh
# If this script 'download_anonymous.sh' can not work for you, please manually download data_anonymous.tar.gz to current path via this link https://drive.google.com/open?id=1d3_WRfjY9fbAXrdnzJ21DakkxfeGG1Z9 .
tar -zxvf data_anonymous.tar.gz
rm -f data_anonymous.tar.gz
cd ..

# 5. Train our MGT. Please see details in our code annotations.
# Please set the input arguments based on your case.
# When the program starts running, a folder named 'experimental_results/${CUSTOMIZED_EXPERIMENT_NAME}' will be created automatically to save your log, checkpoint, and TensorBoard curves.
python train_MGT17.py 
    --exp ${CUSTOMIZED_EXPERIMENT_NAME}   
    --batch_size ${CUSTOMIZED_SIZE}   
    --num_workers ${CUSTOMIZED_NUMBER} 
    --gpu ${CUSTOMIZED_GPU_NUMBER}

# Actually, we got the performance of MGT #17 (reported in Table 3 in our paper) by running the following command.
python train_MGT17.py 
    --exp train_MGT17_001   
    --batch_size 192   
    --num_workers 12 
    --gpu 1

```

## Our Experimental Results
In order to fully demonstrate the traits of our MGT to both graph and sketch researchers,
we will provide the codes of all our ablative models reported in our paper.
We also provide our experimental results including trainging log files, model checkpoints, and TensorBoard curves. The following table provides the download links in Google Drive, which is corresponding to the Table 3 in our paper.  

"GT #1" is the original Transformer [[Vaswani *et al*.]](https://arxiv.org/abs/1706.03762), representing each input graph as a fully-connected graph.  
"GT #7" is a Transformer variant, representing each input graph as a sparse graph, *i.e.*, A^{2-hop} structure defined in our paper.  
"MGT #13" is an ablative variant of our MGT, representing each input graph as two sparse graphs, *i.e.*, A^{2-hop} and A^{global}.  
**“MGT #17”** is the full model of our MGT, representing each input graph as three sparse graphs, *i.e.*, A^{1-hop}, A^{2-hop}, and A^{global}.  
In the following table and diagram, we can see that multiple sparsely-connected graphs improve the performance of Transformer.  
Please see details in our paper.

Network | acc. | ckpts & TensorBoard curves | training script
:-: | :-: | :-: | :-
GT #1 | 0.5249 | [link](https://drive.google.com/open?id=1Bm5EFXaQKrD8dKjADCR_KiuqVQrqsX9V), 49M, MD5 checksum 8ea24042db67078368b7aef801a6d3c1. | [train_GT1.py](https://github.com/anonymousgithubid/code/blob/master/train_GT1.py) 
GT #7 | 0.7082 | [link](https://drive.google.com/open?id=1EsRcPfEwKEdeR--pacuzHc3xxp3Xm638), 49M, MD5 checksum 476498d02ddc15722a7f68b024446997. | [train_GT7.py](https://github.com/anonymousgithubid/code/blob/master/train_GT7.py)
MGT #13 | 0.7237| [link](https://drive.google.com/open?id=10Gza4AYaQuGzu0xxE_KlcK-TOBOEeZmT), 99M, MD5 checksum 643ddbfec020885c4ac1b1192f14accf. | [train_MGT13.py](https://github.com/anonymousgithubid/code/blob/master/train_MGT13.py)
**MGT #17** |  **0.7280**| [link](https://drive.google.com/open?id=1oEqNUo_E_6gKP9NB2qOMnmcvoBNJLMlI), 141M, MD5 checksum 6be5dee690eb38db15518bc008207407. | [train_MGT17.py](https://github.com/anonymousgithubid/code/blob/master/train_MGT17.py) 

<div align=center><img src="https://github.com/anonymousgithubid/code/blob/master/figures/accuracy.gif" width = 60% height = 60% /></div>



## Usage (How to Run Our Baselines)

### Run CNN Baselines

```
# Based on the aforementioned operations and environment configurations.
# For brevity, we only provide the code for the two CNN baselines with best performance, i.e., Inceptionv3 and MobileNetv2.

# 1. Enter the 'dataloader' directory.
cd ${CUSTOMIZED_WORKSPACE}/code/dataloader/

# 2. Download the training/evaluation/testing datasets (.PNG files) and the associated URL lists from our Google Drive folder. Then extract them into 'dataloader' folder. data_4_cnnbaselines.tar.gz is 558MB, and its MD5 checksum is 8f1132b400eb2bd9186f7f02d5c4d501.
chmod +x download_4_cnnbaselines.sh
./download_4_cnnbaselines.sh
tar -zxvf data_4_cnnbaselines.tar.gz
rm -f data_4_cnnbaselines.tar.gz
cd data_4_cnnbaselines
tar -zxvf tiny_train_set.tar.gz
rm -f tiny_train_set.tar.gz
tar -zxvf tiny_val_set.tar.gz
rm -f tiny_val_set.tar.gz
tar -zxvf tiny_test_set.tar.gz
rm -f tiny_test_set.tar.gz

# 3. Switch directory and run the scripts.
# Please set the input arguments based on your case.
# When the program starts running, a folder named 'experimental_results/${CUSTOMIZED_EXPERIMENT_NAME}' will be created automatically into ${CUSTOMIZED_WORKSPACE}/code/baselines/cnn_baselines/, to save your log, checkpoint, and TensorBoard curves.
python train_inceptionv3.py 
    --exp ${CUSTOMIZED_EXPERIMENT_NAME}   
    --batch_size ${CUSTOMIZED_SIZE}   
    --num_workers ${CUSTOMIZED_NUMBER} 
    --gpu ${CUSTOMIZED_GPU_NUMBER}

# Actually, we got the performance of Inceptionv3 (reported in Table 2 in our paper) by running the following command.
python train_inceptionv3.py 
    --exp train_inceptionv3_001   
    --batch_size 64   
    --num_workers 12 
    --gpu 0

```

### Run RNN Baselines

```
# Based on the aforementioned operations and environment configurations.
# For brevity, we only provide the code for the RNN baseline with best performance, i.e., bidirectional GRU.

# 1. Switch directory and run the scripts.
cd ${CUSTOMIZED_WORKSPACE}/code/baselines/rnn_baselines/
# Please set the input arguments based on your case.
# When the program starts running, a folder named 'experimental_results/${CUSTOMIZED_EXPERIMENT_NAME}' will be created automatically into ${CUSTOMIZED_WORKSPACE}/code/baselines/rnn_baselines/, to save your log, checkpoint, and TensorBoard curves.
python train_bigru.py 
    --exp ${CUSTOMIZED_EXPERIMENT_NAME}   
    --batch_size ${CUSTOMIZED_SIZE}   
    --num_workers ${CUSTOMIZED_NUMBER} 
    --gpu ${CUSTOMIZED_GPU_NUMBER}

# Actually, we got the performance of Bi-directional GRU (reported in Table 2 in our paper) by running the following command.
python train_bigru.py 
    --exp train_bigru_001   
    --batch_size 256   
    --num_workers 12 
    --gpu 0

```


## License
This project is licensed under the MIT License.

## Acknowledgement
Many thanks to the great sketch dataset [**Quick, Draw！**](https://github.com/googlecreativelab/quickdraw-dataset) released by Google.

