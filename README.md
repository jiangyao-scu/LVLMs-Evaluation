# Effectiveness Assessment of Recent Large Vision-Language Models

This is the source code and result of our "Effectiveness Assessment of Recent Large Vision-Language Models".

# Evaluation

## Download the data
The data used in our testbed can be downloaded from the following links:
DUTS link
SOC  link
COD10K link
Trans10K link
ColonDB link
ETIS link
ISIC link
MVTec AD link
VisA link



## Training TLFNet
* Modify the "train_data_location" and "eval_data_location" in "train.py" according to the path of the data.
* Download the pre-trained [swin Transformer](https://drive.google.com/file/d/1-T0G3esLOQb4c_vkzl40VgXof2OSSCJZ/view?usp=sharing) or [PVT](https://drive.google.com/file/d/1be31x92t0jKcx2eonpkTLjMD5opLQAl2/view?usp=sharing).
* Start to train Swin Transformer-based TLFNet with
```sh
python -m torch.distributed.launch --nproc_per_node=2 train.py --model_path path/to/save/trained/model/ --log_path path/to/save/log/ --backbone swin --pretrained_model path/of/pre-trained/swin-Transformer/ --image_size 224
```
or train PVT-based TLFNet with
```sh
python -m torch.distributed.launch --nproc_per_node=2 train.py --model_path path/to/save/trained/model/ --log_path path/to/save/log/ --backbone pvt --pretrained_model path/of/pre-trained/PVT/ --image_size 256
```

## Testing TLFNet
* We have released pre-computed saliency maps of TLFNet based on the Swin Transformer and PVT. Please retrieve the results from the following links: [TLFNet-swin](https://drive.google.com/file/d/1-0tb13jeDmygn18QeGgM6jfgtqyuwumZ/view?usp=sharing) and [TLFNet-pvt](https://drive.google.com/file/d/1ssT-NB9vlPQ0rHJGrwU2N0EaefYX8Bn-/view?usp=sharing).
* We have also released the trained weights of TLFNet. You can download them from the following links: [TLFNet-wsin](https://drive.google.com/file/d/19Q67GoRr6N93jOvoq29o6Hqwb1yEPzga/view?usp=sharing) and [TLFNet-pvt](https://drive.google.com/file/d/1MUG1H0W6e7uij6VPht2nmWU2-VypYf2G/view?usp=sharing).
* To generate saliency maps, you will need to modify the "eval_data_location" in the "test.py" according to your data's path. Then, you can generate the saliency maps with:
```sh
python test.py --save_path path/to/save/saliency-maps/ --backbone swin --model_path path/of/pre-trained/TLFNet.pth/ --image_size 224
```
or 
```sh
python test.py --save_path path/to/save/saliency-maps/ --backbone pvt --model_path path/of/pre-trained/TLFNet_PVT.pth/ --image_size 256
```
*It should be noted that, owing to an equipment malfunction, the original PVT-based TLFNet data was unfortunately lost. We subsequently retrained this model and obtained results that closely resemble the initial outcomes. This newly obtained experimental result does not alter the conclusions drawn in this paper.*<br>

# Light Field Salient Object Autofocus
We are building an online service for "Light Field Salient Object Autofocus". Please stay tuned for our upcoming release.

# Citation
Please cite our paper if you find the work useful: 

        @article{Jiang2024TLFNet,
        author={Jiang, Yao and Li, Xin and Fu, Keren and Zhao, Qijun},
        journal={IEEE Transactions on Image Processing}, 
        title={Transformer-Based Light Field Salient Object Detection and Its Application to Autofocus}, 
        year={2024},
        volume={33},
        pages={6647-6659}}
