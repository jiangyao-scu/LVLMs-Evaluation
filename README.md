# Effectiveness Assessment of Recent Large Vision-Language Models

This is the source code and result of our "Effectiveness Assessment of Recent Large Vision-Language Models".

# Evaluation

## Download the data
The data used in our testbed can be downloaded from the following links: [DUTS](http://saliencydetection.net/duts/), [SOC](http://dpfan.net/SOCBenchmark/), [COD10K](https://dengpingfan.github.io/pages/COD.html), [Trans10K](https://xieenze.github.io/projects/TransLAB/TransLAB.html), [ColonDB](http://vi.cvc.uab.es/colon-qa/cvccolondb/), [ETIS](https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb), [ISIC](https://challenge.isic-archive.com/data/), [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://paperswithcode.com/dataset/visa)



## Recognition
* Modify the prompt in "client_shikra.py", "demo_v2_minigpt.py", and the file in LLaVA, and run these files to get the recognition results of Shikra, MiniGPT-v2, and LLaVA-1.5.
* Evaluate the results using "CLS_Metrics.py".


## Localization
* Modify the prompt in "client_shikra.py", "demo_v2_minigpt.py", and the file in LLaVA, and run these files to get the detection results of Shikra, MiniGPT-v2, and LLaVA-1.5.
* Evaluate the results using "Detection_Metrics.py" for MiniGPT-v2 and LLaVA-1.5, and "Detection_Metrics_Shikra.py" exclusively for Shikra.
* Get segmentation results from these models using the above detection results and "seg_with_SAM.py".
* These results are then evaluated using segmentation evaluation [metrics](https://github.com/ArcherFMY/sal_eval_toolbox).

# Citation
Please cite our paper if you find the work useful: 

        @article{jiang2024effectiveness,
        title={Effectiveness assessment of recent large vision-language models},
        author={Jiang, Yao and Yan, Xinyu and Ji, Ge-Peng and Fu, Keren and Sun, Meijun and Xiong, Huan and Fan, Deng-Ping and Khan, Fahad Shahbaz},
        journal={Visual Intelligence},
        volume={2},
        number={1},
        pages={17},
        year={2024},
        publisher={Springer}
        }
