# Self-supervised Rewiring of Pre-trained Speech Encoders
*The code is for Findings-EMNLP 2022 paper: [Self-supervised Rewiring of Pre-trained Speech Encoders: Towards Faster Fine-tuning with Less Labels in Speech Processing](https://aclanthology.org/2022.findings-emnlp.141.pdf)*
## Environment and Dataset
Our work is developed based on [SUPERB benchmark](https://github.com/s3prl/s3prl). Please follow their instructions to set up environment, download dataset and preprocess data.
## Models training
```
cd s3prl/upstream/wav2vec2_hug
```
```
# Neutral Version
python n_mirror.py

# Twin Version
python sumavg-1ep.py

# Mixed Version
python n_mirror_mix.py
```
After training models, please import them in ***s3prl/upstream/wav2vec2_hug/expert.py***
## Fine-tune models on downstream Tasks
We are following SUPERB setting, please select the downstream tasks according to its structions.
```
python3 run_downstream.py -n [output_name] -m train -u wav2vec2_hug_large_ll60k -d [task]
```
## Citation
```
@inproceedings{yang2022self,
  title={Self-supervised Rewiring of Pre-trained Speech Encoders: Towards Faster Fine-tuning with Less Labels in Speech Processing},
  author={Yang, Hao and Zhao, Jinming and Haffari, Gholamreza and Shareghi, Ehsan},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2022},
  pages={1952--1959},
  year={2022}
}
```
