# Norm-Augmented Graph AutoEncoders for Link Prediction

Official Implementation of Norm-Augmented Graph AutoEncoders for Link Prediction.



## Dependencies

- torch
- torch_geometric



## Datasets

| Dataset  | #Nodes | #Edges | #Features | Mean Degree |
| -------- | ------- | ------- | ---------- | ----------- |
| Cora     | 2,708   | 5,278   | 1,433      | 3.9         |
| CiteSeer | 3,327   | 4,552   | 3,703      | 2.7         |
| PubMed   | 19,717  | 44,324  | 500        | 4.5         |
| CoraFull | 19,793  | 63,421  | 8,710      | 6.4         |



## Training

```bash
python GAE-NA.py --dataset=Cora --threshold=4 --val_ratio=0.05 --test_ratio=0.15 --encoder=GCN

python GAE-NA.py --dataset=CiteSeer --threshold=3 --val_ratio=0.05 --test_ratio=0.15 --encoder=GCN

python GAE-NA.py --dataset=PubMed --threshold=2 --val_ratio=0.05 --test_ratio=0.15 --encoder=GCN

python GAE-NA.py --dataset=CoraFull --threshold=2 --val_ratio=0.05 --test_ratio=0.15 --encoder=GCN
```



## Acknowledgements

The code is implemented based on [pytorch_geometric/examples/autoencoder](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py) and [GNAE](https://github.com/SeongJinAhn/VGNAE).



