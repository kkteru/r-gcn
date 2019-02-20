# r-gcn
PyToch implementation of R-GCN model for link prediction

## DistMult Results
[emb_dim = 50, Adam(lr=0.01)]
- Head prediction: Hits@10: 0.31941223273687597, MR: 283.63520170642107
- Tail prediction: Hits@10: 0.38800765180884023, MR: 215.88703424692321
- Average: Hits@10: 0.35370994227285812, MR: 249.76111797667215

[emb_dim = 100, Adam(lr=0.01)]
- Head prediction: Hits@10: 0.38211643615310387, MR: 301.09246499974608
- Tail prediction: Hits@10: 0.45479169135447173, MR: 236.47879670227354
- Average: Hits@10: 0.4184540637537878, MR: 268.78563085100984

## Obvious improvements
- Multiple negative sampling
- Normalize embeddings
- L2 regularization on relation embeddings
- Use Adagrad and hyperparametter tuning
- Filter the rankings (efficiently!)
