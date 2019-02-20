# r-gcn
PyToch implementation of R-GCN model for link prediction

## DistMult Results
[emb_dim = 50, Adam(lr=0.01)]
- Head prediction: Hits@10: 0.31941223273687597, MR: 283.63520170642107
- Tail prediction: Hits@10: 0.38800765180884023, MR: 215.88703424692321
- Average: Hits@10: 0.35370994227285812, MR: 249.76111797667215

## Obvious improvements
- Multiple negative sampling
- Normalize embeddings
- L2 regularization on relation embeddings
- Use Adagrad and hyperparametter tuning
- Filter the rankings (efficiently!)

Requirements: rdflib, wget, scipy, sklearn
