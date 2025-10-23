
# HypeAssign

This repository contains the code for the paper **"HypeAssign: Hypergraph Contrastive Learning for Issue Assignment"**.

## Overview

![img](file:///C:\Users\dell\Documents\Tencent Files\2770553150\nt_qq\nt_data\Pic\2025-08\Thumb\8ec72f45a88a1a307fd150e3aa864ae4_720.png)
> We propose a novel approach named Hypergraph Contrastive Learning for Issue Assignment (HypeAssign). HypeAssign comprises three components: 1) Hypergraph Construction. We first extract the multiple relationships among issues, developers, and source code files to construct a general heterogeneous graph, which is then further transformed into a hypergraph structure. The constructed hypergraph consists of several subgraphs, each representing a distinct sub-hypergraph snapshot, namely issue sub-hypergraph, developer sub-hypergraph, and file sub-hypergraph. 2) Representation Learning. This stage captures the global high-order relational structures in the hypergraph and the local neighborhood information within the general heterogeneous graph. A cross-view contrastive learning module is further introduced to align these two levels of representations and enhance their mutual consistency. 3) Recommendation. This stage calculates the inner product between the learned representations of issues and developers to predict the probability of each developer being assigned to a given issue.



## Usage

```
python main.py
```

## Requirements

```
python 3.8.10
pandas==2.0.1
scipy==1.10.1
torch==2.0.1
networkx==3.1
numpy==1.24.3
tqdm==4.65.0
```

