## DraftRec: Personalized Draft Recommendation for Winning in Multi-Player Online Battle Arena Game (WWW' 2022): Official Project Webpage
This repository provides the official PyTorch implementation of the corresponding paper: [DraftRec](link)

> Authors: Hojoon Lee*, Dongyoon Hwang*, Hyunseung Kim, Byungkun Lee, and Jaegul Choo,<br>
> Affiliation: KAIST AI

![](./assets/draftrec_model.png)


> **Abstract:** 
This paper presents a personalized character recommendation system for Multiplayer Online Battle Arena (MOBA) games which are considered as one of the most popular online video game genres around the world. 
When playing MOBA games, players go through a *draft* stage, where they alternately select a virtual character to play. 
When drafting, players select characters by not only considering their character preferences, but also the synergy and competence of their team's character combination. 
However, the complexity of drafting induces difficulties for beginners to choose the appropriate characters based on the characters of their team while considering their own champion preferences.
To alleviate this problem, we propose DraftRec, a novel hierarchical model which recommends characters by considering each player's champion preferences and the interaction between the players.
DraftRec consists of two networks: the player network and the match network. 
The player network captures the individual player's champion preference, and the match network integrates the complex relationship between the players and their respective champions. 
We train and evaluate our model from a manually collected 280,000 matches of *League of Legends* and a publicly available 50,000 matches of *Dota2*. 
Empirically, our method achieved state-of-the-art performance in character recommendation and match outcome prediction task. 
Furthermore, a comprehensive user survey confirms that DraftRec provides convincing and satisfying recommendations.

## Dataset

[user_history_data (~16gb)](https://davian-lab.quickconnect.to/d/s/o9mpSh3FqCKvOVvkEO5NauhID5OGhRc7/uyXjCG-PHwyJv17OSlAsZSUIzTjg50xt-tbBgS4dDbAk)

![](./assets/dataset.png)

Write brief dataset description (dy)
Write you can find the detail in jupyter blah blah.

## Model checkpoints for the trained models

The pre-trained models can be found below. 

|                             Model checkpoint                             |      ACC        |     HR@10    | 
|--------------------------------------------------------------------------|-----------------|--------------|
|[DraftRec (hidden_dim=64, seq_len=10)](link1)                             |      -          |              |
|[DraftRec (hidden_dim=64, seq_len=50)](link2)                             |      55.1       |     86.8     |
|[DraftRec (hidden_dim=128, seq_len=10)](link3)                            |      -          |              |
|[DraftRec (hidden_dim=128, seq_len=50)](link4)                            |      -          |              |

## Environment setup

Our code can run on a *single* GPU or on *multi*-GPUs.
See requirements.txt for all prerequisites, and you can also install them using the following command.

```
pip install -r requirements.txt
```

## Training

To train the draftrec model with a *single* GPU, try the following command:

```
python main.py --template draftrec --dataset_path [DATASET_PATH]
```

To train the draftrec model with *multiple* GPUs, try the following command:

```
python main.py --template draftrec --dataset_path [DATASET_PATH] --use_parallel true
```

## Cite

```
@article{lee2022draftrec,
  title={DraftRec: Personalized Draft Recommendation for Winning in Multi-Player Online Battle Arena Games},
  author={Hojoon Lee and Dongyoon Hwang and Hyunseung Kim and Byungkun Lee and Jaegul Choo},
  booktitle = {Proceedings of the Web Conference 2022},
  series = {WWW '22},
  year={2022}
}
```

## Disclaimer

This is not an official Riot Games product.