# iTag
Implementation of the paper **An Integral Tag Recommendation Model for Textual Content**.

## Intro
ITag is a deep neural network model used to recommend tags for posts in community question answering websites.
It take text as input and outputs top K recommendations.

In this repository, we provide implementation of ITag model with Python Keras APIs.

## Requirements
+ Python >= 2.7
+ Numpy >= 1.14.3
+ Tensorflow >= 1.8.0
+ Keras >= 2.1.6

## Usage
example usage:
```
python itag.py
```
This command will lead to a full training and prediction process.
And at the same time, the weights of the model will be stored in the file **itag.h5**.

If you want to evaluation the predicted results on the metrics **Recall@n**, **Precision@n** and **F1@n**,
modify the parameter **MAX_LENGTH** in the itag.py because it is used to set the value of **n**.

## Repository contents

| file | description |
| ------ | ------ |
| itag.py | The main code of our model, including the training and prediction of the iTag model. |
| attention.py | Implementation of the Attention Mechanism. |
|shared_dataset.py|The file is used to load data. |
|shared.txt|The file records the mapping between the id of the co-occurring words in the text vocabulary and the id in the tag vocabulary.|
|utils.py|This file helps to obtain the one-hot representation of  tags.|

## Model
![Image text](https://github.com/SoftWiser-group/iTag/blob/master/images/structure.png)
### Three pillars:
**(1) sequential text modeling** meaning that the intrinsic sequential ordering as well as different areas of text might have an important implication on the corresponding tag(s),

**(2) tag correlation** meaning that the tags for a certain piece of textual content are often semantically correlated with each other,

**(3) content-tag overlapping** meaning that the vocabularies of content and tags are overlapped

## Citing
If you find ITag useful in your research, we ask that you cite the following paper:

```
@inproceedings{
author = {Shijie Tang, Yuan Yao, Suwei Zhang, Feng Xu, Tianxiao Gu, Hanghang Tong, Xiaohui Yan, Jian Lu},
title = {An Integral Tag Recommendation Model for Textual Content},
booktitle = {Proceedings of the 33rd National Conference on Artificial Intelligence},
year = {2019},
}
```