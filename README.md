# iTag
Implementation of the paper **An Integral Tag Recommendation Model for Textual Content**.

## Intro
ITag is a deep neural network model used to recommend tags for textual contents.
It take text as input and outputs top K recommendations.

In this repository, we provide the implementation of our ITag model with Python Keras APIs.

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

## Repository contents

| file | description |
| ------ | ------ |
| itag.py | The main code of our model, including the training and prediction of the ITag model. |
| attention.py | Implementation of the Attention Mechanism. |
|shared_dataset.py|The file is used to load data. |
|shared.txt|The file records the mapping between the id of the co-occurring words in the text vocabulary and the id in the tag vocabulary.|
|utils.py|This file helps to obtain the one-hot representation of  tags.|

## Datasets
We didn't provide our data sets for the data of Stack Overflow, Ask Ubuntu and Mathematics is public.
You could use your own data sets in numpy format. 

Here are details:

In shared_dataset.py, 'brs' refers to texts, 'sfs' refers to tags and 'ms' refers to masks. 'ms' is actually not used, so you could set it randomly, like [1, 0, 0, 0, 1, 1] and keep the length of 'ms' equals to the length of 'brs'.

shared.txt contains a dictionary, which map the same word in texts and tags. In shared.txt, key is the id of a word and value is the id of a tag which is the same as the word.

And finally, this example may help you:

```
shared.txt:

{3:1,1:2}


example.py:

import numpy as np

import shared_dataset as sh

def create_data():

    brs = np.array([[3, 1, 2, 4],[3,2,1,6],[6,7,1,5]])

    ms = np.array([[0,1,0,1],[0,1,1,1],[1,1,0,0]])

    sfs = np.array([[1,2],[1,2],[1]])                                 

    np.savez('test.npz', brs=brs, ms=ms, sfs=sfs)

def main():

    (en_train, ms_train, de_train, y_train), (en_test, ms_test, de_test, y_test) = sh.load_data(path='test.npz', num_words=10000, num_sfs=1003)                      print(en_train)

    print(ms_train)

    print(de_train)

    print(y_train)

    print('--------------------------------')

    print(en_test)

    print(ms_test)

    print(de_test)

    print(y_test)

if __name__ == '__main__':

    create_data()

    main()
```

## Model
![Image text](https://github.com/SoftWiser-group/iTag/blob/master/images/structure.png)
### Three pillars:
**(1) sequential text modeling** meaning that the intrinsic sequential ordering as well as different areas of text might have an important implication on the corresponding tag(s),

**(2) tag correlation** meaning that the tags for a certain piece of textual content are often semantically correlated with each other,

**(3) content-tag overlapping** meaning that the vocabularies of content and tags are overlapped.

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
