# iTag
Implementation of **An Integral Tag Recommendation Model for Textual Content**.

## Environment
Python 2.7

Tensorflow 1.8.0

Keras 2.1.6

## Repository contents

| file | description |
| ------ | ------ |
| itag.py | The main code of our model, including the training and prediction of the iTag model. |
| attention.py | Implementation of the Attention Mechanism. |
|shared_dataset.py|The file is used to load data. |
|shared.txt|The file records the mapping between the id of the co-occurring words in the text vocabulary and the id in the tag vocabulary.|
|utils.py|This file helps to obtain the one-hot representation of  tags.|

## Model
![Image text](https://github.com/SoftWiser-group/iTag/blob/master/images/new_structure.jpg)
### Three pillars:
**(1) sequential text modeling** meaning that the intrinsic sequential ordering as well as different areas of text might have an important implication on the corresponding tag(s),

**(2) tag correlation** meaning that the tags for a certain piece of textual content are often semantically correlated with each other,

**(3) content-tag overlapping** meaning that the vocabularies of content and tags are overlapped