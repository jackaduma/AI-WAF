# **AI-WAF**

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/jackaduma/CycleGAN-VC2)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/jackaduma?locale.x=zh_XC)

[**中文说明**](./README.zh-CN.md) | [**English**](./README.md)

------

This code is a implementation of **Web-Application-Firewall** driven by deep learning model, a nice work on **AI-WAF**.

- [x] Dataset
  - [x] good and bad queries
- [x] Usage
  - [x] Training
  - [x] Example 
- [ ] Demo
- [x] Reference

------

## **Update**

------

**This repository contains:** 

1. [model code](./models) which implemented the paper.
2. [data loader code](./data_loader/datasets.py) you can use to load dataset for [training data](data).
3. [training scripts](train.py) to train the model.

------

## **Table of Contents**

- [**AI-WAF**](#ai-waf)
  - [**Update**](#update)
  - [**Table of Contents**](#table-of-contents)
  - [**Requirement**](#requirement)
  - [**Usage**](#usage)
    - [**train**](#train)
  - [**Trained Model Files**](#trained-model-files)
  - [**textcnn1**](#textcnn1)
  - [**textcnn2**](#textcnn2)
  - [**textcnn3**](#textcnn3)
  - [**Demo**](#demo)
  - [**Star-History**](#star-history)
  - [**Reference**](#reference)
  - [**Donation**](#donation)
  - [**License**](#license)
  
------



## **Requirement** 

```bash
pip install -r requirements.txt
```
## **Usage**


### **train** 
```python
python train.py -o 1 -b 8 -e 30
```

------

## **Trained Model Files**

a trained model which use private dataset, stored in **./cache** dir

## **textcnn1**

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 1024)]       0
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1024, 20)     1229180     input_1[0][0]
__________________________________________________________________________________________________
spatial_dropout1d (SpatialDropo (None, 1024, 20)     0           embedding[0][0]
__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, 1024, 32)     672         spatial_dropout1d[0][0]
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 1024, 32)     1952        spatial_dropout1d[0][0]
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 1024, 32)     3232        spatial_dropout1d[0][0]
__________________________________________________________________________________________________
max_pooling1d (MaxPooling1D)    (None, 512, 32)      0           conv1d[0][0]
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 512, 32)      0           conv1d_1[0][0]
__________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)  (None, 512, 32)      0           conv1d_2[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 512, 96)      0           max_pooling1d[0][0]
                                                                 max_pooling1d_1[0][0]
                                                                 max_pooling1d_2[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 49152)        0           concatenate[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 49152)        0           flatten[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          12583168    dropout[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 256)          0           dense[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 2)            514         dropout_1[0][0]
==================================================================================================
Total params: 13,818,718
Trainable params: 13,818,718
Non-trainable params: 0
```

```
Restoring model weights from the end of the best epoch.
Accuracy Score is:  0.9809785480291662
Precision Score is : 0.988558352402746
Recall Score is : 0.9707865168539326
F1 Score:  0.9795918367346939
AUC Score:  0.9804062247146184
```

## **textcnn2**

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 1024)]       0
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1024, 128)    7866368     input_1[0][0]
__________________________________________________________________________________________________
spatial_dropout1d (SpatialDropo (None, 1024, 128)    0           embedding[0][0]
__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, 1022, 8)      3080        spatial_dropout1d[0][0]
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 1020, 8)      5128        spatial_dropout1d[0][0]
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 1018, 8)      7176        spatial_dropout1d[0][0]
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 1016, 8)      9224        spatial_dropout1d[0][0]
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 1014, 8)      11272       spatial_dropout1d[0][0]
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 8)            0           conv1d[0][0]
__________________________________________________________________________________________________
global_max_pooling1d (GlobalMax (None, 8)            0           conv1d[0][0]
__________________________________________________________________________________________________
global_average_pooling1d_1 (Glo (None, 8)            0           conv1d_1[0][0]
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 8)            0           conv1d_1[0][0]
__________________________________________________________________________________________________
global_average_pooling1d_2 (Glo (None, 8)            0           conv1d_2[0][0]
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 8)            0           conv1d_2[0][0]
__________________________________________________________________________________________________
global_average_pooling1d_3 (Glo (None, 8)            0           conv1d_3[0][0]
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 8)            0           conv1d_3[0][0]
__________________________________________________________________________________________________
global_average_pooling1d_4 (Glo (None, 8)            0           conv1d_4[0][0]
__________________________________________________________________________________________________
global_max_pooling1d_4 (GlobalM (None, 8)            0           conv1d_4[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 80)           0           global_average_pooling1d[0][0]
                                                                 global_max_pooling1d[0][0]
                                                                 global_average_pooling1d_1[0][0]
                                                                 global_max_pooling1d_1[0][0]
                                                                 global_average_pooling1d_2[0][0]
                                                                 global_max_pooling1d_2[0][0]
                                                                 global_average_pooling1d_3[0][0]
                                                                 global_max_pooling1d_3[0][0]
                                                                 global_average_pooling1d_4[0][0]
                                                                 global_max_pooling1d_4[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 80)           0           concatenate[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 80)           0           flatten[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          20736       dropout[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 256)          0           dense[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 2)            514         dropout_1[0][0]
==================================================================================================
Total params: 7,923,498
Trainable params: 7,923,498
Non-trainable params: 0
```

```
Restoring model weights from the end of the best epoch.
Accuracy Score is:  0.978862819699852
Precision Score is : 0.9851718714895529
Recall Score is : 0.9703474219960169
F1 Score:  0.9777034559643255
AUC Score:  0.9784976033710613
```


## **textcnn3**

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 1024)]       0
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1024, 128)    7851520     input_1[0][0]
__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, 1024, 512)    131584      embedding[0][0]
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 1024, 512)    197120      embedding[0][0]
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 1024, 512)    262656      embedding[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 3072, 512)    0           conv1d[0][0]
                                                                 conv1d_1[0][0]
                                                                 conv1d_2[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 1572864)      0           concatenate[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 1572864)      0           flatten[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 2)            3145730     dropout[0][0]
==================================================================================================
Total params: 11,588,610
Trainable params: 11,588,610
Non-trainable params: 0
```

```
Restoring model weights from the end of the best epoch.
Accuracy Score is:  0.9844674556213018
Precision Score is : 0.9935260115606936
Recall Score is : 0.973052536231884
F1 Score:  0.9831827022079854
AUC Score:  0.9837528925216473
```

------

## **Demo**

Samples:

```
```

------
## **Star-History**

![star-history](https://api.star-history.com/svg?repos=jackaduma/AI-WAF&type=Date "star-history")

------

## **Reference**

------

## **Donation**
If this project help you reduce time to develop, you can give me a cup of coffee :) 

AliPay(支付宝)
<div align="center">
	<img src="./misc/ali_pay.png" alt="ali_pay" width="400" />
</div>

WechatPay(微信)
<div align="center">
    <img src="./misc/wechat_pay.png" alt="wechat_pay" width="400" />
</div>

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://paypal.me/jackaduma?locale.x=zh_XC)

------

## **License**

[MIT](LICENSE) © Kun
