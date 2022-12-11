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
  - [**trained model files**](#trained-model-files)
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

## **trained model files**

a trained model which use private dataset, stored in **./cache** dir

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
