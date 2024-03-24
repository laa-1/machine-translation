# MachineTranslation 

本项目是一个基于 PyTorch 框架，使用 Transformers 模型应用于翻译任务的例子，并且附带了详细的文档介绍 Transformers 模型在训练和推理中数据是如何变化的，你可以在[这里](doc/README.md)找到它。

## 环境

* python 3.10
* torch 2.0.0

## 数据集

数据集采用的是 AI Challenger 的数据集，其中包含中英文个 1000 万条日常生活常用的短句。

下面是一些训练集的例子。

```text
A pair of red - crowned cranes have staked out their nesting territory
A pair of crows had come to nest on our roof as if they had come for Lhamo.
A couple of boys driving around in daddy's car.
A pair of nines? You pushed in with a pair of nines?
Fighting two against one is never ideal,
```

```text
一对丹顶鹤正监视着它们的筑巢领地
一对乌鸦飞到我们屋顶上的巢里，它们好像专门为拉木而来的。
一对乖乖仔开着老爸的车子。
一对九？一对九你就全下注了？
一对二总不是好事，
```

## 使用

* process_data.py 数据集预处理
* train.py 训练模型
* translate.py 使用模型进行英译中翻译

## 训练结果

由于硬件条件有限，只训练了1个epoch，但由于数据集很大，训练1个epoch就已经由一定的效果了，且loss还有下降的趋势。

下面是测试集的一些例子。

```text
Find a safety chain or something to keep these lights in place. 
So that no parent has to go through what I've known. 
I have to go to the date, learn to dance. Definitely. Now. 
Is when someone we've trusted makes the choice for us. 
Okay. Well, I guess there's not much to do about it right now then. 
I respect that, and I will protect it at all cost. 
```

```text
找到安全链或者保存这些灯。
所以我知道的不是父母。
我要去约会，学会跳舞。当然。现在。
是否有人信任我们。
好吧。那么，我想现在没什么可做的了。
我尊重这一点，我会保护它的。
```
