---
title: "Batch Normalization 정리"
date: 2020-12-26T16:55:39+09:00
description: BN레이어의 동작과, 몇가지 응용을 알아보겠습니다.
cover : /img/post/bn_v1/cover.png
tags :
- DeepLearning
- Layer
- EfficientAI
---

## Intro

Batch Normalization(이하 BN)은 딥러닝 모델을 학습시킬때 사용되는 레이어중 하나로,

2015년에 발표된 이후로 그 성능을 인정받아 현재까지도 매우 활발하게 사용되는 중입니다.

![엄청난 인용수](/img/post/bn_v1/god.png)
(엄청난 인용수...)  
  


BN은 딥러닝 모델을 훈련할시 수렴의 안정성과 속도 향상을 가져옵니다.  
이 포스팅에서는 BN레이어의 동작과, 여러가지 응용을 살펴보겠습니다. BN의 모티베이션이나 상세한 실험 결과들은 해당논문 ([https://arxiv.org/pdf/1502.03167.pdf](https://arxiv.org/pdf/1502.03167.pdf))을 참조해주세요.

## Batch Normalization

BN 레이어는 내부적으로 **네가지 Parameter**를 가집니다.

- Mean
- Variance
- Gamma
- Beta

![BN Algorithm](/img/post/bn_v1/algo1.png)

BN레이어의 동작자체는 위 그림과 같이 간단합니다.

- 레이어의 인풋을 Mean,Variance로 정규화 후
- gamma를 곱해준후, beta를 더해줍니다.

### Gamma, Beta

이때 Gamma와 Beta는 Trainable한 Parameter로, 딥러닝 모델의 학습시 **오차역전파**를 통해 적절한 값으로 학습됩니다.

Gamma는 1, Beta는 0으로 처음 초기화 되게 됩니다.

### Mean , Variance

그러나 Mean, Variance는 **오차역전파를 통해 학습되는 값이 아닙니다**.  또한 Training mode와 Inference mode시에 사용하는 방식이 다릅니다.

### Training Mode

Training mode에서는 레이어의 인풋으로 들어온 **Mini Batch의 Mean, Variance**를 사용하여 정규화를 진행합니다.

이때, Mean, Varaince를 한번 사용하고 버리는 것이 아니고

**Inference Mode에서의 사용을 위해 계속 지수이동평균으로 축적합니다.**

지수이동평균은 다음과 같은 식으로 계산됩니다.

이때 alpha는 **Momentum**이란 계수로 보통 0.9, 0.99, 0.999와 같이 1에 가까운 값을 사용합니다.

Tensorflow, Pytorch 구현에서도 기본값으로 0.99,0.9를 사용함을 확인할 수 있습니다.

```python
tf.keras.layers.BatchNormalization(
    axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones', beta_regularizer=None,
    gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
    renorm=False, renorm_clipping=None, renorm_momentum=0.99, fused=None,
    trainable=True, virtual_batch_size=None, adjustment=None, name=None, **kwargs
) #TF Momentum: 0.99

torch.nn.BatchNorm2d(
	num_features, 
	eps=1e-05, 
	momentum=0.1, 
	affine=True, 
	track_running_stats=True
) #Torch Momentum: 0.9(0.1)
```

### Inference Mode

Inference mode에서는 이러한 방식을 통해,

- 지수이동평균으로 계산된 Mean, Var
- 오차역전파를 통해 찾아진 Gamma , Beta

를 사용하여 BN계산을 진행합니다.

이미 **고정된 값만을 사용하여 진행하기 때문에, O(1)로 계산이 가능합니다.**

## Batch Normalization Folding

CNN모델에서 BN레이어는 보통,

- **채널 단위로 존재하고**
- Convolution → BN → ReLU의 순서로 구성됩니다.

따라서, Inference Mode라면 Convolution → BN 연산은

**y = Gamma * (( Wconv * x ) - Mean )/Var + Beta** 

입니다. 이는

**y = W*x + b**

와 같이 단순화 시킬 수 있습니다.

따라서 실제 추론기 구현시, Batch Normalization레이어를 구성하지 않더라도,**Convolution 레이어의 Weight ,bias를 조작하여 똑같은 계산 결과가 나오게 두 오퍼레이션을 합칠(Fusing)수 있습니다**. 

이러한 최적화 방식을 BN Folding(or Fusing)라고 부릅니다.

**Inference Mode**에서의 **BN은 O(1)**이므로, 상당히 **Memory Bound되는 연산**입니다.**BN Fusing**을 하게 되면, Conv이후 BN레이어를 수행하기 위해 **RAM에 중간 Activation을 저장해야 할 필요가 없습니다.** 

따라서 **램의 Bandwidth를 아껴 추론시 상당한 수행시간의 이득을 볼 수 있는 효과**가 있습니다.

실제 프레임워크에서들에서는 CBR을 Folding하는 최적화를 내부적으로 모두 진행하고 있습니다.

## Batch Renormalization

[TODO]

## References

- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- [https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)
- [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/](https://nenadmarkus.com/p/fusing-batchnorm-and-conv/)
