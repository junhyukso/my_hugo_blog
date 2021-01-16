---
title: "[REVIEW] RepVGG:Making VGG-style ConvNets Great Again"
date: 2021-01-15T21:09:25+09:00
description : "Make VGG Great Again!"
cover : /img/post/RepVGG/fig2.png
---

Make VGG Great Again!

## Intro

![fig1](/img/post/RepVGG/fig1.png)

CNN의 모델은 성능향상을 위해 **구조가 점점 복잡**해지고 있습니다.  **Branch가 중간에 존재**하기도 하고,(ResNet) **1x1 Conv와 Depthwise Convolution**(MobileNet)을 사용하기도 하고, 비선형성을 위해 **H-Swish**와 같은 activation을 사용하기도 합니다.

RepVGG는 이러한 기법들 없이(정확히는 Inference 시만), **오직 3x3Conv와 ReLU만을 사용**하여 Imagenet Top1 Acc 80%라는 인상적인 결과를 보여주었습니다.

## Training

![fig2](/img/post/RepVGG/fig2.png)

VGG와 같은 단순한 ConvNet구조는 **Training시 Gradient Vanishing이라던가 하는 문제로 성능이 좋지 않다는 문제**가 있습니다. 따라서 ResNet처럼 Residual Branch를 만들어, Gradient를 더 잘 흐르게 하여 성능향상을 이뤄내는 경우가 많아졌습니다.

RepVGG는 여기서 영감을 받아, **Training** 시에는 1x1Conv와 Identitiy로 이루어진 **Residual branch**를 사용하고, **Inference** 시에는 이를 reparameterization기법으로 **하나의 3x3Conv**로 합치는 기법을 제시하였습니다.

## Reparameterization

![fig3](/img/post/RepVGG/fig3.png)

RepVGG에서는 각 3x3Conv옆에, 1x1Conv와 Identity branch를 추가했습니다.

이들은 ReLU를 통과하지 않고 합쳐지므로(중간에 비선형성이 없으므로), **하나의 Conv로 합칠수 있습니다.**

과정은 아래와 같습니다.

- 우선 **Identity branch**는 Identity Matrix를 Kernel로 가지는 1x1 Conv와 동치입니다.
- 또한, **1x1Conv**는 상하좌우 패딩이 1씩 들어간 3x3Conv와 동치입니다.
- 또한, **Conv-BN**은 BN Folding을 통해 하나의 Conv, bias로 생각할수 있습니다.
- 이제 각각 **세개의 3x3Conv를 더해서, 하나의 3x3 Conv**로 만듭니다.

## Winograd Convolution

사실 **Branch**가 없어지는 것은 **Memory적 이득** 요소가 크고, **Addition**은 실제 **수행시간의 영향이 크지 않으**므로 VGG형태의 CNN을 사용한다해서 **속도의 차이는 크게 없습**니다.

하지만 **3x3Conv만 사용하여 이득**을 얻을 수 있는 부분이 있는데요, 바로 **Winograd Convolution**입니다.

**Winograd Conv**는 Strassen 행렬곱의 Conv버전이라고도 생각할 수 있는데요, 핵심은 **곱하기 연산의 수를 줄이고 더하기 연산의 수를 늘리는 것**입니다.

보통 하드웨어적으로, **곱셈기의 코스트**가 **덧셈기보다 훨씬 크기**때문에 이러한 개선은 **수행시간적으로 의미있는 향상**이 있습니다.

3x3Conv의 경우엔 **곱셈연산의 수가 4/9**로 줄어듭니다.

자세한 설명은 해당 논문([Lavin et al.](https://arxiv.org/pdf/1509.09308.pdf))를 참조해주세요.

## Experimental Result

![fig4](/img/post/RepVGG/fig4.png)

 **Reparameterization기법**을 통해,**VGG Style CNN을 유지**하면서  SOTA는 아니지만 **정확도와 속도향상을 모두 이뤄냈다는 점**에서 흥미로운 논문이였던 것 같습니다.

감사합니다.

## Reference

[https://arxiv.org/pdf/2101.03697v1.pdf](https://arxiv.org/pdf/2101.03697v1.pdf)

[https://arxiv.org/pdf/1509.09308.pdf](https://arxiv.org/pdf/1509.09308.pdf)