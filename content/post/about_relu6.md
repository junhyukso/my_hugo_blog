---
title: ReLU6 알아보기
date: 2020-10-14
description : ReLU의 변형 중 하나인 ReLU6 에 대해서 알아보겠습니다.
cover : https://user-images.githubusercontent.com/41286195/95899022-ddc28600-0dca-11eb-8659-61f0fb31ab28.png
categories :
- a
tags:
- DeepLearning
- Activation
---
ReLU의 변형 중 하나인 ReLU6 에 대해서 알아보겠습니다.

## ReLU(Rectified Linear Unit)

![](https://user-images.githubusercontent.com/41286195/95898957-c6839880-0dca-11eb-8bda-ce4953537a38.png)  
ReLU는 딥러닝에서 가장 널리 사용되는 활성화 함수 중 하나입니다.

이러한 ReLU를 사용하는 가장 큰 이유는 두가지 입니다.

- 비선형성 : ReLU함수가 비선형 함수이기 때문에, 뉴럴넷에 비선형성을 줄 수 있습니다.
- Gradient Vanishing 방지 : 양수일때, 기울기가 1이므로 Gradient가 손실되지 않습니다.

## ReLU6

![](https://user-images.githubusercontent.com/41286195/95899022-ddc28600-0dca-11eb-8659-61f0fb31ab28.png)

ReLU6는 이러한 ReLU의 변형으로,  ReLU에 6이라는 상한을 걸어둔 것입니다.

MobileNet[1] 에서 주로 사용하는 Activation으로,

- Precision에서 장점을 가질 수 있습니다. 최댓값이 6이므로, 3bit만 사용하여 표현 가능합니다.[1]
- Hard한 Sigmoid로 생각할 수도 있습니다.
    - 실제로 MobileNet V3[3]에선 Hard-Swish 로 사용합니다.
- Sparse한 Feature를 일찍 학습할 수 있습니다. [2]

## 실제 사용

ReLU6는 실제 자주 사용되는 Activation으로, 주요 프레임워크들에 대부분 구현되어 있습니다.

- Tensorflow : [https://www.tensorflow.org/api_docs/python/tf/nn/relu6](https://www.tensorflow.org/api_docs/python/tf/nn/relu6)
- Pytorch : [https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html](https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html)

또한, 대부분의 MobileNet 구현에 사용되어 있습니다.

- MobileNet V1

([link](https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/applications/mobilenet.py#L82-L310))

```python
def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):
 ...

  x = layers.BatchNormalization(
      axis=channel_axis, name='conv_dw_%d_bn' % block_id)(
          x)
  x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

 ...

  return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
```

제가 MobileNetV1 구현을 하며 혼동을 느꼈던 부분으로, 논문에선 언급이 없지만 이때부터 ReLU6를 사용하였습니다.

- MobileNet V2 [1]

MobileNetV2 부터는 논문에서 ReLU6를 사용했음을 언급합니다.

([link](https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/applications/mobilenet_v2.py))

```python
...
x = layers.BatchNormalization(
   axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
x = layers.ReLU(6., name='Conv1_relu')(x)
... 
```

# References

[1]Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[2] Convolutional Deep Belief Networks on CIFAR-10: Krizhevsky et al., 2010

[3] Howard, Andrew, et al. "Searching for mobilenetv3." Proceedings of the IEEE International Conference on Computer Vision. 2019.