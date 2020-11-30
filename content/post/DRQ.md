---
title: "[REVIEW] DRQ: Dynamic Region-based Quantization for Deep Neural Network Acceleration"
description : ISCA20에 발표된 DRQ논문의 리뷰입니다. DRQ는 Input Image에 따라 Dynamic 하게 Quantization을 적용합니다.
cover : /img/post/DRQ/cover.png
date: 2020-12-01T01:56:20+09:00
tags :
- Review
- DeepLearning
- Efficient DL
- Quantization
---

[DRAFT]

DRQ: Dynamic Region-based Quantization for
Deep Neural Network Acceleration

Zhuoran Song1, Bangqi Fu1, Feiyang Wu1, Zhaoming Jiang1, Li Jiang1, Naifeng Jing1*, Xiaoyao Liang1,2
1 Shanghai Jiao Tong University, China, 2 Biren Research, China

## 핵심

- Sentivity Region을 찾고, 그 부분만 High Precision 으로 Quantization
- Unsensitivity Region 은 Low Precision으로 Quantization.
- Sensitivity Region은 Inference machine에서 Runtime에 찾음 - "Dynamic!"
- 이에 따른 가속기 설계

## Sensitivity Region 찾기

Input Feature Map에서

- 어떤 픽셀들이 NN Acc와 직결되는가?
- 어떻게 찾는가? 분포는 어떻게 되는가?

### Sensitivity Values

![Sensitive values](/img/post/DRQ/sensitivity_values.png)

우선 Input Featrue Map을 세 그룹으로 나눕니다. 나누는 기준은 Input의 Magnitude를 기준으로,

상위 20%, 20~80%, 80% 입니다. 

그리고 각각의 그룹에 노이즈를 적용합니다.  예를들어, 상위 20%에만 노이즈를 줬다면 TFF

상위 20%,20~80%에만 줬다면 TTF입니다.

**Observation from experiment**

- TFF가 FTF,FFT보다 Drop이 급격함.
- TFF ,TFT,TTF,TTT가 거의 일정. 즉, 상위20%그룹의 영향이 가장 큼.
- FFT는 큰 노이즈를 허용함.
- Imagenet이 Cifar10보다 좀 더 Noise에 민감.

즉, 어떤 Input들이 다른 Impact를 가짐을 알수있습니다.

### Sensitive Region

![Sensitive Region](/img/post/DRQ/sensitivity_region.png)

우선 저자들은 Visualize 쉬운 구조를 위해, LeNet5에 MNIST데이터셋에 대해 실험을 수행했습니다.

Fig. 3은 LeNet5에 3이란 이미지를 넣었을때, 처음 3 레이어를 나타낸 그림입니다.

결과적으로, Magnitude가 큰 Group - Segment 0은 무작위적으로 분포하지 않고, 집합하는 경향이 있음을 알 수 있습니다. Segment 2는 중요하지 않은 부분에 넓게 분포함을 확인할 수 있습니다.

그렇다면, IPF의 2차원평면을 x*y의 Patch로 나눠서 , Sensitive/ Insensitive한 Patch를 구분합니다.

## Sensitivity Region 찾기 알고리즘

- Runtime에 어떻게 Sensitive Region을 구분할 것인가? 그것도 빠르고 HW Friendly하게?
Weight는 Offline으로 되지만 Input은 Online으로 되어야 한다.
- 어떻게 Efficient Input Sparsity Aware Convolution을 할것인가? Insensitive Region의 위치는 계속 달라질 수 있다...

**Algorithm Overview**

![DRQ Overview](/img/post/DRQ/DRQ_algo_overview.png)

1. Sensitive Region Predictor

Mean Filtering 후 Step Activiation Function 통과 시켜 Binary Mask 생성 → Section III-B for detail.

- h*w사이즈의 IFM을 받아서, 우선 FP32→INT8로 Quantize 합니다.
- IFM을 여러개의 x*y 개의 region으로 분할합니다.
- Mean Filtering 합니다
- 이를 **Predifiend Threshold Activation에 통과 시킵니다.**
- 이를 통해 Binary Mask를 만듭니다. Binary Mask의 dimension은 (h*w)/(x*y) 입니다.
- 

2. Mixed Precision Convolution

![Mixed Precision Convolution](/img/post/DRQ/mixed_precision_conv.png)

Sensitive/Insensitive Region에 대해 다른 Precision(INT8/INT4)로 Conv를 수행합니다. → Senction III-C

- 우선 Kernel Weights들은 INT8로 Qunatize하여 DRAM에 저장합니다.
- Sensitive Region과 연산할때는, 일반 INT8 Convolution을 사용합니다.
- INSensitive Region으로 연산할때는 Input이 INT4로 DRAM에 저장됩니다. Convolution시는, Weights를 바로 INT4범위로 Clipping 해서 INT4 Conv를 수행합니다.
- 더 복잡한 구현설명은 Section IV에 설명됩니다.

**Design Space Exploration**

두가지 DSE요소가 있습니다.

첫째는 Threshold 입니다.

Threshold가 클수록 당연히 속도향상이 클것이지만(Insensitive Region이 많아지므로), ACC drop이 클것입니다.

두번째는 Region(x*y)의 크기 입니다.

Region이 작을 수록 세세한 Input을 보는것이므로 정확도에의 영향이 작겠지만, 속도에는 악 영향일것입니다. 또한 Region의 크기는 HW Friendly해야합니다.

해결방법은 Retrain-Finetune 입니다.

- 우선 IFM의 Distribution을 뽑습니다.
- Threshold와 Region Size를 적당히 큰값으로 작습니다.
- 이러한 Threshold와 Region Size를 적용한 Forward Pass를 Mixed Precision으로 수행합니다.
- Backward Pass는 Full Precision으로 진행합니다.
- 이를 정확도가 수렴할때까지 진행합니다.
- 최종 정확도가 목표한 정확도면 중지하고, 그렇지 않으면 Threshol와 Region Size를 더 줄여(1/2) 다시 진행합니다.
- 위의 과정은 몇 Iter 만에 끝나기때문에 빠르게 가능합니다.


![real threshold and region size](/img/post/DRQ/result_th_reg.png)

## Architecture for DRQ

이부분은 일단 HW에 대한 지식이 부족하여 스킵합니다.

## Experimental Results

Accuracy

![result acc](/img/post/DRQ/result_acc.png)

Performance

![result time](/img/post/DRQ/result_time.png)

Energy Consumption

![result Energy](/img/post/DRQ/result_power.png)

가장 빠르고, 저전력이면서 정확도도 떨어지지 않음!