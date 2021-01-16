---
title: "[REVIEW] DRQ: Dynamic Region-based Quantization for Deep Neural Network Acceleration"
description : ISCA20에 발표된 DRQ논문의 리뷰입니다. DRQ는 Input Image에 따라 Dynamic 하게 Quantization을 적용합니다.
cover : /img/post/DRQ/cover.png
date: 2020-12-01T01:56:20+09:00
tags :
- Review
- DeepLearning
- EfficientAI
- Quantization
---
DRQ: Dynamic Region-based Quantization for
Deep Neural Network Acceleration

Zhuoran Song1, Bangqi Fu1, Feiyang Wu1, Zhaoming Jiang1, Li Jiang1, Naifeng Jing1*, Xiaoyao Liang1,2
1 Shanghai Jiao Tong University, China, 2 Biren Research, China

ISCA20에 발표된 DRQ논문의 리뷰입니다.

## 핵심
- Sentivite Region을 찾고, 그 부분만 High Precision(INT8) 으로 Quantization
    - Insensitive Region은 Low Preciison(INT4)
- Sensitivity Region은 Inference machine에서 Dynamic하게 Runtime에 찾음
    - Predefine 된 Threshold보다 낮은 부분을 Insensitive Region이라고 결정
- Threshold, Region Size는 Fine Tuning 하면서 결정
- 이러한 장점을 최대한 활용할수 있는 Hardware 또한 설계

## Sensitive Region 찾기
### Sensitive Values
![Sensitive values](/img/post/DRQ/sensitivity_values.png)
우선 Sensitive 한 Value라는게 정말 존재할까요? 저자들은 이름 검증하기 위해 아래와 같은 실험을 수행했습니다.  
Input Featrue Map을 세 그룹으로 나눕니다. 나누는 기준은 Input의 Magnitude를 기준으로,  
상위 20%, 20~80%, 80% 입니다.  
그리고 각각의 그룹에 노이즈를 적용합니다.  예를들어, 상위 20%에만 노이즈를 줬다면 TFF  
상위 20%,20~80%에만 줬다면 TTF입니다.  
**Observation from experiment**
- TFF가 FTF,FFT보다 Drop이 급격함.
- TFF ,TFT,TTF,TTT가 거의 일정. 즉, 상위20%그룹의 영향이 가장 큼.
- FFT는 큰 노이즈를 허용함.
- Imagenet이 Cifar10보다 좀 더 Noise에 민감.  

즉, Input의 **Magintude가 클수록 Sensitive** 함을 확인할 수 있습니다.

### Sensitive Region
![Sensitive Region](/img/post/DRQ/sensitivity_region.png)
그렇다면 이러한 Sensitive Value들의 분포는 어떻게 될까요?  
저자들은 이를 우선 Visualize 하기 위해, LeNet5에 MNIST데이터셋에 대해 실험을 수행했습니다.  
위 그림은 LeNet5에 3이란 이미지를 넣었을때, 처음 3 레이어를 나타낸 그림입니다.  

결과적으로, Magnitude가 큰 Group(Segment 0)은 무작위적으로 분포하지 않고, **집합하는 경향**이 있음을 알 수 있습니다. Segment 2는 중요하지 않은 부분에 넓게 분포함을 확인할 수 있습니다.  

그렇다면, 이러한 Input Feature Map의 h*w를 x*y개의 Patch로 나눠서 , Sensitive/ Insensitive한 Patch로 나눌 수 있을 것입니다. 이를 In/Sensitive Region 라고 부릅니다.  

## Sensitive Region 찾기 알고리즘
우선 DRQ Algorithm의 Overview를 잠깐 집고 넘어갑시다.  
### Algorithm Overview
![DRQ Overview](/img/post/DRQ/DRQ_algo_overview.png)

1. Sensitive Region Predictor  
  
Sensitive Region Predictor는 Runtime에, Input Feature Map의 각 Region들이 Sensitive한지 판단하는 역할을 합니다.

- h*w사이즈의 Input Feature Map을 받아서, 우선 FP32→INT8로 Quantize 합니다.
    - Int8이 기본 Precision 입니다.
- IFM을 x*y 개의 region으로 분할합니다.
- 각각의 Region을 Mean Filtering 합니다
- 이를 **Predifiend Threshold Activation에 통과 시킵니다.**
    - Threshold 를 Predefine하는 방법은 밑의 Design Space Exploration절에서 설명합니다.
- 이를 통해 각 Region에 대해 Binary Mask를 만듭니다. Binary Mask의 dimension은 (h*w)/(x*y) 입니다.

2. Mixed Precision Convolution
![Mixed Precision Convolution](/img/post/DRQ/mixed_precision_conv.png)
Mixed Precision Convolution은 Binary Mask에 따라, 각각 다른 precision으로 Convolution을 진행합니다.

- 우선 Kernel Weights들은 INT8로 Qunatize하여 DRAM에 저장합니다.
- Sensitive Region과 연산할때는, 일반 INT8 Convolution을 사용합니다.
- INSensitive Region으로 연산할때는 Input이 INT4로 DRAM에 저장됩니다. Convolution시는, Weights를 바로 INT4범위로 **Clipping** 해서 INT4 Conv를 수행합니다.

### Design Space Exploration

두가지 DSE요소가 있습니다.  
- 첫째는 Threshold 입니다.  
Threshold가 클수록 당연히 속도향상이 클것이지만(Insensitive Region이 많아지므로), 중요한 피쳐를 버릴 가능성이 높아지므로 ACC drop이 클것입니다.
  
- 두번째는 Region(x*y)의 크기 입니다.  
Region이 작을 수록 세세한 Input Region을 보는것이므로 정확도로의 영향이 작겠지만, 가속효과가 적을것입니다. 또한 Region의 크기는 HW Friendly해야합니다.  

- 또한, 이러한 Mixed Precision Convolution을 진행하게되면 Weight가 원래 사용하던 Input이 아니므로 Acc drop이 클 것입니다.  

저자들은 이를 해결하기위해 간단한 Finetuning 방법을 제시합니다.  

- Threshold와 Region Size를 우선 적당히 큰값으로 잡습니다.
- 이러한 Threshold와 Region Size를 적용한 Forward Pass를 Mixed Precision으로 수행합니다.
- 단, Backward Pass는 Full Precision으로 진행합니다.
- 이를 정확도가 수렴할때까지 진행합니다.
- 최종 목표 정확도가 나올때까지 Threshold와 Region Size를 더 줄여(1/2) 다시 진행합니다.
- 위의 과정은 몇 Iter 만에 끝나기때문에 빠르게 가능합니다.

실제 서로다른 네트워크들에 대해 찾아지는 Threshold와 Region Size의 예시는 아래 그림과 같습니다.
![real threshold and region size](/img/post/DRQ/result_th_reg.png)

## Architecture for DRQ

이부분은 일단 HW에 대한 지식이 부족하여 스킵합니다.

## Experimental Results

### Accuracy

![result acc](/img/post/DRQ/result_acc.png)

### Performance

![result time](/img/post/DRQ/result_time.png)

### Energy Consumption

![result Energy](/img/post/DRQ/result_power.png)

실험결과 가장 빠르고, 저전력이면서도 정확도가 기존 방법들에 비해 떨어지지 않음을 확인할 수 있습니다.