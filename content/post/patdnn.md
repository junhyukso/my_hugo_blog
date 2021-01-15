---
title: "[REVIEW]PatDNN: Achieving Real-Time DNN Execution on Mobile Devices with Pattern-based Weight Pruning"
date: 2021-01-02T21:40:20+09:00
cover : /img/post/patdnn/conver.png
description : Pattern based Pruning을 적용한 가속 Framework인 PatDNN에 대해 알아봅시다.
---


## Pruning
![Pruning](/img/post/patdnn/fig2.png)
Pruning은 딥러닝 모델에서 **특정 Weight들을 0으로만들거나, 아예 제거시키는 기법**입니다.    
이렇게 하게되면 **정확도를 조금만 떨어트리거나 혹은 심지어 향상시키면서**, **모델의 사이즈나 추론속도를 빠르게 만들 수 있습니다.**
### Unstructured Pruning
Unstructured Pruning이란 Fig 2.(a)와 같이 **element-wise**하게 Pruning 하는 방법입니다.  
이러면 **낮은 Accuracy Drop**과 **상대적으로 높은 Pruning Rate(Sparsity)**는 얻을 수 있으나, 매우 높은 Sparsity가 아닌 이상 **추론속도를 향상시키기가 실질적으로 어렵**습니다.  
왜냐하면 딥러닝 연산은 대부분 **행렬곱** 연산으로 수행되는데, **Sparse 행렬곱**의 경우는 **Sparsity가 약 8~90%**정도는 되어야만 **가속 효과**가 나타나기 때문입니다. **낮은 Sparsity**에서는 Sparse행렬곱의 변환오버헤드 때문에 **오히려 느립니다.**
### Structured Pruning
Structured Pruning은 Fig2.(b)와 같이, 일정 Group을 지정하여 **Group을 통째로 Prune**하는 방법입니다.  
이때 Group은 **HW아키텍쳐의 레인**에 맞춘 사이즈라던가, 심지어 **Conv의 필터/채널 전체**를 제거할수도 있습니다.  
특히 **CNN**의 경우는 필터/채널을 제거하게 되면 **추론속도의 향상을 바로 담보할 수 있으므로**, 필터/채널 자체를 제거하는 방법을 주로 사용합니다.  
그러나, 이러한 방법은 **Accuracy Drop이 크다**는 단점이 있습니다.

## Pattern based Pruning
![Pruning performace](/img/post/patdnn/table2.png)
정리하면, **Unstructured Pruning** 는 **Acc Drop이 낮지만 가속을 기대하기가 어렵고**,  
**Structured Pruning**은 **Acc Drop이 크지만 높은 가속**을 기대할 수 있습니다.  
**Pattern based Pruning**은 이러한 두 Pruning기법의 장점을 모두 챙기는, **중간지점**으로 제시되었습니다.
### Kernel Pattern Pruning
![Pattern Pruning](/img/post/patdnn/fig3.png)
다음과 같은 Step들로 수행됩니다.
- 모델을 **우선 Pretrain**합니다.
- Conv의 **한 Kernel**에서 **고정된 갯수의 수의 Weight**만 남기고 Pruning 합니다.
    - 논문에서는 **4개의 Weight를 남길때 가장 성능이 좋았다고 합니다.
- **3x3Conv**고, 한 커널당 **4개**의 weight를 살렸다 가정하면 **8C3개**의 경우의 수가 있습니다.
    - 커널의 **가운데는 중요한 정보**를 담고 있기때문에 Pruning 되어선 안됩니다. 따라서 9C4가 아닌 8C3입니다.
- 모델의 전체 Kernel들을 탐색하며, **가장 많이 등장하는 Top-K개의 커널을 조사**합니다.
- 가장 많이 등장한 **K개의 커널만 살리고**, 나머지는 **제거**(Connectivity Pruning,후술)합니다.
    - 논문에서는 8개 Kernel을 살릴때 성능,속도가 최적이라고 합니다.
- 떨어진 정확도를 복구하기 위해 fine tuning을 진행합니다.

### Connectivity Pruning
![Connection Pruning](/img/post/patdnn/fig4.png)
Connectivity Pruning은 Convolution에서 **Input Channel과 Output Channel의 연결을 끊는것**, 즉 **매칭되는 Filter를 제거하는 것**입니다.  
**Kernel Pattern Pruning에서 살리는 Kernel 이외는 제거하여 Connection을 끊습니다.**

### Expreiment Result
![Accuracy Comparison](/img/post/patdnn/table3.png)
이러한 Pruning 기법을 **VGG16, ResNet50**을 적용한결과 **정확도가 오히려 상승**하는 결과를 보였습니다.  
또한, Pattern에 맞춰 **최적화된 Convolution Code**를 작성할 수도 있습니다.  
**Inference code Optimization**과 다른 여러가지 가속 기법(Filter kernel reordering, Load redundancy elimination, GA based parameter auto tuning)등을 적용하여, TFLite나 MNN에 비해 **큰 가속**을 보였습니다.
![Performance Comparison](/img/post/patdnn/fig12.png)


## Reference
https://arxiv.org/pdf/2001.00138.pdf
