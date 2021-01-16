---
title: "[REVIEW] DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference"
date: 2020-12-07T17:44:32+09:00
description : Bert Classifier 에 Early Exit을 적용한 DeeBERT에 대해 알아봅니다.
cover : /img/post/deebert/fig1.png
tags :
- Review
- DeepLearning
- EfficientAI
- NLP
---

## Intro
- Bert는 **무겁습니다.**
- 따라서 **Computation**을 줄이려는 많은 노력들이 있습니다.
    - Transformer의 계산복잡도 자체를 줄이려는 방법.
    - **Pruning**을 하여 Sparse 하게 만드는 방법. (Sparse Transformer)
    - **Knowledge Distillation**을 하여 모델 사이즈 자체를 줄이는 방법.(DistillBERT)
    - 등등..
    - 이 논문은 **Early Exit**을 사용하였습니다.

## DeeBERT Inference
### Early Exit
{{< figure src="/img/post/deebert/fig1.png" position="center">}}
Early Exit은 위 그림과 같이, 레이어 사이에 **보조 분류기**를 삽입하여,  
해당 **보조분류기**가 결과에 대해 **확신** 한다면, 추가적인 Forward를 중지하여, Computation을 줄이는 방법입니다.  
Image Classification 분야에서는, 이미 이를 적용한 여러 논문들이 있습니다.(BranchyNet, BPNet, ...)  
DeeBERT에서는 Transformer 레이어들 사이에 **보조분류기**를 삽입하였습니다.

### Confidence?
결과의 확신이라는 수치를 어떻게 나타낼수 있을까요?  
이 논문에서는 **(Information) Entropy**를 사용합니다. (BranchyNet방법)
$$H(x)=\sum_{x}P(x)ln(1/P(x))$$
P(x)가 각 클래스의 출력확률, 즉 Softmax레이어의 아웃풋입니다.

### Threshold
{{< figure src="/img/post/deebert/fig2.png" position="center">}}
그리고 이러한 **Entropy**가 , 미리 설정된 **Threshold S**보다 작다면, 그 지점에서 **Inference 를 Exit** 시킵니다.  

이러한 **Threshold S**가, **Latency-Accuracy Trade-off**에 **가장 중요한 수치**임은 자명합니다.  
왜나면 **S가 크다면**, 보조분류기가 강하게 확신해야만 추론을 중지하므로 **Acc는 높겠지만 Latency이득이 크지 않을것**이고,  
**S가 작다면**, 보조분류기가 작은 확신에도 추론을 중지할수가 있으므로 **Acc는 낮겠지만 Latency이득은 상대적으로 클것**입니다.  

## Result
{{< figure src="/img/post/deebert/fig3.png" position="center">}}
**DistillBERT**에 비해, **Accuracy-Time Tradeoff 가 더 Optimal**함을 확인할 수 있습니다.

결론적으로, BERT에도 충분히 EarlyExit 이 적용될 수 있다는 가능성을 보여준 Contribution입니다.

감사합니다.

## Reference
https://arxiv.org/abs/2004.12993