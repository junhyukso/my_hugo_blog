---
title: "[REVIEW]Lottery Ticket Hypothesis"
date: 2020-10-19
description : Lottery Ticket Hyphothesis에 대해 알아봅시다.
cover : https://user-images.githubusercontent.com/41286195/96431539-9dd22780-123e-11eb-8a54-e9bcc60f93d2.png
tags:
- Review
- DeepLearning
- EfficientAI
- Pruning
---

## Network Pruning
Pruning은 딥러닝 모델을 경량화 하기 위한 방법으로, **정확도 손실을 최소로** 하며 모델에서 어느 정도의 **파라미터들을 제거**하는 기법입니다.

### Iterative Pruning
![Iterative Pruning. Han et al 2015](https://user-images.githubusercontent.com/41286195/96431539-9dd22780-123e-11eb-8a54-e9bcc60f93d2.png)

가장 널리 사용되는 Pruning 방법인 Iterative Pruning은 위 그림과 같습니다.  
우선 어떠한 기준을 통해 **중요하지 않은** 파라미터를 판단하고, 제거합니다. 그리고 모델을 다시 **재학습**시킵니다.  
이러한 step들을 반복함으로써, 모델의 파라미터를 점점 더 제거합니다.  

### Problem
하지만, 이러한 방식을 통해 얻은 SubNetwork를, Randomly Initialize한후, **처음부터 학습**시키게 되면 본래의 성능을 달성할 수 없었습니다.  
Iterative Pruning자체가 꽤 **많은 HyperParamter**들이 있고, SubNetwork를 학습시킬수있다면 **Train FLOPs또한 큰 폭으로 줄일 수 있기에** 이 문제를 해결하는 것은 중요했습니다.  
  
The Lotter Ticket Hyphothesis[ICLR2019] 에서 저자들은 이러한 문제를 해결할 수 있는 방법을 제시합니다.

## The Lottery Ticket Hyphotesis
우선 Lottery Ticket이란 용어부터 정의합니다.
- Lottery Ticket : _Original Network_ 보다 **적은 Parameter**를 가지고, **성능또한 더 좋은** _SubNetwork_.
- 저자들은 이를 말그대로 **복권**에 비유해, Lottery Ticket이라는 용어를 사용했습니다.

논문에서 제시하는 이러한 **Lottery ticket**을 찾는 방법은 아래와 같습니다.
![Finding Lottery ticket](https://user-images.githubusercontent.com/41286195/96432957-0968c480-1240-11eb-8b5a-33fb3ec394cb.png)
- 1,2,3과정은 통상적인 Train -> Iterative Pruning 과정입니다.
- **4, 이제 Iterative Pruning으로 찾은 SubNetowrk를 1에서 사용했던 "초기값"으로 초기화 합니다.**
    - 이때, 초기값에 사용한 분포는 Xavier와 같은 일반적인 분포입니다.

이러한 과정을 통해 찾은 Lottery Ticket을, **처음부터 학습시킨 결과**는 아래와 같습니다.
## Experimental Results
![results on LeNet](https://user-images.githubusercontent.com/41286195/96435049-8eec7480-1240-11eb-913a-16a016071808.png)
십자가 기호 옆의 숫자는 기존 네트워크 대비 남아있는 파라미터의 비율입니다.  
- 제일 왼쪽 그림을 보게되면, 기존 네트워크(100)보다 Lottery ticket들의 학습결과가 월등함을 알 수 있습니다.
- 가운데 그림을 보게되면, 3.6%(보라색)까지도 기존 네트워크보다 학습결과가 좋지만, 1.9%(갈색)은 결과가 크게 나빠지는 것을 볼 수 있다.
 - 이러한 SubNetwork의 파라미터 수의 어떠한 하한이 있다는 것을 알 수 있습니다.
- 오른쪽 그림을 보게되면, Lottery ticket방법을 적용한 결과가, 적용하지 않은 결과(reinit)보다 월등히 좋음을 확인할 수 있습니다.

저자들은 Simple Convnet이나 Deep Convnet(VGG,ResNet..)에 대해서도 실험을 진행하였는데, 몇가지 휴리스틱이 들어가긴했지만 모두 좋은 결과를 보였습니다.

## Conclusion


## References
Frankle, Jonathan, and Michael Carbin. "The lottery ticket hypothesis: Finding sparse, trainable neural networks." arXiv preprint arXiv:1803.03635 (2018).



 

 