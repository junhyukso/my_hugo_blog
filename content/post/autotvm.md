---
title: "[Review] Learning to Optimize Tensor Program"
date: 2021-03-23T00:43:37+09:00
description : TVM의 머신러닝 기반 오토튜너인 AutoTVM에 대해 알아봅시다.
cover : /img/post/autotvm/overview.png
tags:
- DeepLearning
- TVM
- Compiler
---
## Intro
![schedule](/img/post/autotvm/schedule.png)
- Kernel들은, 성능에 영향을 미치는 수많은 Design space들이 있을 수 있습니다.
    -  Loop Order
    -  Loop Unrolling
    -  Tiling Size
    -  Local word size
    - ...

- 이러한 모든 조합을 고려한 경우의 수는, 보통 Billon단위로, Auto Tuner제작시 이러한 Design Space를 모두 탐색하는 것은 거의 불가능합니다.  
- 그렇다고 해서, 커널의 성능을 저러한 변수들로 **수식적으로 모델링**하는 것도 너무 어렵습니다.

- ->따라서, TVM측에서는 이를 머신러닝에 기반한 성능예측기로, Search space를 획기적으로 좁히고자 하였습니다.

## System Overview
![overview](/img/post/autotvm/overview.png)
- Auto TVM 시스템의 간단한 작동개요는 다음과 같습니다.
    - AutoTVM시스템은 입력으로 Neural Net Graph를, 출력으로 Optimized Backend Kernel을 출력합니다.
    - 우선 Tuning Algorithm내부의 ML Model이, Scedule Space에서 유망해 보이는 몇가지 후보를 추론하여 제공합니다.
        - 여기서 Scedule이란 , Tiling Size나 Loop Unrolling Factor같은 Search space들을 말합니다.
    - TVM이 헤당 Schedule을 받아, 그에 맞는 Backend Kernel을 생성합니다.
    - 이를 실제 하드웨어에 구동시켜, 실제 Latency(Cost)를 측정합니다.
    - 측정된 실제 데이터로, ML Model을 train 시킵니다.
    - **이러한 루프를 계속 반복하여, Optimized된 Kernel을 찾게 됩니다.**

## Tuning Algorithm
![algorithm](/img/post/autotvm/algorithm.png)
- Tuning Algorithm에 대해 더 자세히 살펴보도록 하겠습니다.
- 우선 Massive Parallel Simulated Anealing Algorithm을 통해 ML모델(f')을 사용하여 유망한 후보 Q를 추리게 됩니다.
    - 위 알고리즘을 사용하는 이유는, ML Model을 사용하더라도 Design space가 너무 커 유망한 후보 Top K개를 추리기 어렵기 때문입니다.
- Q에서 Cost를 Minimize시키면서도, 조합의 Diversity를 최대로 하는 조합 S를 Epsilon Greedy Algorithm을 통해 추립니다.
- S에 약간의 랜덤성을 부여한 후, 이를 실제 기기(f)에서 측정하여, My Dataset에 데이터를 추가합니다.
- 얻어진 My Dataset으로, ML Model을 retrain시킵니다.
- 이를 Max N Trial까지 루프돌며 반복하게 됩니다.
- 알고리즙의 출력으로, 찾아진 S중 cost가 가장 낮았던 S'를 출력하게 됩니다.

## ML Model
- 결국 위 알고리즘에서 ML Model을 사용하는 이유는, 시간이 많이 소요되는 과정(Compile -> Evaluation from real HW)을 피하고자 함에 있습니다.
    - 따라서, ML Model의 retrain + inference시간이, 실제 기기에서의 테스팅 시간에 비해 매우 낮아야 의미있는 모델일 것입니다.
- AutoTVM의 저자들은, XGBoost기반 모델을 사용하였습니다.
    - XGBoost의 inference time은 약 0.67ms로, 실제 기기에서 커널 수행시간 보다 약 1000배 이상 빨랐다고 합니다.
    - 저자들은 DL기반의, TreeGRU모델또한 사용해 보았으나 Acc는 비슷한데에 비해 Latency만 커, XGBoost를 사용했다고 합니다.
- 학습시 Loss functing으로, Regression Loss가 아닌 Rank Loss를 사용했습니다.
    - ML Model의 역할은 유망한 후보 Q를 뽑는 목적이므로, 커널의 시간을 정확히 맞추긴 보단 상대적인 우위만 맞추면 됩니다.
![Rank Loss](/img/post/autotvm/rank_loss.png)

## Distributed RPC
![DRPC](/img/post/autotvm/dRPC.png)
- Tuning Algorithm에서 볼 수 있듯, 결국 실제 디바이스에서 커널시간을 측정하는 과정이 많이 필요합니다.
- 하지만, 이러한 과정은 매우 귀찮고 시간이 오래걸릴 수 있습니다.
- TVM에서는 RPC를 사용하여, Host에서 Kernel만 Server로 전송하게 되면, 서버가 기기와 통신해 컴파일된 코드를 보내고, 기기에서 수행시간을 측정해 Host로 보내주는 편리한 Interface를 구축해 이 문제를 해결했습니다.

## Performace
![result](/img/post/autotvm/result.png)
- (2019년 기준으로) Server, Mobile, GPU, CPU모두에 대해 TVM이 가장 우수한 성능을 보여주었습니다.


