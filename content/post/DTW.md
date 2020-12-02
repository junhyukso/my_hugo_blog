---
title: "Dynamic Time Warping"
date: 2020-12-03T03:33:31+09:00
description: 패턴매칭에 자주 사용되는 알고리즘인 DTW에 대해 알아봅시다.
cover : /img/post/DTW/cover.png
- Algorithm
- Patter Recognition
---
Dynamic Time Warping
## Purpose
Dynamic Time Warping은 두 시퀸스의 유사성을 측정하기 위해 사용되는 알고리즘입니다.  
특히, 두 시퀸스가 길이가 다르거나 속도가 달라 동등한 비교가 어려울때 주로 사용합니다.  
예를들어, 전체적인 형태는 비슷하지만 길이,속도가 다른 두 시퀸스가 있습니다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = np.array([5,5,6,7,6,5,5,6,6,5,9,5], dtype=np.float32)
B = np.array([1,2,3,1,1,2,1,4,1], dtype=np.float32)
M = A.shape[0]
N = B.shape[0]

plt.plot(A)
plt.plot(B)
plt.legend(["A","B"])
plt.show()
```
{{< figure src="/img/post/DTW/fig1.png" position="center">}}

이러한 두 시퀸스 a,b 의 유사도를 측정하려면 어떻게 해야할까요?  
우선 단순히 b의 길이 만큼만 a를 뚝 잘라 비교하는 방법이 있을 것입니다.
```python
A_cut = A[:B.shape[0]]

plt.plot(A_cut)
plt.plot(B)
plt.legend(["A_cut","B"])
plt.show()

# 두 배열의 유사도(유클리드 거리 합)을 계산합니다.
def calc_similarity(a,b):
  return np.sum(np.abs(a-b))

similarity = calc_similarity(A_cut , B)
print("유사도 : ",similarity)
```
{{< figure src="/img/post/DTW/fig2.png" position="center">}}

혹은, b를 a의 길이만큼 단순히 **늘려서(Interpolation)** 비교할수도 있을것입니다.
```python
# arr_short 배열을 arr_long배열의 길이만큼으로 늘립니다(Linear Interpolation)
def interp_1d(arr_short,arr_long):
  return np.interp(
    np.arange(0,arr_long.shape[0]),
    np.linspace(0,arr_long.shape[0],num=arr_short.shape[0]),
    arr_short)

B_expand = interp_1d(B,A)

plt.plot(A)
plt.plot(B_expand)
plt.legend(["A","B_expand"])
plt.show()

similarity = calc_similarity(A , B_expand)
print("유사도 : ",similarity)
```
{{< figure src="/img/post/DTW/fig3.png" position="center">}}

## Definition
Dynamic Time Warping은 두 시퀸스(벡터)간에 Optimal Match를 찾는 알고리즘으로,  
여기서 Optimal 이란 두 배열 사이의 Cost(혹은 Similarity)가 최소임을 말합니다.   
두 배열의 Cost는 대응되는 인덱스간 배열값의 차의 절댓값의 합으로 정의됩니다.  
이때 인덱스 매칭은 다음과 같은 조건들을 만족해야 합니다.
- 두 배열의 시작, 끝 인덱스는 반드시 서로의 시작과 끝에 매칭되어야 합니다.
- 인덱스 대응은 일대다 대응이 가능합니다.
- 단, 양의 방향으로만 대응될 수 있습니다.

## Implementation
### Naive Approach(Brute Forcing)
이를 구현하는 가장 간단한 방법은, 두 벡터 A,B간의 서로 모든 값들을 비교한 행렬을 만들어,  
행렬의 시작(0,0)부터 끝(m,n)까지 이동하는 모든 Path를 Brute force 하며 값들을 비교하는 것입니다.  
각 path에 대해 Similarity(Cost)가 가장 작게 나오는 Path가 Optimal 한 Match일 것입니다.
```python
def make_all_possible_paths(M,N):
  #현재 좌표에서, 이동할수 있는 좌표들을 반환.
  def _get_possible_direction(i,j):
    ret = []
    if    i != M-1 and j != N-1 : #중간
      ret.append(   ( i   , j+1  ) )
      ret.append(   ( i+1 , j    ) )
      ret.append(   ( i+1 , j+1  ) )
    elif  i == M-1 and j != N-1 : #밑
      ret.append(   ( i   , j+1  ) )
    elif  i != M-1 and j == N-1 : #오른쪽
      ret.append(   ( i+1 , j    ) )
    return ret

  #이동할수 있는 경로들을 재귀적으로 따라감.
  def _recur(pos,previous_path=[]):
   i,j = pos
   current_path = previous_path[:] #call-by-ref이므로 복사
   current_path.append((i,j)) #현재 지점 기록
   dirs = _get_possible_direction(i,j)
   if len(dirs)==0 : #바닥 상태
     _all_paths.append(current_path) #최종 경로배열에 저장 
   else:
     for dir in dirs:
       _recur( dir, previous_path = current_path)
  
  _all_paths = []#최종 경로목록 배열 
  _recur( (0,0) ,previous_path=[])

  return _all_paths

def make_distance_naive(A,B):
  M = A.shape[0]
  N = B.shape[0]
  D = np.zeros(shape=(M,N))
  for m in range(M):
    for n in range(N):
      D[m,n] = np.abs(A[m] - B[n])
  return D

def calc_cost_with_path(D,path):
  cost = 0
  for (i,j) in path:
    cost += D[i,j]
  return cost

def find_optimal_cost(D,all_paths):
  all_cost = [ calc_cost_with_path(D,path) for path in all_paths ]
  return min((v,i) for i,v in enumerate(all_cost))

D                 = make_distance_naive(A,B)
all_paths         = make_all_possible_paths(M,N)
optimal_cost,idx  = find_optimal_cost(D,all_paths)
print("Distance Matrix    : \n" , D)
print("Number of paths    : "   , len(all_paths))
print("Optimal Match Cost : "   , optimal_cost)
print("Optimal Path : "         ,all_paths[idx][:9])
print("               "         ,all_paths[idx][9:])
```
```
Distance Matrix    : 
 [[4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [5. 4. 3. 5. 5. 4. 5. 2. 5.]
 [6. 5. 4. 6. 6. 5. 6. 3. 6.]
 [5. 4. 3. 5. 5. 4. 5. 2. 5.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [5. 4. 3. 5. 5. 4. 5. 2. 5.]
 [5. 4. 3. 5. 5. 4. 5. 2. 5.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [8. 7. 6. 8. 8. 7. 8. 5. 8.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]]
Number of paths    :  2485825
Optimal Match Cost :  42.0
Optimal Path :  [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 7)]
                [(9, 7), (10, 7), (11, 8)]
```
Brute Force로 찾아낸 Optimal Cost가 제일 작은값을 가짐을 확인할 수 있습니다.  
그러나, 이러한 방식은 M+N개의 이동경로중 N개를 선택하는 경우의 수를 가지므로
$$
O(\frac{(M+N)!}{M!N!})
$$
의 시간복잡도를 가집니다. 사실상 배열의 크기가 커지면 계산이 불가능 해지는 단점이 있습니다.
## DP approach
Dynamic Programming을 사용하여 이러한 문제를 해결할 수 있습니다.  
Distance Matrix에서 중요한 점은, 경로를 탐색시 오른쪽이나 밑으로만 내려갈 수 있다는 것입니다.  
즉, 경로의 한점은 그 왼쪽이나 위 점에 의존성을 가지므로, 왼쪽이나 위의 계산 결과를 굳이 다시 계산할 필요없이 유지시킬 수 있습니다.  
또한, 우리는 최소비용경로에만 관심이 있으므로, 왼쪽이나 위의 계산결과에서 최소값만 유지시키면 됩니다.  
이를 바탕으로 DTW를 위한 Distance Matrix를 다시 구현하면 다음과 같습니다
```python
def make_DTW_distance(A,B):
  M = A.shape[0]
  N = B.shape[0]

  def _cost(a,b):
    return np.abs(a-b)

  D = np.full(shape=(M+1,N+1) , fill_value=np.inf) # Inf 로 초기화
  D[0,0] = _cost(A[0],B[0])
  for i in range(1,M):#first col
    D[i,0] = D[i-1,0] + _cost(A[i],B[0])
  for j in range(1,N):#first row
    D[0,j] = D[0,j-1] + _cost(A[0],B[j])
  for i in range(1, M):#others
    for j in range(1,N):
      D[i, j] = min( D[i - 1, j - 1], D[i, j-1], D[i-1, j]) + _cost(A[i], B[j])

  return D

def find_optimal_path_DTW(D):
  M, N = D.shape
  m, n = 0, 0
  path = []
  while (m, n) != (M-2, N-2):
    path.append((m, n))
    m, n = min((m + 1, n), (m, n + 1), (m + 1, n + 1), key = lambda x: D[x[0], x[1]])
  path.append((M-2,N-2))
  return D[-2, -2], path


DTW_D = make_DTW_distance(A,B)
cost,path = find_optimal_path_DTW(DTW_D)
print("Distance Matrix    : \n" , D)
print("DTW Matrix         : \n" , DTW_D)
print("Optimal Cost : ",cost)
print("Optimal Path :\n",path[:9])
print(path[9:])
```
```
Distance Matrix    : 
 [[4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [5. 4. 3. 5. 5. 4. 5. 2. 5.]
 [6. 5. 4. 6. 6. 5. 6. 3. 6.]
 [5. 4. 3. 5. 5. 4. 5. 2. 5.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [5. 4. 3. 5. 5. 4. 5. 2. 5.]
 [5. 4. 3. 5. 5. 4. 5. 2. 5.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]
 [8. 7. 6. 8. 8. 7. 8. 5. 8.]
 [4. 3. 2. 4. 4. 3. 4. 1. 4.]]
DTW Matrix         : 
 [[ 4.  7.  9. 13. 17. 20. 24. 25. 29. inf]
 [ 8.  7.  9. 13. 17. 20. 24. 25. 29. inf]
 [13. 11. 10. 14. 18. 21. 25. 26. 30. inf]
 [19. 16. 14. 16. 20. 23. 27. 28. 32. inf]
 [24. 20. 17. 19. 21. 24. 28. 29. 33. inf]
 [28. 23. 19. 21. 23. 24. 28. 29. 33. inf]
 [32. 26. 21. 23. 25. 26. 28. 29. 33. inf]
 [37. 30. 24. 26. 28. 29. 31. 30. 34. inf]
 [42. 34. 27. 29. 31. 32. 34. 32. 35. inf]
 [46. 37. 29. 31. 33. 34. 36. 33. 36. inf]
 [54. 44. 35. 37. 39. 40. 42. 38. 41. inf]
 [58. 47. 37. 39. 41. 42. 44. 39. 42. inf]
 [inf inf inf inf inf inf inf inf inf inf]]
Optimal Cost :  42.0
Optimal Path :
 [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (3, 2), (3, 3), (4, 3), (5, 3)]
[(6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 7), (8, 7), (9, 7), (9, 8), (10, 8), (11, 8)]
```
Brute Force로 찾은 Optimal Cost와 동일함을 확인할 수 있습니다.  
DP로 찾은 경로와, Brute Force로 찾은 경로가 다른 이유는 같은 Optimal Cost를 가지는 여러 경로가 있을수 있기 때문입니다.

이러한 Dynamic Programming을 사용한 방식은, DTW Matrix를 한번 구성해주면 끝나게되고  
DTW Matrix는 코드에서 확인할 수 있듯 전체행렬의 원소를 한번씩만 돌며 O(1)의 연산을 진행하기 때문에
$$O(M*N)$$
의 시간복잡도를 가집니다.

## 시간복잡도 비교

앞에서 언급했듯,  Brute force 방식은  
$$O(\frac{(M+N)!}{M!N!})$$
의 시간복잡도를,  
DP 방식은
$$O(M*N)$$
의 시간복잡도를 가집니다.  
  
  
비교의 편의를 위해, M==N이라고 한다면,
Brute force , DP의 시간복잡도는 각각 
$$O(\frac{(2N)!}{N!N!}) ,   O(N^2)$$
$$$$
의 시간복잡도를 가집니다.  
실제 코드를 수행해, 수행시간을 그래프로 나타내면 아래와 같습니다.
```python
import time
def do_brute_force(N):
  A = np.random.rand(N)
  B = np.random.rand(N)
  t1 = time.time()
  D                 = make_distance_naive(A,B)
  all_paths         = make_all_possible_paths(N,N)
  optimal_cost,idx  = find_optimal_cost(D,all_paths)
  t2 = time.time()
  return t2-t1

def do_dp(N):
  A = np.random.rand(N)
  B = np.random.rand(N)
  t1 = time.time()
  DTW_D             = make_DTW_distance(A,B)
  all_paths         = find_optimal_path_DTW(DTW_D)
  t2 = time.time()
  return t2-t1

bf_time = []
dp_time = []
for N in range(2,11):
  bf_time.append(do_brute_force(N))
  dp_time.append(do_dp(N))

plt.plot(range(2,11), bf_time)
plt.plot(range(2,11), dp_time)
plt.legend(["Brute Force","Dynamic Programming"])
plt.xlabel("N")
plt.ylabel("Time(s)")
plt.show()
```
{{< figure src="/img/post/DTW/fig4.png" position="center">}}

N이 10이상이면 Brute Force의 수행시간이 너무 커져 실험하지 않았습니다.
그래프에서 확인할수 있듯, BF방식이 DP보다 수행시간이 기하급수적으로 큼을 알 수 있습니다.

감사합니다.

## Reference
https://en.wikipedia.org/wiki/Dynamic_time_warping

