---
title: "Numpy로 1D 배열 Interpolation 하기"
date: 2020-12-02T04:00:14+09:00
description: Numpy를 이용하여 1D 배열을 Interpolation 하는 방법을 알아봅니다.
cover : /img/post/numpy_1d_interpolation/fig2.png
tags :
- Programming
- Numpy
---
```python
a = np.array([1,1,1,2,2,2,-1,-1,3,3,3,3], dtype=np.float32)
b = np.array([1,1,2,2,-1,3,3,3], dtype=np.float32)

plt.plot(a)
plt.plot(b)
plt.legend(["a","b"])
plt.show()
```

![Fig. 1.](/img/post/numpy_1d_interpolation/fig1.png)

이러한 길이가 다른 두 1D 배열이 있을때, 길이를 맞춰줄 일이 생겼습니다.

보통 Dynamic Time Warping이라던가 하는 방법을 많이 쓰지만, 이번엔 단순히 Interpolation으로 길이만 맞출일이 필요해서 코드를 작성해 보았습니다.

Numpy의 np.interp() 함수를 사용합니다.

[공식문서](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)

```python
np.interp(x , xp, fp)
```

- x 는 interpolation(Linear)을 진행할 x범위를 말합니다.
    - 위 상황에서는 0,1,2, ...,10,11 이겠네요.
- xp는 원래 데이터가 존재하는 X 배열을 말합니다.
    - 위 상황에서는 0~11범위에서 8구간으로 쪼갠 범위겠네요.
- fp 는 원래 데이터가 존재하는 Y배열을 말합니다.
    - 원래 데이터(b) 이겠네요.

따라서, 다음과 같은 함수를 작성할 수 있습니다.

```python
# arr_short 배열을 arr_long배열의 길이만큼으로 늘립니다(Interpolation)
def interp_1d(arr_short,arr_long):
  return np.interp(
    np.arange(0,arr_long.shape[0]),
    np.linspace(0,arr_long.shape[0],num=arr_short.shape[0]),
    arr_short)
```

이 함수를 사용하여 Interpolation 한 결과는 다음과 같습니다.

![Fig. 2.](/img/post/numpy_1d_interpolation/fig2.png)

감사합니다.
