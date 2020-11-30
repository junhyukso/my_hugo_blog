---
title: "Ubuntu에서 Tor 사용하기"
date: 2020-11-28T03:43:18+09:00
description : Tor를 Ubuntu환경에 설치, 사용하는 법을 알아보겠습니다.
cover : https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Tor-logo-2011-flat.svg/320px-Tor-logo-2011-flat.svg.png 
tags :
- Linux
---

Tor는 네트워크 **우회나 익명성보장**을 위해 사용하는 툴 중 하나로, 여러 노드들을 경유하며 모든 노드간 암호화를 수행하기 때문에, 가장 안전한 네트워크 익명화 툴 중 하나입니다. Tor browser을 Linux(Ubuntu 18.04)에 **설치하는 방법**과, **기본적인 사용법**을 알아보도록 하겠습니다.

[https://www.torproject.org](https://www.torproject.org/download/)

## Tor Browser Repo 추가

- Tor browser를 Ubuntu Pacakge Manager로 설치하기 위해 필요합니다.

```jsx
sudo add-apt-repository ppa:micahflee/ppa
```

## Tor Browser 설치

```
sudo apt update 
sudo apt install torbrowser-launcher
```

## Tor Browser 실행

```jsx
sudo service tor start
```

위 명령을 실행하게되면 **기본적**으로

[**localhost:9050](http://localhost:9050)** 에서

**SOCK5로 Tor 프록시 서버**가 열리게 됩니다.

## Tor Browser 정상작동 테스트

myip.com에 **접속해 정상적으로 Tor가 작동하는지 확인**해 봅시다.

```jsx
curl -s --socks5-hostname 127.0.0.1:9050 https://api.myip.com
```

**본인의 ip와 다른 ip가 출력됨**을 볼 수 있습니다.

## References

[https://linuxize.com/post/how-to-install-tor-browser-on-ubuntu-18-04/](https://linuxize.com/post/how-to-install-tor-browser-on-ubuntu-18-04/)

[https://tor.stackexchange.com/questions/12341/how-to-properly-start-tor-service](https://tor.stackexchange.com/questions/12341/how-to-properly-start-tor-service)

[https://miloserdov.org/?p=3837](https://miloserdov.org/?p=3837)

[https://stackoverflow.com/questions/39257293/using-curl-with-tor-as-a-proxy-on-centos](https://stackoverflow.com/questions/39257293/using-curl-with-tor-as-a-proxy-on-centos)