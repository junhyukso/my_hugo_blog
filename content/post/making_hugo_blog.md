---
title: HUGO블로그 시작하기
date: 2020-11-27
description : HUGO의 설치법과 간단한 사이트 제작법을 알아보도록 하겠습니다.
cover : https://d33wubrfki0l68.cloudfront.net/c38c7334cc3f23585738e40334284fddcaf03d5e/2e17c/images/hugo-logo-wide.svg
tags:
- Web
- HUGO
---

# HUGO

HUGO는 GO언어로 작성된 정적 사이트 생성기 입니다.

정적 사이트 생성기로 유명한 Framework들은,

- Jekyll : Ruby
- Hexo : NodeJS
- Gatsby : React
- HUGO : Go

들이 있습니다. 저는 원래 Hexo를 사용하다가, HUGO로 이전하기로 하였는데 그 이유는 다음과 같습니다.

- **커뮤니티가 대부분 영어**
    - Hexo같은 경우는 대부분 중국 커뮤니티입니다.
- **빠름**
    - HUGO는 가장 빠른 정적 프레임워크 생성기로도 유명합니다.
- **테마 관리가 깔끔함**
    - HUGO는 Look Up Order라는것이 있어, 내부 테마를 전혀 수정하지 않고도 커스텀이 가능합니다.

# HUGO 설치

- **Windows10 기준** 입니다.
- 공식 설치방법은 해당 문서를 참조바랍니다.([https://gohugo.io/getting-started/installing](https://gohugo.io/getting-started/installing))

### Chocolatey 설치

Chocolatey는 패키지 매니지먼트 툴의 일종입니다.

- Powershell.exe를 관리자 권한으로 열어준후, 아래 명령어를 입력해줍니다.

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```

- 설치에 대략 1분정도 걸립니다.
- 설치가 완료되었다면, `choco` 를 입력하여 설치가 되었는지 확인합니다.

```powershell
PS C:\Windows\system32> choco
Chocolatey v0.10.15
Please run 'choco -?' or 'choco <command> -?' for help menu.
```

### Hugo 설치

- Chocolatey를 이용하여 Hugo를 설치합니다.

```powershell
choco install hugo -confirm
```

- `hugo version` 을 입력하여 설치가 정상적으로 완료되었는지 확인합니다.

# HUGO 사이트 만들기

```powershell
hugo new site [SITE_NAME]
```

- [SITE_NAME]이 만들어질 사이트의 이름입니다. 원하시는걸로 입력하세요

# HUGO 테마 추가하기

- [https://themes.gohugo.io/](https://themes.gohugo.io/)
- 위 주소로 가시면 여러가지 테마가 있습니다.
- 원하시는 테마를 선택하시고 해당 테마의 Git을 themes 아래에 clone 합니다.
    - 혹은 submodule로 등록해도 됩니다.

Theme 설치 구조 예시 (ananke)

- HUGO에서 테마의 설치는 프로젝트 폴더의 themes 아래에 테마 폴더가 있으면 됩니다.
- 이후, config.toml에 테마를 지정합니다.
    - theme = "테마이름"

# HUGO 글 작성하기

```powershell
hugo new posts/[파일이름.md]
```
content/post 폴더에 해당 파일이 생성됩니다.

# HUGO 서버 실행하기

- 로컬에서 간단하게 웹서버를 돌려보는 용도입니다.

```powershell
hugo server -D
```

# 정적 블로그 생성

프로젝트 폴더에서,

```powershell
hugo
```

입력시 public 폴더에 만들어진 정적파일들이 생성됩니다.  
이후 호스팅시, 이 public폴더의 내용들을 호스팅하면 됩니다.  

HUGO로 간단하게 블로그를 제작해보았습니다.