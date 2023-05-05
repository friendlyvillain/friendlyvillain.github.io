---
title: Github 페이지 Chirpy 테마 적용 방법 (ver. 2023)
author: jh
date: 2023-04-01 22:40:00 +0900
categories: [Git Page, Chirpy]
tags: [git, jekyll, chirpy]
comments: true
pin: true
---

## 준비

로컬에 ruby, bundle 환경 구축이 완료되어 `bundle exec jekyll serve` 명령어로 [127.0.0.1:4000](https://127.0.0.1:4000) 을 통해 Chripy 테마 확인 가능


## 개요

Chripy 테마가 적용된 Github 페이지 생성을 위해 구글링을 하다보면 로컬에서는 Chripy 테마 적용이 잘 되지만 Github와 연동하면 적용이 안된다는 글을 많이 볼 수 있다. 

### 문제상황 1

가장 흔한 이슈가 자신의 Github Page repo에 커밋한 이후, 링크에 들어가면 다음과 같이 ---layout: home # Index page --- 만 보인다는 것이다.

![Desktop View](/assets/img/posts/tutorial/page_error.png){: width="500" height="100" }

이에 대한 해결 방안으로 Github repo 설정에서 branch를 gh-page로 변경하라고 안내하고 있지만, 설정 화면에 들어가면 main branch 외에 아무것도 보이지 않는 것을 확인할 수 있다. 

#### 해결방안 1

많은 사람들이 위와 같은 상황을 겪었고, 이에 대한 조치로 다음과 같은 설정을 통해 해결되었다는 내용을 확인 할 수 있다.

>Github Setting - Pages - Bulid and development 에서 Source를 Github Actions 로 변경 이후, jekyll.yml 커밋
{: .prompt-tip }

### 문제상황 2

정상적인 경우, [**위와 같이 조치**](#해결방안-1)하면 Github repo 주소와 Chripy 테마가 연동이 되지만 여전히 ---layout: home # Index page --- 가 뜨는 문제 상황이 발생하는 경우가 있다.
Setting에서 확인해보면 Github Actions에서 Source가 적용되지 않았기 때문이다.
구체적인 원인을 확인해 보기 위해, Actions 탭에서 확인해보면, build 과정에서 오류가 발생한 것을 확인할 수 있다.  
이는 Linux 기반이 아닌 플랫폼의 로컬 (Win OS, MAC OS) 에서 push를 한 경우 발생하는 문제이다. 

#### 해결방안 2

필자도 Apple Silicon 기반의 로컬 환경에서 위와 같은 상황이 발생했고, [Chripy Demo: Getting Started](https://chirpy.cotes.page/posts/getting-started/#deploy-by-using-github-actions)에서 Chripy 테마 개발자가 기술하였듯이 로컬이 Linux 기반의 플랫폼이 아닌 경우, 다음과 같은 bundle 명령어를 통해 Linux platform을 추가해야 되는 것을 알 수 있다.

  ```console
  $ bundle lock --add-platform x86_64-linux
  ```

위의 명령어를 로컬에서 수행하고, git repo에 커밋한 이후, 다시 Git Actions로 설정에서 Jekyll.yml을 커밋하면 build가 정상적으로 수행되고 얼마 이후, Github repo 주소와 Chripy 테마가 적용되는 것을 확인할 수 있다. 

## 결론

Chripy 테마는 Github 페이지로 인기있는 테마인 만큼, 많은 사용자들의 의견을 확인할 수 있다. 
사용하면서 발생한 문제에 대한 대부분의 해결방안은 Chripy 테마 개발자 페이지의 issue 탭에서 키워드로 검색해보거나 Chripy 데모 페이지에서 가이드를 확인해보면 해결방안을 찾을 수 있다.

## 참고 링크

[Chripy Theme 개발자 페이지](https://github.com/cotes2020/jekyll-theme-chirpy)

[Chripy Demo 페이지](https://chirpy.cotes.page/)