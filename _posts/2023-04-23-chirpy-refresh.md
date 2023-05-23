---
title: Github 페이지 Chirpy 테마 변경내용 실시간 적용 방법
author: jh
date: 2023-04-23 19:53:20 +0900
categories: [Git Page, Chirpy]
tags: [git, jekyll, chirpy, refresh, twa, service-worker]
comments: true
pin: true
---

## 개요

로컬에서는 Chirpy 테마가 적용된 페이지를 수정 e.g., 디자인 수정, 업로드/삭제 등을 하면 반영이 되지만 Github 페이지에는 정상적으로 빌드가 되었음에도 불구하고 수정사항이 실시간으로 반영되지 않을 것이다.
이에 대한 임시 해결책으로 웹 브라우저 내에서 강한 새로고침 (`Ctrl+Shift+R` 또는 `Cmd+Shift+R`)을 통해 수정사항이 적용된 것을 확인할 수 있지만 다시 페이지에 방문해보면 여전히 수정사항이 반영되지 않고, 그대로인 상황이 발생한다.
커밋하고 일정 시간이 지난 이후, 페이지에 접속하면 방문자에게 `A new version of content is available` 창을 띄워주면서 Content를 업데이트할 수 있다는 메시지를 띄워주고 update 버튼을 방문자가 클릭해야 수정사항이 반영되도록 동작한다. 

### 문제원인
개발자가 페이지에 접속하지 못하는 상황이 발생하더라도 한번이라도 방문한 적이 있는 페이지라면 PWA를 통해 브라우저에 저장된 캐시를 통해 이전에 방문한 페이지를 로드할 수 있도록 구현하였기 때문이다. 

### 해결방안
필자의 경우에는 [CV 페이지](https://friendlyvillain.github.io/digital_cv/)를 또다른 GitHub 페이지와 연동을 해두었는데 메인 깃허브 페이지에서 update를 누르지 않는 경우, CV 페이지의 내용이 수정되지 않는 것을 확인하였다. 
대부분의 경우, 방문자가 update 버튼을 클릭하도록 유도하는 것이 번거롭고, 수정된 내용이 실시간으로 GitHub 페이지에 잘 적용되었는지 확인하고 싶은 경우가 많을 것이다.
Chripy 개발자 페이지에서 확인해보니 나와 같이 실시간으로 수정된 내용이 반영되기를 원하는 [사용자의 글](https://github.com/cotes2020/jekyll-theme-chirpy/issues/527#issuecomment-1079998986)을 확인할 수 있었고, 다음과 같이 service worker 설정 수정을 통해 커밋과 동시에 페이지 업데이트가 되도록 설정하여 해결할 수 있었다.

> `/assets/js/pwa/sw.js`{: .filepath} 의 기존 내용을 모두 삭제한 이후, 다음 코드를 추가
{: .prompt-info }

```js

  self.addEventListener("install", (event) => {
    self.skipWaiting();
  });
  
  self.addEventListener("activate", (event) => {
    self.registration
      .unregister()
      .then(() => self.clients.matchAll())
      .then((clients) => {
        clients.forEach((client) => {
          if (client.url && "navigate" in client) {
            client.navigate(client.url);
          }
        });
      });
  });

  ``` 


## 참고 링크

[Service worker self-destroying](https://github.com/NekR/self-destroying-sw)