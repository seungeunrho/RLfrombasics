# RLfrombasics

<img src="https://user-images.githubusercontent.com/8207326/93460041-a7096500-f91d-11ea-9797-583677d2c898.jpg" height="400"></img>

"바닥부터 배우는 강화학습"에 수록된 코드를 모아놓은 레포입니다.
누군가에게 도움이 되는 책이었으면 좋겠습니다.
감사합니다.

This repo provides all the codes from the book "RLfrombasics"
Hope this book is useful to somebody
Thankyou :)

# Versions (버전)

GYM 라이브러리 버전 0.26.2 이상 에서 테스트 되었습니다.
이보다 낮은 버전의 GYM 라이브러리를 사용하신다면 본 코드가 올바르게 동작되지 않을 것입니다!

Please use latest version of the GYM library. 

# Typo(오타)

1. 챕터 5 : ch5_mclearning.py 코드 97라인  <br>
(수정 전) cum_reward = cum_reward + gamma * reward <br>
(수정 후) cum_reward = reward + gamma * cum_reward <br>
Thanks to goodjian7

2. 챕터 3 : 67, 69 페이지 OX 퀴즈 <br>
(수정 전) r_t+1 + gamma * r_t+1 + ... <br>
(수정 후) r_t+2 + gamma * r_t+2 + ... <br>
Thanks to namdori61
