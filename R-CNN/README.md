# 프로포노 ICT에 사용 되는 인공 신경망 MASK R-CNN 공부하기!

---

## 중간 과정 기록

1. 2021 . 05 . 04
   - 현재 가지고 있는 자료가 동작하지 않는 부분으로 인해서 컴퓨팅 파워 문제 있지 , 코드 자체 문제인지 모르기 때문에 기존 사용할 인공 신경망에 대해서 공부를 진행하고 개발 할 예정이다.
   - 왜 굳이 MASK R-CNN을 사용하는지에 대해서 자료 조사 하기

---

- 컴퓨터 비전에서의 문제들을 크게 다음 4가지 분류할 수 있다.

1. Classification
2. Object Detection
3. Image Segmentation
4. Visual relationship

![비교 이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqAscC%2FbtqA8ikQxfq%2FpoxIz7m0wjHadq4uJ5VwJ1%2Fimg.png)

- 상단의 이미지를 통해서 명확한 차이를 알 수 있다.
- Classification : Single object에 대해서 object의 클래스르 분류하는 문제
- classification + Localization : single object에 대해서 object의 위치를 bounding box로 찾고 (localization) + 클래스르 분류하는 문제이다.(Classification)
- Object Detection : Multi object에서의 각각의 object에 대해 classification + Localization 을 수행하는 것이다.
- Instance Segmentation : Object Detection과 유사하지만 다른점은 object의 위치를 bounding box가 아닌 실제 edge로 찾는것이다.

* Object detection에는 1-stage detector , 2-stage detector가 있다.
  - 2-stage detector의 동작 과정
    - Selective search , Region proposal network와 같은 알고리즘을 및 네트워크를 통해 object가 있을만한 영역을 우선 뽑아낸다 . 이 영역을 ROI(Region of Interest)라고 한다.
    - 이런 영역들을 우선 뽑아내고 나면 각 영역들을 convolution network를 통해 classification , box regression(Localization)을 수행한다.
  - 1-stage는 2-stage와 다르게 RoI영역을 먼저 추출하지 않고 전체 image에 대해서 convolution networ
