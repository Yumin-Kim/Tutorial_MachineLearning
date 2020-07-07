# Machine Learning 시작!!!
## 1주차 Machine Learning 이란!!
* 하나의 소프트웨어 이다!! 하나의 문제에 여러 문제,룰들이 존재하여 프로그래밍하는데 많은 시간이 걸려 이런 부분을 단축하기 학습을 통해 판단하도록 프로그래밍을 하게끔하는것이다!!   
    * Supervised Learning >> label을 통해 주기적으로 교육시킴(Training data set) ex) 고양이 사진에 고양이 label을 통해 구별한다   
        *  ex) 공부 시간 투자로 시험 성적 예상 프르그램,-휴대폰 성능에 따른 배터리 시간 측정 - regression(회귀 분석) 
            * regression(회귀)분석 이란 : 연속형 변수들에 대해 두 변수 사이의 모형을 구현된 적합도를 측정해 내는 분석 방법    
            시간에 따라 변화하는 데이터나 어떤 영향, 가설적 실험, 인과 관계의 모델링등의 통계적 예측에 이용 
    * UnSupervised Learning >> 그룹화되어 있는 데이터를 보고 자유롭게 학습하는것
* tensorflow 기초
    * tensorflow 라이브러리를 활용하여 그래프(자료구조) 생성 한후에 Session을 생성하여 Session.run해야 우리가 원하는 값을 확인할 수 있다
    * tensorflow 에서 변수나 상수를 만든후에 바로 print하게 되면 Tensor()라는 데이터 형에 대한 정보가 나옵니다!!(Session을 통해서만 원하는 값을 확인 할 수 있는거 같다)
    * 현재 시점에서는 v2이고 Session은 v1에서만 사용 되었기 때문에 import tensorflow.compat.v1 as tf \n tf.disable_v2_behavior() 해주어야지 Session이 동작한다!!
    * 비어 있는 노드로 Graph를 생성하고 싶을때 tensorflow.placeholder(tensorflow.자료형) 하게 되면 비어있는 node를 생성하여 원하는 값을 생성 할수 있다
    * tensorflow 라이브러리를 사용할시 Tensor에 대한 이해도 중요!! ( Tensor( Rank : 몇차원의 array인지!! , Shapes : 각 요소당 몇개의 몇개가 들어 있는지!! , Types : 데이터 타입(float32,int32를 보통 많이 사용)  ) )
* Linear Regression(선형 회귀)
    * trainning 데이터를 통해 Linear한 일차 그래프를 찾아나는데 trainning을 많이 할수록 더욱 더 정확한 일차 그래프를 찾아 낼수 있다!!
    * Cost : 실제 데이터와 그래프 간의 거리
    * Variable : 우리가 일반적은로 사용하는 변수가 아닌 tensorflow가 사용하는 변수이며 trianable 한 Variable (훈련 할 수 있는 변수??)>>이와 같이 사용"tf.Variable(<initial-value>, name=<optional-name>)"
    * 구현 순서 : 일차함수 와 cost 최소화 함수 구현 >> placeholder를 사용했을때 sess.run(일차 함수 , feed_dict={})활용하여 구현!!
    ```
    간단한 파이썬 문법
    
    ```
