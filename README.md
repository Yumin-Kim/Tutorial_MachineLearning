# Machine Learning 시작!!!
## Machine Learning 1일차
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
* Cost Minimize 
    * cost function은 실제 값에서 그래프의 값까지 차이가 가장 자긍 값을 구하고자 하는것이다
    * Gradient descent algorithm   
    주 Minimize cost Function 이용  어떤 지점에서 시작할 수 있다    
    대부분 어느 지점 에서 시작 해도 경사가 없는 지점에 도달하며 미분을 주로 이용한다  
    cost function 설계 시 convex function(구현시 삼차원으로 구현시 빕 그릇을 엎어둔 모양처럼 보이며 어떤 지점에서 시작해도 똑같은 W ,b값을 구할 수 있다 )으로 구현 하지 않을 경우 각기 다른지점을 선택하여 진행했을때 다른 결과를 받을 수 있기 때문에 주의 해야한다
    * matplotlib.pyplot 구현 하는중 데이터 시각화 할때 자주 사용 되는 파이썬 라이브러리 이다
    * 일차 함수가 복잡할수록 미분의 알고리즘을 더욱 복잡해지면서 구현이 힘들다 이 문제를 해결 하기 위해서는 optimizer를 사용함으로써 수월하게 구할수 있다
## 2일차   
* MultiVariable Linear regression
    * 복습 : 가설의 일차 방정식(Hypothesis) , Cost Function , Gradient descent algorithm 
    * 여러개의 변수를 활용 하고 싶을때는 matrix(행렬)을 활용하여 손쉽게 Hypothsis를 구현할 수 있다.   
    * matrix를 활용할시 한번에 hypothsis의 값을 구할 수 있기때문에 빠르다!!   
    * tensorflow에서 제공하는 matmul를 사용하여 행렬 간의 곱셈을 진행 한다
        * 일일 데이터를 적어 줄수 없기 때문에 파일 로드 하는방법!!
            * pandas , numpy활용하여 읽어 올 수 있다
* Logistic (regression) classification(합병의 분류??)
    * binary Classification(binary는 둘중의 하나라는 의미 이고 Classification은 분류를 의미 ) 둘중의 카테고리를 분류하는것     
    ex) 페이스북 재미있는 영상만 보임 , 스팸메일과 원하는 메일 구별
    * 이와 같은 함수르 구현 하는 도중 0 ~ 1 로 결론을 내려야 하는데 큰값을 가지고 있는 것이 1 보다 무수히 클 확률이 대다수 이기 때문에 Linear Regression에서 사용했던 hypothesis를 조금 변형 하게된다   
    변형된 함수를 Logistic Function(simoid라고 말하는 이유는 S자 처럼 함수가 생겼기 때문이라도 일단 생각!!)라는 함수를 만들게 찾아 내게된다 큰값이 와도 1에서 준하게 된다!!
* Logistic 에서 사용될 Cost function
    * cost Function은 평균 * c() >> c(H(x) , y) { y = 1 >> -logH(x) , y = 0  >> -log(1-H(x)) } >> log 함수와 유사하여 이런식으로 구현한다!! >>이와 같이 구현 하게 되면 기존에 봤던 costFunction을 볼수 있다
    * cost funtion 구현후 Gradient descent 시작!!   
* 서브 프로젝트로 당뇨병을 판단할 수 있는 프로그램 만들어보기!!(https://www.kaggle.com/)이용하기!!

## 3일차
* Multinoimal Classification(softmax_regression)
    * 복습 : Logistic regression은 확률값으로 표현하고 이를 통해 두가지 중에 하나로 결론 내리는 방벙을 말한다
    * softmax regression은 이를 확정한 것인데 다중 분류를 하기 위해서 모델을 제안한 값이다(확률로 통해 A, B ,C를 매긴다)    
    여러 개의 연산 결과를 정규화하여 모든 클래스의 확률값의 합을 1로 만들자는 간단한 아이디어다.
    * Martrix multiplication 
        * 예를 들어 A, B ,C 라는 성적을 값을 구하는 식에서는  Multi Matrix를 활용하여 한번에 구현 하는 방법이다!!
        * 구현해서 결과는 0-1안에 값으로 표현되기를 원하면 초기 구현 당시는 1보다 큰값이 나올 확률이 높다 >> softmax function이라고 한다(실수를 확률로 구현하여 0or1또는 맞다 아니다로 표현하는것이 아니라  A , B , C 로 표현 할 수 있게 도와준다!!)
        * cost function or Gradient descent 동일하게 사용하여 구현 한다.
        * cost function명칭을 cross entropy, one_hot  다시 한번 보기!!
    * Logistic function은 내재적으로 cross-entropy loss와 깊은 연관이 있다
        * 로스를 줄이는 것이 Classification을 해결하는 과정!! 
            * Odds , log odds , logit , logistic(어떤 사건이 일어날 확률을 다양한 방법으로 표현할 수 있다)
                * Odds는 질병통계학이나 도박과 같이 카테고리가 두 개(성공과 실패/발병과 비발병)인 분야에서 자주 사용하는 표현이다
                * Log odds는 여전히 값이 클수록 성공확률이 큰 것을 의미하는 함수이다.
                * Logistic function은 Logit function의 역함수이다
    * cross-entropy , entropy 정리
        *  entropy는 불확실한 척도이고 , 정보 이론에서 불확실성을 의미 한다 >>> 여기서 불확실성이란 어떤 데이터가 나올지 예측라기 어려운 경우를 의미한다
        *  크로스 엔트로피는 실제 분포 에 대하여 알지 못하는 상태에서, 모델링을 통하여 구한 분포인 를 통하여 를 예측하는 것입니다. 
        * 크로스 엔트로피에서는 실제값과 예측값이 맞는 경우에는 0으로 수렴하고, 값이 틀릴경우에는 값이 커지기 때문에, 실제 값과 예측 값의 차이를 줄이기 위한 엔트로피
* Application & Tips
    * Learning rate , data preprocessing,overfitting
        * Learning rate 사용시기     
        ```
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
        ```
        이런식으로 Gradient Descent 할때 사용함!!   
        경우1. learning_rate 값을 너무 큰값을 할당 했을때 Gradient Descent활용하여 W,cost값을 구할때 점접간의 격차및 정확성이 떨어진다(overshoting 현상)      
        경우2. learning_rate 값을 너무 작은값을 할당할때는 굉장히 많은 시간이 걸린다      
        특별한 답은 없으면 보편적으로 0.01로 시작한다!! cost값을 비교하며 조율 필요!!
        * X data preprocessing for gradient descent(X데이터에 관련되어 선처리(미리)해야 하는 경우가 있다)      
        gradient descent를 통해 함수를 그리는데 x데이터 간의 격차 심할경우 양의 2차 곡선이 그려 지지 않고 타원으로 그려지는 경우가 된다(한쪽으로 외곡된 형태가 그려진다)주로 cost의 값이 분산되어 나오는 경우에 볼 수 있다!!         
        데이터 간격간의 격차를 최소화 하기위해서 original data >> normalize(일반화,정상화) data 할 필요가 있다     
        보편적으로 normalize 하는 경우 standardization을 사용하고 아래와 같은 python 코드를 작성한다    
        ```
        x_std[:,0] = ( X[:,0] - X[:,0] - X[:,0].mean() / X[:,0].std() )
        ```    
        * Overfitting         
        train data에 딱 맞게 모델링 하게 되면 생기는 문제점이며 train data에 없는 데이터를 삽입할때 에러가 발생할 수 있다
        * 해결방안
            1. 많은 training data 수집!!
            2. data 간의 중복을 제거한다!!
            3. 일반화 시킨다!!     
                * 너무 큰 weight(곡선 조금 제거)
                * cost function 에 시그마W^2(regularization) 더해서 W의 값을 최소화하기 위해서 구현은 아래와 같다!!   
                ```
                    l2reg = 0.001 * tf.reduce_sum(tf.square(W))
                ```    
* Machine Learning 동작 여부 확인
    * training set(기존의 학습시킬 데이터)에서 30~40%정도만 학습시키고 남은 data는 test set을 활용한다!!

## 4일차
* Deep Learning
* Tensor 이해
    * Shape      
    배열안 요소의 갯수(행,열에 관련된 정보를 준다)
    * ranks   
    몇차원 배열
    * Axis      
    axis는 0부터 시작해서 안으로 들어갈때마다 하나씩 증가한다 (axis[-1]은 마지막 axis를 의미)
    * Broadcasting    
    행렬이 달라고 계산할수있게 도와준다!!
    * reduce_mean(평균값) , reduce_sum , Argmax(배열의 index값)
* Neural Network 
    * XOR using NN    
    XOR로 인해 Linear 그래프를 찾지 못하는 현상이 벌어지고잇음   
    Multinomial classification을 사용하여 구현한다!!
    ```
    구현 코드
    K = tf.sigmoid(tf.matmul(X,W1)+b1)
    hypothesis = tf.sigmoid(tf.matmul(K,W2)+b2)
    ```
* Back propagation(다시 한번 찾아보기!!)
    * Tensorflow에서 내부적으로 구현되어있다!!
* Tensor board   
Neural Net시 출력하는결과를 가독성있게 보기 위해서 사용한다    

## 5일차
* DeeoLearning 책으로 배웠던 부분 다시 복습!!
* 이론 부분
    * tensorflow는 Tensor를 흘려 보내면서 데이터를 처리하는 라이브러리!!
        * Tensor는 다차원배열을 의미한다!! 
        * Rank , shape 이해!!
        * 계산 그래프 구조을 통해 노드에서 노드로 이동 합니다!!     
        텐서들을 흘려보내는 그래프 구조를 정의 하고, 정의한 그래프에 텐서들을 흘려 보내주도록 디자인 되었다   
        1. 그래프 생성
        2. 그래프 실행
        * 이와 같이 생성과 실행을 분리하는 이유는 많은 그래프를 생성했더라도 전체 그래프중에서 일부만 사용할 수 있게 하기위해서!!     
        전체 그래프 대신 일부 필요한 그래프만 실행할 수 있게끔 최대한의 효율로 연산이 수행된다!!     
        이 기법을 lazy Evalution 이라고 한다!!
        * Session    
        웹에서 사용하는 Session과 같은 의미로 사용되면 session을 열어 정보를 주고 받고 저장한후 세션을 종료하면 저장하고 있는것들을 비운다!!
        * TensorBoard는 시각화하는 툴로 그래프구조를 쉽게 볼 수 있다
* 구현 부분
    * random_normal() 정규분포에서 임의의 값을 추출함      
    random_normal() 균등 분포에서 임의의 값을 추출함
    * cost function(loss function)     
    적절한 파라미터 값을 알아내기 위해서 우리가 풀고자하는 목적에 적합한값인지 측정하는 역할!! >>손실함수 정의 다른 말로는 cost function     
    그중 많이 사용하는 함수로 평균 제곱 오차 우리의 예측값이 실제로 가까울수록 작은값을 가짐!!
    * tensorboard 
        * 현재 작업 하는 디렉토리 까지 cmd로 찾아 들어가서 tensorboard --logdir=./저장 파일명
        * 간단한 사용방법
            1. cost function 정의
            2. tensorboard를 위한 요약 정보 정의
                * tf.summary.scalar('y축명',costfunction명)
                * 요약한 정보를 직렬화(Serialize)해서 파일 형태로 저장하기위해서 tf.summary.FileWriter
            3. 요약한 정보들을 하나로 합친다!! merged = tf.summary.merge_all                  
            4. tensorboard_Writer = tf.summary.FileWriter("파일명",sess.graph)
            5. 반복문을 통해 결과값 파일에 추가!!
## 6일차
* Batch Gradient Descent ,Mini Batch Graident Descent ,stochastic Gradient Descent
    * Batch(배치) 사전적인 의미는 집단을 의미한다,배치는 한번에 여러개의 데이터를 묶어서 입력하는 것인데 GPU의 병렬 연산 기능을 최대란 효율적으로 사용하기 위한 방법이다!!
        * CPU : 입력받는 명령을 해석 / 연산후 이를 통해 결과 값을 출력하는 장치
            * 성능 영향 미치는 요인
                1. 클럭(동작 속도)의 수치 - 수치가 높을수록 단일 작업을 빠르게 처리함
                2. 코어(핵심 회로)의 수 - 멀티 테스킹(다중 작업)을 하거나 멀티코어 연산에 최적화된 프로그램을 구동하는데 이점
                3. 캐시 메모리의 용량 - 덩치가 큰 프로그램을 구동하거나 자주하는 작업을 반복 처리할때 작업 효율 높음
        * GPU : 최종 목적은 병렬 연산에 특화된 구조를 통한 높은 3d그래픽 처리 였지만 최근에는 범용적으로 이용되는 추세이다
        * cpu 와 gpu는 무엇이 다른가??
            * cpu는 명령어가 입력된 순서대로 데이터를 처리하는 직렬 처리 방식으로 특화되어 내부 면적의 절반이상이 캐시 메모리로 채워져있다
            * gpusms 여러 명령을 동시에 처리하는 병렬 처리 방식에 특화된 구조를 가지고 있다.또한 캐시메모리의 비중이 적다
    * Batch Gradient Descent는 Batch(데이터의 묶음)을 통해 한번에 이루어지기 때문에 업데이트, 계산하는 횟수가 적다(병렬 처리에 유리하다)
    * Stochastic gradient descent는 하나의 error gradient를 계산하고 Gradient 알고리즘을 적용하는 방법
    *  Batch Gradient Descent , Stochastic gradient descent의 절충적인 알고리즘인 mini-Batch gradinet descent이다
        * 전체 데이터셋에서 뽑은 Mini-batch 안의 데이터 m 개에 대해서 각 데이터에 대한 기울기를 m 개 구한 뒤, 그것의 평균 기울기를 통해 모델을 업데이트 하는 방법이다.
* ANN(Artificial Nenural Networks) - 인공 신경망
    * 등장 배경     
    생물학적 신경망 에서 아이디어를 얻었으며 인간의 뇌에 대한 연구가 발전하면서 인간의 뇌는 여러개의 신경 세포(Neuron)가 서로 연결되어 있고 이들이 병렬적으로 연산을 진행하면서 정보를 처리한다는 사실을 발견하게되고 각각의 뉴런들은 서로 전기적인 신호르 받으면서 연산을 수행하는 모습을 증명하고 찾아 볼 수 있게되었다    
    이에 반하여 컴퓨터는 메모리에서 값을 불러와서 cpu에서 순차적으로 연산을 처리한다.(순차적인 연산 처리기)이 결과롤 컴퓨터는 하나의 연산 속도가 빠른것을 알 수 있지만 기초적인 인지활동을 잘해내지 못하는게 문제점이였다    
    이문제를 보고 개발자들은 "컴퓨터도 인간의 뉴런처럼 병렬 처리 연산을 수행하도록 만들면 인지활동을 하지 않을까"의 의문을 통해 ANN이 나오게 되었다.
## 7일차
* unsupervised Learning    
    기존에 사용하였던 방식인 어떤 값을 예측하거나 분류하는 방식인 지도하는 방식이 아닌 데이터의 숨겨진 구조를 발견하는 목표의 학습 방법이다
    * Autoencoder(오토인 코더)      
        책에서 배울 Autoencoder는  비지도 학습 방법중 하나이다!    
        입력층의 노드의 개수와 출력층의 노드의 개수가 동일한 구조의 인공 신경망이며 출력은 원본 데이터를 재구축한 결과를 보내준다!! >> 재구축하는 것을 Pre-Training   
        <strong>하지만 Autoencoder(오토인 코더)는 원본 데이터를 재구축한 결과보다 은닉층의 출력값을 핵심으로 하고 있다</strong>    
        은닉층의 노드 갯수가 입력층 , 출력층의 노드의 개수보다 적으며 적은 이유는 더 작은 표현력으로 모든 특징을 분석하기 위함이고     
        은닉층의 출력값은 원본 데이터에서 불필요한 특징을 제거한<strong>압축된 특징</strong>을 지니고 있다
        * 예시 : 주어진 데이터를 통해 강아지인지 고양이 인지 분석하기!!      
            주어진 데이터: 주인의 나이 , 주인의 몸무게 , 동물의 길이 , 동물의 몸무게      
            주어진 데이터를 보아 주인의 나이나 주인의 몸무게는 동물의 분루에서는 아무 의미 없는 데이터로 잉여 데이터라고 한다   
            은닉층은 이와 같은 불필요한 데이터를 걸러 남은 특징을 의미하며  은닉층의 출력값은 동물 분류에서 필요한 데이터 2개만 가지고 있다!!   
            이렇게 압축된 특징을 나타내는 은닉층의 출력값을 원본 데이터 대신 분류기의 입력으로 사용하여 더욱 좋은 분류의 기능 기대할 수 있다
    * 파인 튜닝(Fine - Tuning) 과 전이 학습(Transfer - Learning)    
        Mnist의 데이터를 그대로 softmax 분류기의 입력값에 넣은것과 비교하면 시간 , 비용적인면에서 확연히 차이가 난다    
        파인 튜닝 기법을 다른 말로는 전이학습이라고도 하는데 사용할 경우 임의의 값으로 초기화한 파라미터를 처음부터 다시 학습하는것보다 훨씬빠름(쟐 분류한 데이터를 가져와서 우리가 또 다시 새로운 분류를 원하고자 하는 데이터만를 분류기에 입력한다!!)    
        기존에는 다른 목적의 문제를 해결하려고 할때마다 매번 새로운 트레이닝을 진행했지만 전이 학습을 통해 새로운 문제를 해결할때 다은 목적을 해결하고자 학습 파라미터를 지식으로 간주하여 해결하는 방식이다     
    * 과정 정리 데이터 재구축(데이터 분류에 대한 목적은 고려하지 않고 오직 Mnist 데이터 재구축하는데 최적화!!) >> 재구축한 데이터를 활용한 파인튜닝을 통해 Mnist 숫자 분류 최적화!!         
## 8일차
* overfitting 생기는 이유    
    Train시에는 오류없이 100%성공 하지만 Testdata를 사용하게 되면 우리가 생각했던되로 동작 하지 않는다!!   
    이유는 Train Data에 정획히 맞아 들어가서 그런 현상이 일어난다        
    해소 방법 많은 데이터 사용 , feature를 줄이고 Regularization을 사용해야 한다.   
    weight값을 줄인다 ( 12reg = 0.001(Learning_rate) * tf.reduce_sum(tf.square(W)) )
    * Dropout : 잘구성되어 있는 Neural Networks의 관계을 줄이는방법이다!!(몇개의 노드를 사용하지 않는것이다!!)
* CNN(Convolution Neural Networks)컴볼루션 신경망     
    이미지 분야를 다루기에 최적화된 인공 신경망 구조이다 크게 Convolution Layer, Pooling Layer(Subsampling)으로 구성되었다   
    conevolution을 구성 하기 위해서 RELU를 활용하여 그다음 W값을 구성 하게 되는데 중간 중간 Pooling Layer이 있는것을 확인 할 수 있다    이미지를 Filtering을 통해 convolution Layer를 구성하게(Layer의 두께는 filter의 갯수에 따라 달라 진다) 되는데 layer에서 하나의 레이어를 가지고 와서 resize(sampling 이미지 사이즈를 줄임)을 하게되는데 이 행위를 Pooling이라고 하며 이 행위를 반복하여 Pooling Layer을 구성하게 된다  
    Pooling의 EX)4X4 layer를 max pool 2X2 filter를 사용하면 2X2가 구성된 Layer가 생기게되는데 값을 max pool이라는것은 2x2 filter를 할때 현재 filter안에서 가장 큰 값을 가지고 오는것을 말한다!!     
    Pooling과 RELU의 순서는 상관없이 개발자의 선택 사항이다!!   
    이 모든 과정은 이미지의 특징을 뽑아 낸다고 생각하면 될듯!! 
### 간단한 개념 정리
* 인공지능 용어
    * AI >> Machine Learning >> Deap Learnning
    * AI 인간의 지능을 기계 등에 인공적으로 구현 한것
    * Machine Learnning 기계 학습의 한 분야로 컴퓨터가 학습 할 수 있도록 알고리즘 , 기술 개발하는 분야
    * Deap Learning 여러 비선형 변환 기법의 조합을 통해 높은 수준의 추상화( 다량의 복잡한 자료들에서 핵심적인 내용만 추려내는 작업 )을 시도하는 기계학습 알고리즘의 집합      

* ANN ( Artificial Neural Network )
    * 머신러닝의 한분야로 딥러닝은 인공신경망을 기초로 하고있다
    * 인간의 신경망 원리와 구조를 모방하여 만든 기계 학습 알고리즘
    * 인간의 뇌에서 뉴런들이 어떤 신호, 자극 등을 받고, 그 자극이 어떠한 임계값(threshold)을 넘어서면 결과 신호를 전달하는 과정에서 착안한 것입니다.
    * 은닉층에서는 활성화함수(Cost Function)를 사용하여 최적의 Weight와 Bias를 찾아내는 역할을 합니다.     
ex) Optimizer할때       

* DNN ( Deep Neural Network )      
    * 준비중
* CNN ( Convolution Neural Network )
    * 데이터의 특징을 추출하여 특징들의 패턴을 파악하는 구조입니다.
    * 이 CNN 알고리즘은 Convolution과정과 Pooling과정을 통해 진행됩니다.
    * Convolution Layer와 Pooling Layer를 복합적으로 구성하여 알고리즘을 만듭니다 .      

* RNN ( Recurrent Neural Network )
    * 알고리즘은 반복적이고 순차적인 데이터(Sequential data)학습에 특화된 인공신경망의 한 종류로써 내부의 순환구조가 들어있다는 특징


* Linear regression(선형 회귀)     
    * 준비중
* cost Function 정의!!
    * 준비중
* Optimizer
    * cost function Minimize algorithm을 구현 하는데 사용
    * 여러 방법 중에 Gradient Descent를 사용하며 미분 공식을 따르고 있다
    * Gradient Descent를 사용 하기 위해 cost function을 (측정값 - 예측값 )의 평균이 아닌 ((측정값 - 예측값 ) ^ 2) / n을 해줘야 한다      
    
#### 머신러닝하면서 파이썬 문법 정리!!
    ```
    간단한 파이썬 문법
    range(시작점 , 종료 숫자 , step) 시작 숫자나 step은 생략 가능하다
    for i in range(5):
    print(i)   
    result   
    0
    1
    2
    3
    4
    리스트 관련 메소드 
    리스트에 요소 추가(append)
    a = [1,2,3]
    a.append(4)
    a
    result
    [1,2,3,4]
    요소 제거(remove)
    with function as variableName 
    호출하지 않아도 자동으로 호출이됩니다. 
    _(underScore)의 의미
    1.인터 프리터에서 마지막 값을 저장 한다
    2.값을 무시하고 싶을때
    3.변수나 함수명에 특별한 의미또는 기능을 부여할때
    slice notation(표기법)
    slice는 start:stop[:step]의 형식으로 쓸 수있습니다. 여기서 [:step]은 써도 되고 안써도 된다는 의미입니다.

    step을 명시하지 않을 경우에는

    a[start:end] # start부터 end-1까지의 item
    a[start:] # start부터 리스트 끝까지 item
    a[:end] # 처음부터 end-1까지의 item
    a[:] # 리스트의 모든 item

    배열 , List 선언 초기화 시 arr = [1,2,3] 불러오기 arr[0]
    튜플 선언 초기화 시 tup = () 불러 오기 tup[0]
    딕셔너리 선언 초기화 시 dic = {"js":1 , "java" : 2} 불러오기 dic["js"]

    plt.imshow(image.reshape(3,3),cmap='Greys')
    인자의 값을 넣을때 앞에 키워드 = "값" 함수를 호출하는 경우가 많다
    이런 경우 python 문법에 키워드 매개변수라는것이 있는데 위치 매개변수 전달 방식이라고 한다
    순서대로 인자르 입력받아 실행한다
    이와 번대로 직접 매개변수명을 지정하여 매개변수를 전달하는 방법 또 있다!!
    def value_times(*values,times = 2):
        for value in values:
            print(times * value)
    value_times(1,2,3,4,times=5)         
    이런식으로 사용되며 함수 정의 할때 매개변수로 times로 키워드로 지정하여서 함수 호출시 다른 인자 값을 넣어주게되면 error가 발생 하게 된다.
    ```
