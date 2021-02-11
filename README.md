# AI 관련 용어 및 상식 정리
* 머신러닝 : 컴퓨터에 샘플데이터를 통한 지속적인 학습을 통해서 문제에 대한 답을 얻어내는 방법 , 컴퓨터 공학 , 수학, 통계학의 교집합에 속함
* Supervise learing: 학습에 사용되는 데이터 결과가 정해져 있는 경우를 Supervised Learing이라고 한다.
* Un-Supervised learnging : Training Data에 target value가 없는경우이며 흔히 라벨링이 안되어 있다. 
* Regression Problem vs Classification : Supervised learing은 문제의 타입에 따라 두가지를 분류될 수 있다.
    * Regression problem : 기대되는 목적값이 연속성을 가지고 있을때 Regression problem
    * Classification problem : 목적값이 연속성이 없이 몇가지 값으로 끊어지는 경우
* **Linear Regression (선형 회귀)**
    * 선형 회귀 문제 : 데이터의 분포를 분석 하였을때 방정식으로과 같은 선형 그래프 형태로 정의될 수 있는 문제를 말한다.    
    ![예제>>출처https://bcho.tistory.com/967?category=555440](https://t1.daumcdn.net/cfile/tistory/261BBC3E544FA08635)

* 딥러닝에서 사용되느 여러 유형의 Convolution 소개
    * Convolutions
        * Convolutional layer을 정의하기 위한 몇개의 파리미터를 알아야 한다.   
            [파란색이 input이며 초록색이 output입니다](https://cdn-images-1.medium.com/max/1200/1*1okwhewf5KCtIPaFib4XaA.gif)
            * kernel Size : kernel size는 convolution의 시야(view)를 결정합니다. 보통 2D에서는 3x3 pixel로 사용한다.
            * Stride(걸음) : Stride는 이미지를 횡단할 때 커널의 스템사이즈를 결정합니다.기본값은 1이지만 보통 Max Pooling과 비슷하게 이미지를 다운샘플링하기 위해 stride를 2로 사용할 수 있다.
            * Padding : Padding은 샘플 테두리를 어떻게 조절할지를 결정한다. 패딩된 Convolution은 input과 동일한 output차원을 유지하는 반면 패딩되지 않은 Convolution은 커널은 1보다 큰 경우 테두리의 일부를 잘라버릴 수 있다.
            * input & output Channels: Convoltion layer는 input 채널의 특정 수를 받아 output채널의 특정 수로 계산합니다. 이런 계층에서 필요한 파라미터의 수는 i * O * K로 계산할 수 있다.K는 커널의 수 입니다.
    * Transposed Convolutions 
        * Transposed Convolution은 deconvolutional layer와 동일한 공간 해상도를 생성하기 점은 유사하지만 실제 수행되는 수락 연산은 다르다. Transposed Convolution layer는 정기적인 convolution을 수행하며 공간의 변화를 되돌립니다.
        * 어떤 곳에선 deconvolution이라는 이름을 사용하지만 실제론 deconvolution이 아니기 떄문에 부적절합니다.상황을 악화 시키기 위해 존재하지만 딥러닝 분야에선 흔하지 않는다.실제 deconvolution은 convolution의 과정을 되돌립니다.하나의 convolutional layer에 이미지를 입력한다고 상상하면 이제 출력물을 가져와 블랙박스에 넣으면 원본 이미지가 다시 나타납니다.이럴 경우 블랙 박스가 deconvolution을 수행한다고 할 수 있다.이 deconvolution이 convolution layer가 수행하는 것의 수학적 역 연산이다.
