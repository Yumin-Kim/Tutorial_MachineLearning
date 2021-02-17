# AI 관련 용어 및 상식 정리
* 해당 자료는 https://bcho.tistory.com/ 참고한 자료이다.
* 머신러닝 : 컴퓨터에 샘플데이터를 통한 지속적인 학습을 통해서 문제에 대한 답을 얻어내는 방법 , 컴퓨터 공학 , 수학, 통계학의 교집합에 속함
    * 이러한 머신 러닝의 분야 중에서 인공 지능망(ANN-Aritification Nenual network)라는 기법이 있는데 사랑의 뇌의 구조를 분석하여 사람뇌의 모양이 여러개의 누런이 모여서 이루어진것 처럼 머신 러닝의 학습 모델을 여러개의 계산 노드를 여러 층으로 연결해서 만들어 낼수 있다.
    * 초창기 문제 해결부분에서 부정적인 영향이 있었으나 최근 연구를 통해 어려움을 극복할 수 잇는 있다는 것을 증명하고 딥러닝이라는 이름으로 다시 브랜딩 되었다.
* Supervise learing: 학습에 사용되는 데이터 결과가 정해져 있는 경우를 Supervised Learing이라고 한다. 문제에 대한 정답을 주고 학습을 한후 나중에 문제를 줬을때 정답을 구하는도록 학습하는 방식
    * 문제에 대한 답을 전문 용어로 labeled data(라벨된 데이터)라고 한다
* Un-Supervised learnging : Training Data에 target value가 없는경우이며 흔히 라벨링이 안되어 있다. 문데로만 학습을 시키는 방식
* Regression Problem vs Classification : Supervised learing은 문제의 타입에 따라 두가지를 분류될 수 있다.
    * Regression problem : 기대되는 목적값이 연속성을 가지고 있을때 Regression problem    
        ![예제 사진](https://t1.daumcdn.net/cfile/tistory/246FEC4F57F34BC701)    
        해당 사진을 통해서 알 수 있듯이 결과갓이 연속성을 가지고 있을때 Regression 문제라고하며 에제는 택시거리에 따른 비용의 문제이다. 결과는 택시 값이 기대되는 경우로 변수와 결과 값이 연속적으로 이루어지는 것을 사진을 통해서 알 수 있다.
    * Classification problem : 목적값이 연속성이 없이 몇가지 값으로 끊어지는 경우
        ![예제 사진](https://t1.daumcdn.net/cfile/tistory/24610B4F57F34BC30D)     
        입력값에 대한 결과값이 연속적이지 않고 몇개의 종류로 딱딱 나눠서 끊이지는 결과가 나오는것을 야기한다. 강아지나 고양이 차 등등 특정 종류에 대한 결과 값이 나오는것 역시 classification이라 한다.    
* **Linear Regression (선형 회귀)**
    * 선형 회귀 문제 : 데이터의 분포를 분석 하였을때 방정식으로과 같은 선형 그래프 형태로 정의될 수 있는 문제를 말한다.
    * 결과 값이 있고 그 결과값을 결정할것이라고 추정되는 입력 값과 결과값의 연관 관계를 찾는것이고 이를 선형 관걔을 통해 차는 방법이 선형 회귀이다.    
    ![예제>>출처https://bcho.tistory.com/967?category=555440](https://t1.daumcdn.net/cfile/tistory/261BBC3E544FA08635)
    * Hypothesis(추론)는 추론 알고리즘의 집합 , 하나의 수학 공식을 정의하는 단계라고 이해??
    * 코스트 비용(cost Function)
        * 가설을 통해 정의된 함수에서 W , b 임의로 지정하여 실측데이터와 오차 범위를 줄이기 위해 정의하며 오차 범위를 최소가 되도록 w ,b값을 구하여야 한다.
        * 다른 포스팅을 통해서 정리
            * 모델을 학습 할때 cost즉 오류를 최소화하는 방향으로 진행된다.
            * 비용이 최소화되는 곳이 성능이 가장 잘나오는 부분이며 가능한 비용이 적은 부분을 찾는것이 최적화이고 일반화의 방법이다.
            * 비용은 cost 혹은 loss이 얼마나 있는지 나타내는것이 cost funcion , loss function이라고 한다.
                *  cost 와 loss function은 큰 차이를 가지지 않지만 미묘한 차이가 있다.(현재는 거의 비슷하다고 이해)
                * loss function : single data set을 다룬다.
                * cost function : loss function의 합 , 평균 에러를 다룬다. single data set이 아니라 entire data set을 다룹니다.**순간순간의 loss를 판단할떈 loss function을 사용하고 학습이 완료된 후에는 cost function을 확인한다.**
                * Objective function : 모델(함수)에 대해서 우리가 가장 일반적으로 사용하느 용어로서 최댓값 , 최솟값을 구하는 함수를 말한다. 
                * 결론 : **loss fucntion <= cost function <= objective function**
    * Optimizer
        * 코스트 함수의 최소값을 찾는 알고리즘으 옵티마이져라고 하는데 상황에 따라 여러 종류의 옵티마이저를 사용할 수 있다. 여기서는 경사 하강법(Gradient Descent)라는 옵티마이져에 대해서 기초이다.
        * Gradient Descent(경사 하강법)
            * 코스트 함수를 최적화 시킬수 있느 여러가지 방법이 있지만 Linear regression의 경우에는 경사 하강법라는 방식을 사용한다.
  * 간단한 용어 정리
        * epoch
            * 인공 신경망에서 전체 데이터 셋에 대해 forward pass / backward pass 과정을 거친것을 말하며 즉 전체 데이터 셋에 대해 한번 학습을 완료한 상태
            * 신경망에서 사용되는 backpropagation algorithm은 파리미터를 사용하여 입력부터 출력까지의 각 계층의 weight를 계산하는 과정을 거치는 forward pass를 반대로 거슬러 올라가며 다시 한번 계산 과정을 거처 기존의 weight를 수정하는 backfoward pass로 나뉜다.
            * 이 전체 데이터 셋에 대해서 forward pass + backward pass가 완료되면 한번의 epoch가 진행되었다고한다. 
            * **우리는 모델을 만들때 적절한 epoch값을 설정해야만 overfitting 와 underfitting을 방지할 수 있다.**
        * iteration - 반복의 , 되풀이하는 
            * 알고리즘이 iterative 하다는것 : gradient descent(경사 하강)와 같이 결과를 내기위해서 여러번의 최적화 과정을 거쳐야되는 알고리즘
            * 다루어야하는 데이터가 많기도하고 한번에 최적화된 값을 찾느것은 힘들기 때문에 이와 같은 과정을 수행한다.
        * batch-size
            * 한번의 batch마다 주는 데이터 샘플의 size,여기서 batch는 나눠진 데이터 셋을 뜻하며 iteration은 epoch를 나누어서 실행하는 횟수
            * 메모리의 한계와 속도 저하떄문에 대부분의 경우에는 한번의 epoch에서 모든 데이터를 한꺼번에 집어넣을수는 없어 이와같이 데이터를 나누어서 주게 되는데 이때 몇번 나누어서 주는가를 iteraition 각 iteration마다 주는 데이터 사이즈를 batch-size하고 한다.
        ![예시 출처 : https://www.slideshare.net/w0ong/ss-82372826](https://mblogthumb-phinf.pstatic.net/MjAxOTAxMjNfMjU4/MDAxNTQ4MjM1Nzg3NTA2.UtvnGsckZhLHOPPOBWH841IWsZFzNcgwZvYKi2nxImEg.CdtqIxOjWeBo4eNBD2pXu5uwYGa3ZVUr8WZvtldArtYg.PNG.qbxlvnf11/20190123_182720.png?type=w800)  
        * 예시
            * 전체 2000개의 데이터가 있고 epochs = 20 , batch-size=500 가정
            * 그렇다면 1epoch는 각 데이터의 size가 500인 batch가 들어간 네번의 iteration으로 나누어진다.
            * 그리고 전체 데이터셋에 대해서 20번 학습이 이루어졌으며 iteration 기준으로 보면 총 80번의 학습이 이루어진다.    
## 머신 러닝의 순서
* 기본 개념은 데이터를 기반으로해서 어떤 가설(공식)을 만들어 낸 다음 그 가설에서 나온 값이 실제 측정값과의 차이(코스트 함수)가 최소한의 값을 가지도록 변수에 대한 값을 컴퓨터를 이용해서 찾은 후, 찾아진 값을 가지고 학습된 모델을 정의해서 예측을 수행하는것이다.
    * 학습단계 : 즉 모델을 만들기 위해서 실제 데이터를 수집하고 이 수집된 데이터에서 어떤 특징을 가지고 예측을 할것인지 특징을 정의한 다음에 이 특징을 기반으로 예측을 한 가설을 정의하고 가설을 기반으로 학습시킨다.
    * 예측 단계 : 학습이 끝나면 모델(함수)가 주어지고 예측은 단순하게 모델에 값을 넣으면 학습된 모델에 의해서 결과 값을 리턴해준다
## CNN(Convolution Neurak Network)
* CNN은 전통적인 뉴럴 네트워트 앞에서 여러 계층의 Convolution을 붙인 모양이 되는데 그이유는 CNN은 앞의 Convolution 계층을 통해서 입력 받은 이미지에 대한 특징를 추출하게 되고 이렇게  추출된 특징을 기반으로 기존의 뉴럴 네트워크를 이용하영 분류를 해내게 된다.   
![CNN 이미지](https://t1.daumcdn.net/cfile/tistory/213C6141583ED6AB0A)
* Convolution Layer
    * Convoluiton Layer는 입력 데이터를 부터 특징을 추출하는 역할을 한다.
    * Convolution Layer는 특징을 추출하는 기능을 하는 filter(필터)와 이 필터의 값을 비선형 값으로 바꾸어 주는 Activation function으로 이루어진다.   
        ![Filter와 Activation function으로 이루어진 Convolution Layer 사진](https://t1.daumcdn.net/cfile/tistory/23561441583ED6AB29)
        *  Filter
            * 필터는 그 특징이 데이터에 있는지 없는지를 검출해주는 함수이다. 필터는 구현에 있어 행렬로 정의 된다.
            * 필터는 입력받은 데이터에서 그 특성을 가지고 있으면 결과 값이 큰값이 나오고 특성을 가지고 있지 않으면 결과 값이 0에 가까운값이 나오게 되서 데이터가 그 특성을 가지고 있는지 없는지 여부 확인 가능
            * 입력 값에는 여러가지 특징이 있기 때문에 하나의 필터가 아닌 여러개의 다중 필터를 같이 적용가능하다.
            * 각기 다른 특징을 추출하는 필터를 조합하여 네트워크에 적용하면 원본 데이터가 어떤 형태의 특징을 가지고 있는지 없는지 확인 판단이 가능하다. 
                * Stride
                    * 5 X 5 원본 이미지를 3 X 3인 필터를 적용해서 추출하는 과정
                    * 필터를 적용하는 간격값을 stride라고 하고 필터를 적용해서 얻어낸 결과를 Feature map 또는 acitivation map이라고 한다.
                    * 이 필터를 어떻게 적용하기 위해서 존재한다.    
                    ![해당 이미지는 적용하는 단계를 보여주고 있다.](https://t1.daumcdn.net/cfile/tistory/210B0A39583EDBBB05)
                * Padding
                    * CNN은 하나의 필터 레이어가 아니라 여러 단계에 걸쳐서 계속 필터를 연속적으로 적용하여 특징을 추출하며 최적화 해나가는데 필터를 적용할때마다 이미지는 작아지고 특징이 많이 유실되는 부분을 고려하기 위해서 사용된다.
                    * 원본이미지 주변에 원하는 padding 값을 추가하면 테두리를 만들 수 있게 되고 그후 필터를 진행하는 방식으로 이루어진다.
                    * padding을 통해서 overfitting도 방지하게 되며 원본이미지 특징 유실 문제 또한 해결이 가능하다.
            * 필터를 만드는 방법 : 필터는 데이터를 넣고 학습을 시키면 자동으로 학습 데이터에서 학습을 통해서 특징을 인식하고 필터를 만들어 낸다.
        * Activation Function 
            * Activation function은 필터를 통해서 추출된 값이 들어가 있는 값을 정량적인 값으로 나오기 때문에 그특징이 있다 / 없다의 비선형 값으로 바꿔 주는 과정을 해준다.
            * 필터를 통해서 Feature map이 추출되면 이 Feature map에 Activation function을 적용하게 된다.
            * 대표적으로 ReLU를 CNN에서 자주 사용하는데 대표적으로 Activation function에서 sigmoid라는 함수 설명해주는데 이 함수를 사용하지 않는이유 ?
                * Back Propagation(전체 레이어를 한번 계산한후 그 계산 값을 재 활용하여 다시 계산하는 알고리즘)을 사용하는데 sigmoid에서 사용하게 되면 뒤에서 앞으로 전달될때 값이 희석된다고 한다.
        * pooing(Sub Sampling or Pooling)
            * Convolution layer를 거쳐서 추출된 특징들은 필요에 따라서 서브 샘플링이라는 과정을 거친다.
            * 모든 특징을 가지고 판단할 필요 없기 때문에 사용된다고 한다.
        * Max pooling
            * Activatoin map을 MxN의 크기로 잘라낸후 그안에서 가장 큰값을 뽑아내는 방법이다.
            * 특징의 값이 큰값이 다른 특징들을 대표한다는 개념을 기반으로 하고 있다.
    * 전반적인 정리
        ![전반적인 Convoultion Nenural Network](https://t1.daumcdn.net/cfile/tistory/254DF041583ED6AF34)    
        * 위 그림과 같이 Convolution filter 와 Activation function(ReLU) 그리고 Pooling Layer를 반복적으로 조합하여 특징을 추출한다.
    * Fully connected Layer
        * Convolution Layer에서 특징이 추출이 되었으면 이 추출된 특징 값이 기존의 뉴럴 네트워크에 넣어서 분류 한다.
        * Softmax 함수
            * Softamx도 Activation function의 일종이며 sigmoid는 이산 분류 함수라면 softmax는 여러개의 분류를 가질수 있는 함수이다
            * 전반적인 정리에서 본 결과에서 우측에 여러 분류들이 보이는데 그중 car에 준하는 모습을 softmax로 확률을 환산 할 수 있다.
    * Dropout계층 
        * 드롭아웃은 overfitting을 막기 위한 방법으로 뉴럴 네트워크가 학습중일때 랜덤하게 뉴런을 꺼서 학습을 방해함으로써 학습이 학습용 데이터에 치우치는 현상을 막아준다.

## 딥러닝 알고리즘
* 딥러닝에서 사용되느 여러 유형의 Convolution 소개
    * Convolutions
        * Convolutional layer을 정의하기 위한 몇개의 파리미터를 알아야 한다.   
            [파란색이 input이며 초록색이 output입니다](https://cdn-images-1.medium.com/max/1200/1*1okwhewf5KCtIPaFib4XaA.gif)
            * kernel Size : kernel size는 convolution의 시야(view)를 결정합니다. 보통 2D에서는 3x3 pixel로 사용한다.
            * Stride(걸음) : Stride는 이미지를 횡단할 때 커널의 스템사이즈를 결정합니다.기본값은 1이지만 보통 Max Pooling과 비슷하게 이미지를 다운샘플링하기 위해 stride를 2로 사용할 수 있다.
            * Padding : Padding은 샘플 테두리를 어떻게 조절할지를 결정한다. 패딩된 Convolution은 input과 동일한 output차원을 유지하는 반면 패딩되지 않은 Convolution은 커널은 1보다 큰 경우 테두리의 일부를 잘라버릴 수 있다.
            * input & output Channels: Convoltion layer는 input 채널의 특정 수를 받아 output채널의 특정 수로 계산합니다. 이런 계층에서 필요한 파라미터의 수는 i * O * K로 계산할 수 있다.K는 커널의 수 입니다.
    * Transposed Convolutions 
        * Transposed Convolution은 deconvolutional layer와 동일한 공간 해상도를 생성하기 점은 유사하지만 실제 수행되는 수학 연산은 다르다. Transposed Convolution layer는 정기적인 convolution을 수행하며 공간의 변화를 되돌립니다.
        * 어떤 곳에선 deconvolution이라는 이름을 사용하지만 실제론 deconvolution이 아니기 떄문에 부적절합니다.상황을 악화 시키기 위해 존재하지만 딥러닝 분야에선 흔하지 않는다.실제 deconvolution은 convolution의 과정을 되돌립니다.하나의 convolutional layer에 이미지를 입력한다고 상상하면 이제 출력물을 가져와 블랙박스에 넣으면 원본 이미지가 다시 나타납니다.이럴 경우 블랙 박스가 deconvolution을 수행한다고 할 수 있다.이 deconvolution이 convolution layer가 수행하는 것의 수학적 역 연산이다.
## TODO
* pytorch Gan  : https://github.com/eriklindernoren/PyTorch-GAN#infogan
* https://github.com/mafda/generative_adversarial_networks_101
* NVIDA  inpainting image : https://github.com/NVIDIA/partialconv     
* 목표 : 이미지 복원 >> image inpainting
1. 원하는 데이터 사용하는방법!!
2. 다양한 GAN 알고리즘 접하기
3. 

* 목표 : 얼굴 인식하여 해당 인식한 얼굴 판별
    * python face_cognition 사용하여 구현
    * 학습 모델을 함수로 호출하여 얼굴 인식
    * 밑에 있는 URL은 compare함에 있어 부족함을 채워줄것이다.
    * https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems    

* datatset 설명 mnist vs imageNet vs CoCo vs CIFAR
    * https://page-box.tistory.com/5    

* 2d image conver 3D model
    * https://www.youtube.com/watch?v=qCVnPlr7eSY
    * colab
        * https://colab.research.google.com/drive/11z58bl3meSzo6kFqkahMa35G5jmh2Wgt#scrollTo=eclLG4xlJRIE
    * 3dmodeling
        * https://github.com/timzhang642/3D-Machine-Learning
        * https://github.com/natowi/3D-Reconstruction-with-Deep-Learning-Methods     

* 이미지 복원 
    * https://jayhey.github.io/deep%20learning/2018/01/05/image_completion/
    * 유튜브 및 코드 지원 
        * https://www.youtube.com/channel/UCpujNlw4SUpgTU5rrDXH0Jw/videos

* 현재 사용해본 GAN 관련 알고리즘
    ![GAN 에서 파생된 다양한 알고리즘](https://user-images.githubusercontent.com/37301677/94356821-94392200-00cd-11eb-8f57-5a2c2b18a2a1.png)     
    * [3DGAN] 3DGAN - pytorch 응용 X  , 예제 X , 알고리즘만
    * [Deep Convolution]DCGAN - pytorch 응용 O  , 예제 O
    * [Context Conditional]CCGAN - pytorch 응용 O  , 예제 O
    * StyleGAN - pytorch 응용 X  , 예제 O
    * Context Encoder - pytorch 응용 O  , 예제 O
