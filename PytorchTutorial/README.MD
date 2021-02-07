# 파이토치(Pytorch)  DCGAN튜토리얼!! 정리
* 목표
    * 많은 실제 유명인의 사진을 보여준후 새로운 유명인의 사진을 생성할 수 있도록 적대적 생성 신경망(GAN)을 훈련시키는 것이 목표이다
* GAN 이란 무엇일까??
    * GAN은 trinsing 테이터의 분포를 알아내어 같은 분포로부터의 새로운 데이터를 생성할 수 있는 DL모델
    * GAN은 Generator(생성자)와 Discriminator(판별자)라는 두개의 구분되는 모델로 구성되어 있다.
    * Generaotor가 하는일은 training데이터와 유사한 "가짜" 이미지를 만들어 내는것이다.
    * Discriminaotor가 하는일은 이미지를 본뒤 실제 training 데이터인지 genrator로 부터 생성된 가짜이미지인지 출력하는것이다.
    * 트레이닝 동안 generator는 끊임없이 더 나은 가짜 이미지를 생성해 discriminator를 능가하려는 노력하는 반면에 discriminator는 진짜 이미지 가짜 이미지를 더 잘감지하고 분류하기 위해서 노력한다.
    * generator가 training 데이터에서 꺼내온 듯한 꺼내온듯한 완벽한 가짜 이미지를 만들어내고 있을때이고 discriminator는 항상 generator의 결과가 진짜인지 가짜인지 50%인 신뢰도를 가지고 추측한다.
* DCGAN 이란 무엇일까?
    * DCGAN은 위에서 설명한 GAN의 확장으로 generator와 discriminator에서 각각 Convultion레이엉와 Convolution-transpose레이어를 사용한다는점에서 차이가 있다,

* Input값 정의
    * dataroot - 데이터세트 폴더의 루트 경로
    * workers - Dataloader를 이용해 데이터를 로드하기 위해 사용되는 스레드 수
    * batch_size - 학습에서 사용되는 배치 크기. 논문에서는 128. 
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
            [https://mblogthumb-phinf.pstatic.net/MjAxOTAxMjNfMjU4/MDAxNTQ4MjM1Nzg3NTA2.UtvnGsckZhLHOPPOBWH841IWsZFzNcgwZvYKi2nxImEg.CdtqIxOjWeBo4eNBD2pXu5uwYGa3ZVUr8WZvtldArtYg.PNG.qbxlvnf11/20190123_182720.png?type=w800](예시)  
            