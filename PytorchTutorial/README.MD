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
* 
        
