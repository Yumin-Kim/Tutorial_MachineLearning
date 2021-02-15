# Pytorch GAN
* 해당 링크를 통해서 여러 GAN관련 알고리즘의 예제를 확인할 수 있다.
* https://github.com/eriklindernoren/PyTorch-GAN#infogan 
# 파이토치(Pytorch)  DCGAN
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
# Context Encoder
* 해당 사이트를 통해서 정리하게 되었다. https://labcontext.github.io/restoration/contextEncoder/
* context기반의 픽셀 예측을 통해 비지도 학습된 이미지 학습 알고리즘이다.
* Auto encoder와 유사한 상태이며 이는 주변 환경에 조건화된 임의 영상 영역의 콘텐츠를 생성 하도록 훈련된 복잡한 신경 네트워크이다. 
    * Auto encoder는 input image를 이용하여 장면의 compact한 표현을 얻기 위한 목적으로 low dimensional bottle 레이어를 통과한 후 재구성 한다.
* 누락된 부분을 채우는 작업은 전체 이미지의 내용을 이해하고 누락된 부분에 대해서 그럴듯한 가설을 만들어야한다. 이를 통해 context encoder를 train할때 표준 픽셀 단위 재구성 손실과 재구성 및 상대적 손실을 모두 실험 했다고 한다.
* Context encoder는 외형뿐만이 아닌 시각 구조의 의미도 학습한다는 것도 발견했으며 semantic inpating 작업을 할 수 있다는것을 발견 >> 해당 문제를 CNN를 이용해서 해결
* 
        
