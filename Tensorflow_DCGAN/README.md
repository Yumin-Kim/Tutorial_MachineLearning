# # 3DGAN을

1. OOO 3D GAN에 대한 학습/샘플 실행 (4개 정도)
   https://github.com/jpjuvo/64-3D-RaSGAN (3D GAN, 3D IWGAN, RSGAN, RaSGAN)

2. 2D Image to 3D Point Cloud 추출(연구사례가 아주 많기 때문에)

3. #2 -> #3 연결

4. 128^3의 Object(Tower/Pagoda) Point Cloud를 기반으로 GAN -> 논문 주제

5. High-Resolution GAN (2D -> 2D High Resolution), (3D -> 3D High Resolution)

6. #5 (X,Y), (Y,Z), (Z,X) Pair -> (X',Y').... 3개의 쌍에 대한 GAN (Slide 형식(Z축을 기준으로 나누고, Y축을 기준으로 나누고, X축을 기준으로))
   (32 by 32) => (256 by 256)으로 돌려보는것 -> 논문 주제

7. 버그수정, New Algorithms 적용. (3D Style GAN)

---

# TODO 리스트 또는 예제를 통해서 모르는 용어 정리

- SGAN[ Semi-supervised Generaltive Adversarial Networks ]
  - https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=221274014481&proxyReferer=https:%2F%2Fwww.google.com%2F
- RaSGAN
- ModelNet

---

## 3D GAN 관련 상식 정리 및 포스팅

- https://m.blog.naver.com/laonple/221201915691
-
