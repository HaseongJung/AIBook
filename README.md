# 아이북 프로젝트

## 📖 내용

- 사용자 설정에 따라 **개인화된 동화책**을 **생성**하는 인공지능 기반의 웹 서비스이다.
- 사용자의 **음성 인식(fine-tuned Whisper)**을 통해 주제를 설정, **GPT-4**를 통해 이야기를 생성, **Stable diffusion**을 통하여 이미지를 생성하여 최종적으로는 동화책을 제작하는 것을 목표로 한다.
- 서비스의 대상이 유아인 것 만큼, 발음이 부정확한 유아의 음성에 대한 인식률을 높이기 위하여 음성인식 모델인 Whisper를 fine-tuning한다.

![아이북 포스터](https://github.com/HaseongJung/AIBook/assets/107913513/5769cf84-64ea-464d-8c11-57e9393d7922)

## 🙋‍♂️ 역할

- Whisper모델을 fine-tuning할 때 사용할 음성 데이터를 전처리한다
    
    → `병렬처리를 통해 기존 소요시간의 1/10 이상 감소`
    
- Open AI의 Whisper모델을 fine-tunging하여 유아 음성의 인식률을 개선한다
    
    → `Cer(음절 오류율) 기준:  11.9 → 1.08`
    
- 전처리한 데이터셋과 fine-tuning한 모델들을 Hugging-face hub를 통해 관리한다.
    
    [haseong8012/whisper-large-v2_child10K_LoRA · Hugging Face](https://huggingface.co/haseong8012/whisper-large-v2_child10K_LoRA)
    
- Stable Diffusion에서의 생성하는 이미지의 품질을 향상시키고, 생성시간을 단축시킨다.
    
    `기존 이미지 1장 당 25초 → 12초`
    

---

## Dataset

AI Hub에서 제공하는 ‘명령어 음성(소아, 유아)’ 데이터를 사용한다.

데이터에는 각 음성의 오디오 파일(wave)과 전사정보를 포함한 파일(json)이 존재한다.

[AI-Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=95)

<br>

## Preprocessing

1. 데이터의 전사정보(json)에서 필요없는 괄호, 특수문자 및 노이즈 표기 등을 제거한다.
2. 데이터의 오디오 파일(wave)을 `48kHz→16kHz` resampling 및 Z-socre정규화를 진행한다.
3. 각 매칭되는 음성파일(wave)와 전사정보 파일(json)의 인덱스를 동일하게 하여, `audio` 와 `text` 두 개의 컬럼으로 이루어진 데이터셋을 구축한다.
4. 구축한 Dataset은 Hugging-face hub에 업로드하여 관리한다.

적게는 10,000개 많게는 2,070,000개로 이루어지고 용량이 큰 음성 데이터의 특성 상 엄청난 리소스와 소요시간이 발생한다.

→ multiprocessing&parmap을 이용한 병렬처리를 통해 프로세서를 최대로 사용하여 `기존 소요시간의 1/10 이상 단축`하였다.

<br>

## Whisper fine-tuning

![whisper성능](https://github.com/HaseongJung/AIBook/assets/107913513/2b744cd2-7dd5-4aef-841a-194bf289a827)

OpenAI의 whisper 모델 성능

![Whisper 모델 Size별 정보](https://github.com/HaseongJung/AIBook/assets/107913513/c802dc80-1ba8-4411-a5b0-d0627becd600)

Whisper 모델 Size별 정보

- 대량의 다양한 언어들로 사전 학습됨
    
    → 일반 남여의 음성에 대한 기본적인 성능 보장
    
- 주요 언어가 아닌 한국어에서는 비교적 낮은 인식도
    
    → fine-tuning시 성능이 향상된 연구를 통해 가능성 확인
    

![whisper 아키텍처](https://github.com/HaseongJung/AIBook/assets/107913513/c8cb4e4a-9e18-4752-b474-8e1c478db001)

Whisper 모델 아키텍처

![fine-tuning시 고려할 변수](https://github.com/HaseongJung/AIBook/assets/107913513/4f611314-335d-4735-838c-a4a99ec801ee)

fine-tuning시 고려할 변수

Whisper fine-tunging 가이드를 참고하여, Batch size, Learning rate 및 dropout 등과 같은 하이퍼 파라미터는 공식 문서 및 선행 연구자의 결과에서 도출한 조언에 따라 각 환경에 맞추어서 진행하였다. https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event#recommended-training-configurations

![Data augmentation 및 lr scheduler에 따른 성능 비교](https://github.com/HaseongJung/AIBook/assets/107913513/4d4a9764-2c0e-414b-a62c-209f83029536)

Data augmentation 및 lr scheduler에 따른 성능 비교

![Train dataset size에 따른 성능 비교](https://github.com/HaseongJung/AIBook/assets/107913513/babcdefd-8a5a-415d-ab37-5aba3d1e6bb5)

Train dataset size에 따른 성능 비교

![fine-tuned Whisper-small (child-50k, linear, adamW)](https://github.com/HaseongJung/AIBook/assets/107913513/2688d21c-2371-44b2-82c6-931655ea61a3)

fine-tuned Whisper-small (child-50k, linear, adamW)

데스크탑 사양 제약에 따라 Whisper tiny/small에 대한 test 및 성능 비교 진행하였다.

최종적으로 augmentaion을 진행하지 않고, linear lr scheduler, 10k의 데이터셋을 사용하여 fine-tuning하는 것이 성능이 가장 좋은 것으로 판단하였다.

### LoRA fine-tuning

tiny/small 모델을 비교하였을 때, 모델 사이즈에 따라 성능 차이가 큰 것을 확인 후, 제약 된 환경에서 Whisper-large모델을 fine-tuning할 수 있는 방법을 모색 중 `LoRA fine-tuning`기법을 발견하였다.

사전 학습된 대규모 언어 모델의 가중치 행렬을 구성하는 모든 가중치를 미세 조정하는 대신, 이 큰 행렬을 근사화하는 두 개의 작은 행렬을 미세 조정하는 개선된 미세 조정 방식이다.

![fine-tuned Whisper large-v2 (Best model)](https://github.com/HaseongJung/AIBook/assets/107913513/c11b5dc9-7366-4941-b977-8d3067046e8d)  

fine-tuned Whisper large-v2 (Best model)

이후 Hugging-face에서 제공하는 [PEFT](https://github.com/huggingface/peft)라이브러리를 통하여 한정된 GPU Ram에서 하지 못했던 Whisper-large-v2모델을 RoLA fine-tuning한 결과 child/general 음성 모두에서 최상의 성능을 뽑아내었다.

<br>

## Stable Diffusioin

![Stable Diffusion - user preference chart](https://github.com/HaseongJung/AIBook/assets/107913513/d96faf51-2dd8-4989-8803-94fd781d93fa)

Stable Diffusion - user preference chart

기존에 [Diffusers](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)라이브러리에서 제공하는 `Stable Diffusion-XL 1.0`모델을 사용하여 이미지를 생성하였을 때, `1장 당 약 25초`가 소요되었다.

→ 기존의 50추론단계를 필요로 하는 **[PNDMScheduler](https://huggingface.co/docs/diffusers/v0.26.3/en/api/schedulers/pndm#diffusers.PNDMScheduler)**를 20~25추론단계를 필요로 하는 **[DPMSolverMultistepScheduler](https://huggingface.co/docs/diffusers/v0.26.3/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler)**로 변경하여 성능을 유지한 채 `이미지 생성시간을 단축`하였다.

→ 기존 SDXL 1.0모델에 refiner모듈을 추가하여 `이미지의 품질을 향상`시켰다.

이미지 1장 당 총 소요시간: SDXL1.0 (약 6초) +refiner(약 6초) = 약 12초 

<br>

## 🎯 결과 및 성과

- 서초 AI칼리지 프로젝트 과정 우수팀 선정
- 「생성형 AI기반의 동화책 제작 서비스 설계 및 구현」, ****HCIK2024 ****
