import random as rd
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline
import boto3
import requests
import subprocess
from flask import Flask, request
from typing import Any
import sys
import time
import openai
import datetime
import re
import numpy as np

import parmap
import multiprocessing
from multiprocessing import freeze_support

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import settings


# AWS 계정 정보 및 S3 버킷 정보 설정
aws_access_key_id = 'AKIAUKTRTP4VWEX5UKMW'
aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
bucket_name = 'seocho-voicetest'

from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig

# S3 클라이언트 생성
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# openai api key
api_key = settings.OPENAI_KEY
openai.api_key = api_key

# stt model
# stt_model_path = "haseong8012/whisper-large-v2_child10K_LoRA"
# stt_processor = WhisperProcessor.from_pretrained("./whisper-model")
# stt_model = WhisperForConditionalGeneration.from_pretrained("./whisper-model")    # model load 실패
peft_model_id = "./whisper-model/whisper-largeV2_child10K_LoRA" # Use the same model ID as before.
language = "ko"
task = "transcribe"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
stt_model = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)


# stable diffusion models
# models = [stabilityai/stable-diffusion-xl-base-1.0: (그림 퀄 좋음, 1장당 42초 -> 18초, inference step 50 -> 20: 7초),
#           "stabilityai/stable-diffusion-xl-base-0.9": (그림 퀄 1.0이랑 비슷, 1장당 17초),
#           "prompthero/openjourney": 그림퀄 똥, 1장당 9초,
#           "CompVis/stable-diffusion-v1-4": 그림퀄 개똥, 장당 7초,
#           "stabilityai/stable-diffusion-xl-1.0 + refiner": 그림퀄 젤 좋음, 장당 7+2=9초 
# ]
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
sd_model = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
sd_model = sd_model.to("cuda")
generator = torch.Generator("cuda").manual_seed(42)
sd_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_model.scheduler.config) #  PNDMScheduler -> DPMSolverMultistepScheduler로 변경, 이 스케줄러는 성능이 더 뛰어나 적은 inference step을 필요로 함

sd_refiner = pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                            torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
sd_refiner.to("cuda")
sd_refiner.scheduler = DPMSolverMultistepScheduler.from_config(sd_refiner.scheduler.config)
# sd_model.enable_xformers_memory_efficient_attention() # pytorch==1.13 or 2.0> 만 지원...
# sd_model.unet = torch.compile(sd_model.unet, mode="reduce-overhead", fullgraph=True) # Windows에서 불가능, Linux만 지원, pytorch>2.0부터 지원....


# s3에서 webm 데이터 받아온 후 wav 파일로 변환
def get_webm(s3_audio_url, webm_path):
    try:
        response = requests.get(s3_audio_url)
        if response.status_code == 200:
            with open(webm_path, 'wb') as local_file:
                local_file.write(response.content)
            print(f"오디오 파일이 다운로드되었습니다. 로컬 파일 경로: {webm_path}")
        else:
            print(f"다운로드 중 오류가 발생했습니다. 응답 코드: {response.status_code}")
    except Exception as e:
        print(f"다운로드 중 오류가 발생했습니다: {str(e)}")   
        
def webm_to_wav(filename, wav_path):
    ffmpeg_command = [
    'ffmpeg',  # ffmpeg 실행 명령어
    '-i', filename,  # 입력 파일 경로
    '-vn',  # 비디오 스트림 무시
    '-acodec', 'pcm_s16le',  # 오디오 코덱 설정 (16-bit PCM)
    '-ar', '16000',  # 샘플링 레이트 설정 (예: 44100 Hz)
    '-ac', '1',  # 채널 수 설정 (2 채널 = 스테레오)
    wav_path  # 출력 파일 경로
    ]
    subprocess.run(ffmpeg_command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        

# 음성 인식
# def stt(audio_file):

#     arr, sampling_rate = librosa.load(audio_file, sr=16000)

#     input_features = stt_processor(arr, return_tensors="pt", sampling_rate=sampling_rate).input_features

#     forced_decoder_ids = stt_processor.get_decoder_prompt_ids(language="ko", task="transcribe")
#     predicted_ids = stt_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
#     transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)
#     print(transcription)
#     return transcription
    
def stt(audio):
    with torch.cuda.amp.autocast():
        text = stt_model(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
    print(text)
    return text


# Prompt Engineering
def load_template(type):
    # UnicodeDecodeError로 encoding='utf-8'추가
    with open('prompt/prompt_template'+type+'.txt', 'r', encoding='utf-8') as f:
        return f.read()
    
def make_gpt_format(query: str = None):
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    return message


def gpt3(
        main_char: str = None,
        subject: str = None,
        story: str = None,
        model_name: str = "gpt-4",
        max_len: int = 4000,
        temp: float = 1.0,
        n: int = 1,
        freq_penalty: float = 0.0,
        pres_penalty: float = 0.0,
        top_p: float = 1.0,
        type: str = '1',
    ):
    if not story and not (main_char and subject):
        raise ValueError("There is no query sentence!")
    prompt_template = load_template(type)
    
    prompt = prompt_template.format(main_char=main_char, subject=subject, story=story)
    
    messages = make_gpt_format(prompt)
    # print(messages)
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=max_len,
                temperature=temp,
                n=n,
                presence_penalty=pres_penalty,
                frequency_penalty=freq_penalty,
                top_p=top_p,
            )
            received = True
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g., prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{messages}\n\n")
                assert False
            time.sleep(2)
    resp = response.choices[0].message["content"]
    
    print(resp)
    return resp

# Stable Diffusion
def text_to_image(text):

    # prompt = "a photograph of an astronaut riding a horse"
    # generator: seed값 고정, num_inference_steps: 기존 50-> 20 변경
    image = sd_model(text, generator=generator, num_inference_steps=20).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
    # refine
    image = sd_refiner(prompt=text, image=image, generator=generator, num_inference_steps=20).images[0]

    #이미지 저장
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S')
    image_file_path = 'image/image'+formatted_time+'.jpg'
    image.save(image_file_path)

    
    try:
        s3.upload_file(image_file_path, bucket_name, image_file_path)
        print(f"{image_file_path} 파일이 {bucket_name} 버킷에 성공적으로 업로드되었습니다.")
    except Exception as e:
        print(f"업로드 중 오류가 발생했습니다: {str(e)}")
    
    url = 'https://seocho-voicetest.s3.ap-northeast-2.amazonaws.com/'+image_file_path

    #이미지 출력
    return url

def get_sentence(sentences):
    sentence_list = []

    # 페이지별로 문장 추출
    pages = sentences.split('\n')
    for page in pages:
        if page.strip():  # 빈 줄은 무시
            # 페이지에서 문장만 추출
            sentence = page.split(': ')[1]
            sentence_list.append(sentence)
    
    return sentence_list


def get_title(title):
    # 정규 표현식 패턴
    pattern = r'제목:\s+"([^"]+)"'

    # 정규 표현식을 사용하여 제목 추출
    match = re.search(pattern, title)  
    
    if match:
        title = match.group(1)
        print(f"제목: {title}")
    else:
        print("제목을 찾을 수 없습니다.")
        
    return title


def speech_to_text(s3_audio_url):
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S')
    
    # # WebM 파일을 저장할 경로
    # webm_file_path = 'audio/audio'+formatted_time+'.webm'

    # # S3 URL에서 오디오 파일 다운로드
    # get_webm(s3_audio_url, webm_file_path)   

    # # wav 파일을 저장할 경로
    # wav_file_path = 'audio/audio'+formatted_time+'.wav'
    
    # # webm 파일을 wav 파일로 변환
    # webm_to_wav(webm_file_path, wav_file_path)
    wav_file_path = 'audio/test_audio.wav'
    
    # speech to text
    return stt(wav_file_path)   
 
def generate_book(main_char, subject):
    
    # 페이지 수 설정
    num = 8
    
    # 동화책 내용 생성
    story = gpt3(main_char=main_char, subject=subject, type='4')
    
    # 결과에서 문장 추출
    page_list = get_sentence(story) 
    
    # 동화책 제목 생성
    title = gpt3(story=story, type='3')
    
    # 결과에서 제목 추출
    title = get_title(title)
    
    # 이미지 생성용 프롬프트 생성
    prompt_result = gpt3(story=story, type='5')
    
    # 결과에서 프롬프트 추출
    prompt_list = get_sentence(prompt_result)
    
    # 랜덤으로 주고 싶은 prompt 값
    color = ['painted in bright water colors']
    medium = ["Sebastian, children's book illustrator, Whimsical and colorful style, Medium: Watercolor, Color Scheme: Bright and lively color palette",'cinematic,character design by mark ryden and pixar and hayao miyazaki, unreal 5,',"unreal engine, cozy indoor lighting, artstation, detailed, digital painting,cinematic,character design by mark ryden and pixar and hayao miyazaki, unreal 5, daz, hyperrealistic, octane render Negative prompt: ugly, ugly arms, ugly hands,",'colored pencil', 'oil pastel', 'acrylic painting', 'a color pencil sketch inspired by Edwin Landseer', 'large pastel, a color pencil sketch inspired by Edwin Landseer', 'a storybook illustration by Marten Post', 'naive art', 'cute storybook illustration', 'a storybook illustration by tim burton']
    setting = ","+ medium[rd.randint(0, len(medium)-1)] +", (Lighting) in warm sunlight, (Artist) in a cute and adorable style, cute storybook illustration, (Medium) digital art, (Color Scheme) vibrant and cheerful, "
    
    # 동화책 그림 생성
    img_list = []
    for i in range(num):
        print(prompt_list[i]+setting)
        img_list.append(text_to_image(prompt_list[i]+setting))
    
    # s3 image url 반환
    response_data = {
        'image_story_pairs': [],
    }
    
    # 이미지와 문장 쌍을 response_data에 추가
    for i in range(len(img_list)):
        image_story_pair = {
            'image': img_list[i],
            'story': page_list[i]
        }
        response_data['image_story_pairs'].append(image_story_pair)
    
    return response_data


main_char = ""
subject = ""
    
def main(s3_audio_url):
    global main_char, subject
    main_char = "공주 메리다"
    subject = speech_to_text(s3_audio_url)
    
    print(subject[0])
    
    generate_book(main_char, "전쟁")
    

if __name__ == "__main__":
    main("https://seocho-voicetest.s3.ap-northeast-2.amazonaws.com/audio-1696055947410.webm")