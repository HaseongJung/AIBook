import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diffusers import StableDiffusionPipeline
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

import settings
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# openai api key
api_key = settings.OPENAI_KEY
openai.api_key = api_key

# Prompt Engineering
def load_template(type):
    # UnicodeDecodeError로 encoding='utf-8'추가
    with open('prompt_template'+type+'.txt', 'r', encoding='utf-8') as f:
        return f.read()
    
def make_gpt_format(query: str = None):
    message = [
        {"role": "system", "content": "You are a creative children's storybook writer."},
        {"role": "user", "content": query}
    ]
    print
    return message

def gpt3(
        query: str = None,
        model_name: str = "gpt-4",
        max_len: int = 2000,
        temp: float = 1.0,
        n: int = 1,
        freq_penalty: float = 0.0,
        pres_penalty: float = 0.0,
        top_p: float = 1.0,
        type: str = '1',
    ):
    
    if not query:
        raise ValueError("There is no query sentence!")
    prompt_template = load_template(type)
    
    prompt = prompt_template.format(transcribed_text=query)
    
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

    
def main():
    
    text = '강아지'
    
    # 페이지 수 설정
    num = 3
    # 동화책 내용 생성
    story = gpt3(query=text, type='4')
    
    # 페이지별로 문장 추출
    page_sentences = []
    pages = story.split('\n')
    for page in pages:
        if page.strip():  # 빈 줄은 무시
            # 페이지에서 문장만 추출
            sentence = page.split(': ')[1]
            page_sentences.append(sentence)
    
    print(page_sentences)
    
    title = gpt3(query=story, type='3')
    
    # 정규 표현식 패턴
    pattern = r'제목:\s+"([^"]+)"'

    # 정규 표현식을 사용하여 제목 추출
    match = re.search(pattern, title)

    if match:
        title = match.group(1)
        print(f"제목: {title}")
    else:
        print("제목을 찾을 수 없습니다.")
    
    prompt = gpt3(query=story, type='5')
    
    
    #sentences = story.split(':')
    #print(sentences)

    #prompt = gpt3(query= '"오늘은 어디로 산책을 가볼까?" 멍멍이는 지도를 검색하고 있어요.', type='2')
    # stable diffusion에 넣을 prompt

    '''
    prompt = []
    for i in range(num):
        print(i)
        prompt.append(gpt3(query=sentences[i], type='4'))
    
    print(sentences)
    print(prompt)
    '''
    return 0


if __name__ == "__main__":
    main()