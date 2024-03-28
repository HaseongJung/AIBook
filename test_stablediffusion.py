import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diffusers import StableDiffusionPipeline

from flask import Flask, request
from typing import Any

import datetime

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# stable diffusion modelz
sd_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
sd_model.to("cuda")

# Stable Diffusion
def text_to_image(text):

    #prompt = "a photograph of an astronaut riding a horse"
    image = sd_model(text).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

    #이미지 저장
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S')
    image_file_path = 'image'+formatted_time+'.jpg'
    image.save(image_file_path)
    
    image
    
prompt = "Prompt: (Subject) Puppy, (Action) playing, (Context) in a lively park, (Environment) surrounded by trees and flowers, (Artist) in an cute style, (Style) reminiscent of a scene of a children's storybook, (Medium) digital art, (Type) illustration,Negative Prompts: extra limbs, ugly, pixelated, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, body out of frame, blurry"
text_to_image(prompt)