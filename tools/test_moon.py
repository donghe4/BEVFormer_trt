from transformers import AutoImageProcessor, NatForImageClassification
import torch

import sys
import os

# HuggingFace의 'datasets' 라이브러리가 설치된 경로를 찾아야 합니다.
# 이 경로는 환경에 따라 다를 수 있습니다.
# 예를 들어, 아래는 anaconda 환경에서 'datasets' 라이브러리가 설치된 경로의 예입니다.
huggingface_datasets_path = '/usr/local/lib/python3.8/dist-packages/datasets'

# 해당 경로가 sys.path에 존재하는지 확인합니다.
if huggingface_datasets_path not in sys.path:
    # 경로를 sys.path의 맨 앞에 추가합니다.
    # 이렇게 하면 Python은 HuggingFace의 'datasets'를 먼저 찾게 됩니다.
    sys.path.insert(0, huggingface_datasets_path)

from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
model = NatForImageClassification.from_pretrained("shi-labs/nat-mini-in1k-224")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
