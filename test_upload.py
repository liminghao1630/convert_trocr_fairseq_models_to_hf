from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoFeatureExtractor, XLMRobertaTokenizer
from PIL import Image
import requests
import torch

# load image from the IAM database
url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# For the time being, TrOCRProcessor does not support the small models, so the following temporary solution can be adopted
# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-stage1')
feature_extractor = AutoFeatureExtractor.from_pretrained('liminghao1630/trocr-small-stage1')
model = VisionEncoderDecoderModel.from_pretrained('liminghao1630/trocr-small-stage1')

# training
pixel_values = feature_extractor(image, return_tensors="pt").pixel_values  # Batch size 1
decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])
outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)