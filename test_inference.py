from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, XLMRobertaTokenizer
import requests 
from PIL import Image

def prepare_img(model_name):
    if "handwritten" in model_name:
        url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"  # industry
    elif "printed" in model_name or "stage1" in model_name:
        url = "https://www.researchgate.net/profile/Dinh-Sang/publication/338099565/figure/fig8/AS:840413229350922@1577381536857/An-receipt-example-in-the-SROIE-2019-dataset_Q640.jpg"
    im = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return im

if __name__ == '__main__':
    model_name = 'trocr-small-stage1'
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)    
    # model.config.eos_token_id = 2

    # load image from the IAM dataset url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg" 
    # image = prepare_img(model_name)
    image = Image.open("test.jpg")

    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values 
    generated_ids = model.generate(pixel_values)

    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] 
    print(generated_text)