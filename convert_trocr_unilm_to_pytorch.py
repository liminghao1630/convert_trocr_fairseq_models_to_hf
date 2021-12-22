# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert TrOCR checkpoints from the unilm repository."""


import argparse
from math import exp
from pathlib import Path

import torch
from PIL import Image

import requests
from transformers import (
    RobertaTokenizer,
    TrOCRConfig,
    TrOCRForCausalLM,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTFeatureExtractor,
    ViTModel,
    DeiTModel,
    DeiTFeatureExtractor,
    XLMRobertaTokenizer
)
from transformers.utils import logging
from fairseq import file_utils


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(encoder_config, decoder_config, has_distill_token=False):
    rename_keys = []
    for i in range(encoder_config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm1.weight", f"encoder.encoder.layer.{i}.layernorm_before.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm1.bias", f"encoder.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.weight", f"encoder.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.bias", f"encoder.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm2.weight", f"encoder.encoder.layer.{i}.layernorm_after.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm2.bias", f"encoder.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.weight", f"encoder.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.bias", f"encoder.encoder.layer.{i}.intermediate.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc2.weight", f"encoder.encoder.layer.{i}.output.dense.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.mlp.fc2.bias", f"encoder.encoder.layer.{i}.output.dense.bias"))

    # cls token, position embeddings and patch embeddings of encoder
    rename_keys.extend(
        [
            ("encoder.deit.cls_token", "encoder.embeddings.cls_token"),
            ("encoder.deit.pos_embed", "encoder.embeddings.position_embeddings"),
            ("encoder.deit.patch_embed.proj.weight", "encoder.embeddings.patch_embeddings.projection.weight"),
            ("encoder.deit.patch_embed.proj.bias", "encoder.embeddings.patch_embeddings.projection.bias"),
            ("encoder.deit.norm.weight", "encoder.layernorm.weight"),
            ("encoder.deit.norm.bias", "encoder.layernorm.bias"),
        ]
    )

    if has_distill_token:
        rename_keys.append(("encoder.deit.dist_token", "encoder.embeddings.distillation_token"))

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, encoder_config, has_bias=False):
    for i in range(encoder_config.num_hidden_layers):
        # queries, keys and values 
        in_proj_weight = state_dict.pop(f"encoder.deit.blocks.{i}.attn.qkv.weight")

        state_dict[f"encoder.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : encoder_config.hidden_size, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            encoder_config.hidden_size : encoder_config.hidden_size * 2, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -encoder_config.hidden_size :, :
        ]

        if has_bias:
            in_proj_bias = state_dict.pop(f"encoder.deit.blocks.{i}.attn.qkv.bias")
            state_dict[f"encoder.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: encoder_config.hidden_size]
            state_dict[f"encoder.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
                encoder_config.hidden_size : encoder_config.hidden_size * 2
            ]
            state_dict[f"encoder.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[
                -encoder_config.hidden_size :
            ]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of the IAM Handwriting Database
def prepare_img(checkpoint_url):
    if "handwritten" in checkpoint_url:
        url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"  # industry
        # [0.8980, 0.9059, 0.9137, 0.9294, 0.9373, 0.9373, 0.9373, 0.9373, 0.9294, 0.9216] [0, 0, 0, :10]
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-12.jpg" # have
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-10.jpg" # let
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"  #
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122.jpg"
    elif "printed" in checkpoint_url or "stage1" in checkpoint_url:
        url = "https://www.researchgate.net/profile/Dinh-Sang/publication/338099565/figure/fig8/AS:840413229350922@1577381536857/An-receipt-example-in-the-SROIE-2019-dataset_Q640.jpg"
    im = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return im


@torch.no_grad()
def convert_tr_ocr_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our VisionEncoderDecoderModel structure.
    """
    # define encoder and decoder configs based on checkpoint_url
    encoder_config = ViTConfig(image_size=384, qkv_bias="small" in checkpoint_url)
    decoder_config = TrOCRConfig()

    # size of the architecture
    if "base" in checkpoint_url:
        decoder_config.cross_attention_hidden_size = 768
    elif "large" in checkpoint_url:
        # use ViT-large encoder
        encoder_config.hidden_size = 1024
        encoder_config.intermediate_size = 4096
        encoder_config.num_hidden_layers = 24
        encoder_config.num_attention_heads = 16

        decoder_config.cross_attention_hidden_size = 1024
    elif "small" in checkpoint_url:
        encoder_config.hidden_size = 384
        encoder_config.intermediate_size = encoder_config.hidden_size * 4
        encoder_config.num_hidden_layers = 12
        encoder_config.num_attention_heads = 6

        decoder_config.vocab_size = 64044
        decoder_config.cross_attention_hidden_size = 384
        decoder_config.d_model = 256
        decoder_config.num_hidden_layers = 6
        decoder_config.num_attention_heads = 8
        decoder_config.decoder_ffn_dim = 1024
        decoder_config.tie_word_embeddings = False
        decoder_config.scale_embedding = True
        decoder_config.activation_function = "relu"
    else:
        raise ValueError("Should either find 'base' or 'large' or 'small' in checkpoint URL")

    # the large-printed + stage1 checkpoints uses sinusoidal position embeddings, no layernorm afterwards
    # the small models are exceptions, they use layernorm after the encoder for all settings, i.e. trocr-small-handwritten, trocr-small-printed, trocr-small-stage1
    if "large-printed" in checkpoint_url or ("stage1" in checkpoint_url and "small" not in checkpoint_url):
        decoder_config.tie_word_embeddings = False
        decoder_config.activation_function = "relu"
        decoder_config.max_position_embeddings = 1024
        decoder_config.scale_embedding = True
        decoder_config.use_learned_position_embeddings = False
        decoder_config.layernorm_embedding = False        

    # load HuggingFace model
    if "small" in checkpoint_url:
        encoder = DeiTModel(encoder_config, add_pooling_layer=False)
    else:
        encoder = ViTModel(encoder_config, add_pooling_layer=False)
    decoder = TrOCRForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    # load state_dict of original model, rename some keys
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)["model"]

    rename_keys = create_rename_keys(encoder_config, decoder_config, has_distill_token="small" in checkpoint_url)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, encoder_config, has_bias="small" in checkpoint_url)

    # remove parameters we don't need
    del state_dict["encoder.deit.head.weight"]
    del state_dict["encoder.deit.head.bias"]
    del state_dict["decoder.version"]
    if "small" in checkpoint_url:
        del state_dict["encoder.deit.head_dist.weight"]
        del state_dict["encoder.deit.head_dist.bias"]

    # add prefix to decoder keys
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith("decoder") and "output_projection" not in key:
            state_dict["decoder.model." + key] = val
        else:
            state_dict[key] = val

    # load state dict
    model.load_state_dict(state_dict)  # 神奇的是decoder.output_projection.weight在load的时候会覆盖掉decoder.model.decoder.embed_tokens.weight的值
    # model.state_dict()['decoder.model.decoder.embed_tokens.weight'].copy_(state_dict['decoder.model.decoder.embed_tokens.weight'])
    # model.state_dict()['decoder.output_projection.weight'].copy_(state_dict['decoder.output_projection.weight'])

    # double check
    for key in model.state_dict():
        assert model.state_dict()[key].shape == state_dict[key].shape, key + " shape mismatch"
        assert torch.allclose(model.state_dict()[key], state_dict[key]), key + " value mismatch"

    # Check outputs on an image
    if "small" in checkpoint_url:
        feature_extractor = DeiTFeatureExtractor(size=encoder_config.image_size, do_center_crop=False, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        vacob = file_utils.cached_path('https://layoutlm.blob.core.windows.net/trocr/dictionaries/unilm3-cased.model')
        tokenizer = XLMRobertaTokenizer(vacob)

        processor = feature_extractor        
    else:        
        feature_extractor = ViTFeatureExtractor(size=encoder_config.image_size)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        processor = TrOCRProcessor(feature_extractor, tokenizer)
    
    pixel_values = processor(images=prepare_img(checkpoint_url), return_tensors="pt").pixel_values

    # verify logits
    decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])
    outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits

    if "small" in checkpoint_url:
        expected_shape = torch.Size([1, 1, 64044])
        if "trocr-small-handwritten" in checkpoint_url:
            expected_slice = torch.tensor(
                [-5.1575, -4.8786,  4.4805,  4.0093,  5.9993,  7.6883,  8.6437, 11.3955, 16.2947,  7.3855]
            )
        elif "trocr-small-printed" in checkpoint_url:
            expected_slice = torch.tensor(
                [-6.6595, -7.1750,  4.7468, -6.0797,  3.7086, -1.6098, -3.5780,  2.2483, 1.7086,  2.6549]
            )
        elif "trocr-small-stage1" in checkpoint_url:
            expected_slice = torch.tensor(
                [-8.7469, -8.4940,  2.6503, -2.7512,  4.8163,  1.9893,  1.3361,  4.2446, 3.6917,  3.7275]
            )
    else:
        expected_shape = torch.Size([1, 1, 50265])
        if "trocr-base-handwritten" in checkpoint_url:
            expected_slice = torch.tensor(
                [-1.4502, -4.6683, -0.5347, -2.9291, 9.1435, -3.0571, 8.9764, 1.7560, 8.7358, -1.5311]
            )
        elif "trocr-large-handwritten" in checkpoint_url:
            expected_slice = torch.tensor(
                [-2.6437, -1.3129, -2.2596, -5.3455, 6.3539, 1.7604, 5.4991, 1.4702, 5.6113, 2.0170]
            )
        elif "trocr-base-printed" in checkpoint_url:
            expected_slice = torch.tensor(
                [-5.6816, -5.8388, 1.1398, -6.9034, 6.8505, -2.4393, 1.2284, -1.0232, -1.9661, -3.9210]
            )
        elif "trocr-large-printed" in checkpoint_url:
            expected_slice = torch.tensor(
                [-6.0162, -7.0959, 4.4155, -5.1063, 7.0468, -3.1631, 2.6466, -0.3081, -0.8106, -1.7535]
            )

    if "stage1" not in checkpoint_url:
        assert logits.shape == expected_shape, "Shape of logits not as expected"
        assert torch.allclose(logits[0, 0, :10], expected_slice, atol=1e-3), "First elements of logits not as expected"

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving processor to {pytorch_dump_folder_path}")
    processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default='./', type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()

    args.checkpoint_url = 'https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-handwritten.pt'
    convert_tr_ocr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)