# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import abc

import seq_aligner
import torch.nn.functional as nnf

MAX_NUM_WORDS = 77

from diffusers.models.attention_processor import AttnProcessor, Attention

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


# @torch.no_grad()
# def text2image_ldm(
#     model,
#     prompt:  List[str],
#     controller,
#     num_inference_steps: int = 50,
#     guidance_scale: Optional[float] = 7.,
#     generator: Optional[torch.Generator] = None,
#     latent: Optional[torch.FloatTensor] = None,
# ):
#     register_attention_control(model, controller)
#     height = width = 256
#     batch_size = len(prompt)
    
#     uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
#     uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
#     text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
#     text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
#     latent, latents = init_latent(latent, model, height, width, generator, batch_size)
#     context = torch.cat([uncond_embeddings, text_embeddings])
    
#     model.scheduler.set_timesteps(num_inference_steps)
#     for t in tqdm(model.scheduler.timesteps):
#         latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
#     image = latent2image(model.vqvae, latents)
   
#     return image, latent


# @torch.no_grad()
# def text2image_ldm_stable(
#     model,
#     prompt: List[str],
#     controller,
#     num_inference_steps: int = 50,
#     guidance_scale: float = 7.5,
#     generator: Optional[torch.Generator] = None,
#     latent: Optional[torch.FloatTensor] = None,
#     low_resource: bool = False,
#     height: int = 512, # New parameter with default value
#     width: int = 512,  # New parameter with default value
# ):
#     register_attention_control(model, controller)
#     batch_size = len(prompt)

#     text_input = model.tokenizer(
#         prompt,
#         padding="max_length",
#         max_length=model.tokenizer.model_max_length,
#         truncation=True,
#         return_tensors="pt",
#     )
#     text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
#     max_length = text_input.input_ids.shape[-1]
#     uncond_input = model.tokenizer(
#         [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
#     )
#     uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
#     context = [uncond_embeddings, text_embeddings]
#     if not low_resource:
#         context = torch.cat(context)
        
#     # Use the new height and width parameters
#     latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
#     model.scheduler.set_timesteps(num_inference_steps)
#     for t in tqdm(model.scheduler.timesteps):
#         latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
#     image = latent2image(model.vae, latents)
  
#     return image, latent

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    height: int = 512, # Still has a default for generating from noise
    width: int = 512   # Still has a default for generating from noise
):
    # --- NEW AUTOMATIC DIMENSION INFERENCE LOGIC ---
    if latent is not None:
        # If a latent is provided, automatically infer the height and width.
        height = latent.shape[2] * 8
        width = latent.shape[3] * 8
    
    batch_size = len(prompt)
    register_attention_control(model, controller)

    # --- NEW: Store height and width on the controller ---
    # This makes the dimensions automatically available to visualization functions.
    if controller is not None:
        controller.height = height
        controller.width = width
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        # The unique logic for handling pre-computed embeddings is preserved
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


# ptp_utils.py (Corrected with Attention Processors)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
from PIL import Image

# Helper functions like view_images, text_under_image, etc. remain the same.
# We will only define the core new logic here.

# --- New Attention Processor and Controller ---

class AttentionProcessor(AttnProcessor):
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # This is the forward pass of the attention module
        # It's what gets called by the UNet
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Here is the hook where we pass control to our custom controller
        attention_probs = self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def register_attention_control(model, controller):
    """
    This is the updated registration function.
    If the controller is not None, it sets up our custom AttentionProcessor.
    If the controller IS None, it restores the default behavior, safely disabling P2P.
    """
    if controller is not None:
        attn_procs = {}
        cross_att_count = 0
        for name in model.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttentionProcessor(
                controller=controller,
                place_in_unet=place_in_unet
            )

        model.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count
    else:
        # If the controller is None, manually restore the default attention processor.
        # This bypasses the diffusers safeguard and correctly resets the model's state.
        attn_procs = {}
        for name in model.unet.attn_processors.keys():
            attn_procs[name] = AttnProcessor()
        model.unet.set_attn_processor(attn_procs)

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words




# This cell contains all the patched controller classes.

# class LocalBlend:
#     # This class seems correct and doesn't need changes based on the errors we've seen.
#     # We will keep it as is.
#     def get_mask(self, maps, alpha, use_pool):
#         k = 1
#         maps = (maps * alpha).sum(-1).mean(1)
#         if use_pool:
#             maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
#         mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
#         mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
#         mask = mask.gt(self.th[1-int(use_pool)])
#         mask = mask[:1] + mask
#         return mask
    
#     def __call__(self, x_t, attention_store):
#         self.counter += 1
#         if self.counter > self.start_blend:
           
#             maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
#             maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
#             maps = torch.cat(maps, dim=1)
#             mask = self.get_mask(maps, self.alpha_layers, True)
#             if self.substruct_layers is not None:
#                 maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
#                 mask = mask * maps_sub
#             mask = mask.float()
#             x_t = x_t[:1] + mask * (x_t - x_t[:1])
#         return x_t
       
#     def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
#         alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
#         for i, (prompt, words_) in enumerate(zip(prompts, words)):
#             if type(words_) is str:
#                 words_ = [words_]
#             for word in words_:
#                 ind = get_word_inds(prompt, word, tokenizer)
#                 alpha_layers[i, :, :, :, :, ind] = 1
        
#         if substruct_words is not None:
#             substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
#             for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
#                 if type(words_) is str:
#                     words_ = [words_]
#                 for word in words_:
#                     ind = get_word_inds(prompt, word, tokenizer)
#                     substruct_layers[i, :, :, :, :, ind] = 1
#             self.substruct_layers = substruct_layers.to(device)
#         else:
#             self.substruct_layers = None
#         self.alpha_layers = alpha_layers.to(device)
#         self.start_blend = int(start_blend * NUM_DDIM_STEPS)
#         self.counter = 0 
#         self.th=th

class LocalBlend:
    
    def get_mask(self, x_t, maps, alpha, use_pool):
        """ MODIFIED: Now accepts `x_t` to get the latent shape for interpolation. """
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.max_num_words) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(x_t, maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(x_t, maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: List[List[str]], tokenizer, device,
                 substruct_words=None, start_blend=0.2, th=(.3, .3),
                 num_ddim_steps=50, max_num_words=77):
        """
        MODIFIED: Accepts `tokenizer`, `device`, `num_ddim_steps`, and `max_num_words`
        to eliminate global dependencies.
        """
        self.max_num_words = max_num_words
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.max_num_words)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0 
        self.th=th

# --- Patched AttentionControl Base Class ---
# class AttentionControl(abc.ABC):
    
#     def step_callback(self, x_t):
#         return x_t
    
#     def between_steps(self):
#         return
    
#     @property
#     def num_uncond_att_layers(self):
#         return 0

#     @abc.abstractmethod
#     def forward (self, attn, is_cross: bool, place_in_unet: str):
#         raise NotImplementedError

#     def __call__(self, attn, is_cross: bool, place_in_unet: str):
#         # This is the corrected logic from our previous session.
#         if self.cur_att_layer >= self.num_uncond_att_layers:
#             h = attn.shape[0] // 2
#             attn[h:] = self.forward(attn[h:], is_cross, place_in_unet)
        
#         self.cur_att_layer += 1
#         if self.cur_att_layer == self.num_att_layers:
#             self.cur_att_layer = 0
#             self.cur_step += 1
#             self.between_steps()
#         return attn
    
#     def reset(self):
#         self.cur_step = 0
#         self.cur_att_layer = 0

#     def __init__(self):
#         self.cur_step = 0
#         self.num_att_layers = -1
#         self.cur_att_layer = 0

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0] // 2
            attn[h:] = self.forward(attn[h:], is_cross, place_in_unet)
        
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

# --- Patched EmptyControl with the missing method ---
class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    def replace_cross_attention(self, attn_base, att_replace):
        return attn_base

# All other controller classes (AttentionStore, AttentionControlEdit, etc.)
# can remain the same as in the original notebook. They inherit the correct
# logic from the patched AttentionControl base class.

# class SpatialReplace(EmptyControl):
    
    # def step_callback(self, x_t):
    #     if self.cur_step < self.stop_inject:
    #         b = x_t.shape[0]
    #         x_t = x_t[:1].expand(b, *x_t.shape[1:])
    #     return x_t

    # def __init__(self, stop_inject: float):
    #     super(SpatialReplace, self).__init__()
    #     self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float, num_ddim_steps: int = 50):
        """ MODIFIED: Accepts `num_ddim_steps` to eliminate global dependency. """
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * num_ddim_steps)

        
# class AttentionStore(AttentionControl):

#     @staticmethod
#     def get_empty_store():
#         return {"down_cross": [], "mid_cross": [], "up_cross": [],
#                 "down_self": [],  "mid_self": [],  "up_self": []}

#     def forward(self, attn, is_cross: bool, place_in_unet: str):
#         key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
#         if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
#             self.step_store[key].append(attn)
#         return attn

#     def between_steps(self):
#         if len(self.attention_store) == 0:
#             self.attention_store = self.step_store
#         else:
#             for key in self.attention_store:
#                 for i in range(len(self.attention_store[key])):
#                     self.attention_store[key][i] += self.step_store[key][i]
#         self.step_store = self.get_empty_store()

#     def get_average_attention(self):
#         average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
#         return average_attention

#     def reset(self):
#         super(AttentionStore, self).reset()
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}

#     def __init__(self):
#         super(AttentionStore, self).__init__()
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                if key in self.attention_store and key in self.step_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] if self.cur_step > 0 else [] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}



# class AttentionControlEdit(AttentionStore, abc.ABC):
    
#     def step_callback(self, x_t):
#         if self.local_blend is not None:
#             x_t = self.local_blend(x_t, self.attention_store)
#         return x_t
        
#     def replace_self_attention(self, attn_base, att_replace, place_in_unet):
#         if att_replace.shape[2] <= 32 ** 2:
#             attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
#             return attn_base
#         else:
#             return att_replace
    
#     @abc.abstractmethod
#     def replace_cross_attention(self, attn_base, att_replace):
#         raise NotImplementedError
    
#     def forward(self, attn, is_cross: bool, place_in_unet: str):
#         super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
#         if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
#             h = attn.shape[0] // (self.batch_size)
#             attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
#             attn_base, attn_repalce = attn[0], attn[1:]
#             if is_cross:
#                 alpha_words = self.cross_replace_alpha[self.cur_step]
#                 attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
#                 attn[1:] = attn_repalce_new
#             else:
#                 attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
#             attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
#         return attn
    
#     def __init__(self, prompts, num_steps: int,
#                  cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
#                  self_replace_steps: Union[float, Tuple[float, float]],
#                  local_blend: Optional[LocalBlend]):
#         super(AttentionControlEdit, self).__init__()
#         self.batch_size = len(prompts)
#         self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
#         if type(self_replace_steps) is float:
#             self_replace_steps = 0, self_replace_steps
#         self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
#         self.local_blend = local_blend


class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        
        # MODIFIED: This is the crucial logic from your recovered code.
        # It checks if the current attention layer should be edited.
        if place_in_unet not in self.edit_layers:
            return attn

        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],
                 tokenizer, device,
                 # MODIFIED: Added the optional `edit_layers` argument.
                 edit_layers: Optional[List[str]] = None):
        """ MODIFIED: Accepts `tokenizer`, `device`, and `edit_layers`. """
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        
        # MODIFIED: Stores the edit_layers, defaulting to all layers if not provided.
        self.edit_layers = edit_layers if edit_layers is not None else ["down", "mid", "up"]



# class AttentionReplace(AttentionControlEdit):

#     def replace_cross_attention(self, attn_base, att_replace):
#         return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
#     def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
#                  local_blend: Optional[LocalBlend] = None):
#         super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
#         self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

# class AttentionRefine(AttentionControlEdit):

#     def replace_cross_attention(self, attn_base, att_replace):
#         attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
#         attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
#         return attn_replace

#     def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
#                  local_blend: Optional[LocalBlend] = None):
#         super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
#         self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
#         self.mapper, alphas = self.mapper.to(device), alphas.to(device)
#         self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


# class AttentionReweight(AttentionControlEdit):

#     def replace_cross_attention(self, attn_base, att_replace):
#         if self.prev_controller is not None:
#             attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
#         attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
#         return attn_replace

#     def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
#                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
#         super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
#         self.equalizer = equalizer.to(device)
#         self.prev_controller = controller


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer=None, device=None,
                 edit_layers: Optional[List[str]] = None):
        """ MODIFIED: Accepts `tokenizer`, `device`, and `edit_layers`. """
        # MODIFIED: Pass all arguments, including edit_layers, to the super constructor.
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, 
                                               local_blend, tokenizer, device, edit_layers)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer=None, device=None,
                 # MODIFIED: Added the optional `edit_layers` argument.
                 edit_layers: Optional[List[str]] = None):
        """ MODIFIED: Accepts `tokenizer`, `device`, and `edit_layers`. """
        # MODIFIED: Pass all arguments, including edit_layers, to the super constructor.
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, 
                                              local_blend, tokenizer, device, edit_layers)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None,
                 tokenizer=None, device=None):
        """ MODIFIED: Accepts `tokenizer` and `device`. """
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


# Other helper functions like get_equalizer, make_controller, etc.
# can also remain the same.

# def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
#                   Tuple[float, ...]]):
#     if type(word_select) is int or type(word_select) is str:
#         word_select = (word_select,)
#     equalizer = torch.ones(1, 77)
    
#     for word, val in zip(word_select, values):
#         inds = get_word_inds(text, word, tokenizer)
#         equalizer[:, inds] = val
#     return equalizer

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]], tokenizer):
    """
    Creates an equalizer tensor to re-weight attention on specific words.
    - MODIFIED: Now accepts `tokenizer` as an argument.
    """
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        # Uses the passed-in tokenizer
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

# def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
#     out = []
#     attention_maps = attention_store.get_average_attention()
#     num_pixels = res ** 2
#     for location in from_where:
#         for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
#             if item.shape[1] == num_pixels:
#                 cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
#                 out.append(cross_maps)
#     out = torch.cat(out, dim=0)
#     out = out.sum(0) / out.shape[0]
#     return out.cpu()

# This is the updated code for your notebook cell

def aggregate_attention(
    attention_store: "AttentionStore", 
    res: int, 
    from_where: Tuple[str, ...], 
    is_cross: bool, 
    select: int, 
    prompts: List[str]  # Added prompts as an argument
):
    """
    This is the smarter version of the function that automatically finds the
    attention map closest to the requested resolution.
    """
    out = []
    attention_maps = attention_store.get_average_attention()
    
    height = attention_store.height
    width = attention_store.width
    
    target_res = res ** 2
    
    maps = []
    for location in from_where:
        key = f"{location}_{'cross' if is_cross else 'self'}"
        if key in attention_maps: # Check if key exists
            maps.extend(attention_maps[key])
    
    if not maps:
        return torch.zeros(0) # Return empty if no maps found for any location

    best_map = None
    min_diff = float('inf')
    for item in maps:
        diff = abs(item.shape[1] - target_res)
        if diff < min_diff:
            min_diff = diff
            best_map = item
    
    if best_map is None:
        return torch.zeros(0)

    aspect_ratio = width / height
    h = int((best_map.shape[1] / aspect_ratio) ** 0.5)
    w = int(h * aspect_ratio)
    
    # The `prompts` list is now passed in, making the function self-contained
    cross_maps = best_map.reshape(len(prompts), -1, h, w, best_map.shape[-1])[select]
    out.append(cross_maps)
        
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(
    attention_store: "AttentionStore", 
    res: int, 
    from_where: List[str], 
    tokenizer,               # Added tokenizer as an argument
    prompts: List[str],         # Added prompts as an argument
    select: int = 0
):
    
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    
    # Pass prompts down to the aggregation function
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompts)
    
    if attention_maps.nelement() == 0:
        print("Could not find any attention maps for the given resolution.")
        return

    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    view_images(np.stack(images, axis=0))



# def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
#     if blend_words is None:
#         lb = None
#     else:
#         lb = LocalBlend(prompts, blend_words) # Corrected a typo here: blend_word -> blend_words
#     if is_replace_controller:
#         controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
#     else:
#         controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
#     if equilizer_params is not None:
#         eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
#         controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
#                                        self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
#     return controller

def make_controller(prompts: List[str], is_replace_controller: bool,
                    cross_replace_steps: Dict[str, float], self_replace_steps: float,
                    tokenizer, device, num_ddim_steps=50,
                    blend_words=None, equilizer_params=None) -> "AttentionControlEdit":
    """
    This is the updated factory function. It now accepts `tokenizer` and `device`
    and correctly passes them down to the controller classes it creates.
    """
    if blend_words is None:
        lb = None
    else:
        # Pass all required dependencies to LocalBlend
        lb = LocalBlend(prompts, blend_words, tokenizer=tokenizer, device=device,
                        num_ddim_steps=num_ddim_steps)
    
    if is_replace_controller:
        controller = AttentionReplace(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                      self_replace_steps=self_replace_steps, local_blend=lb,
                                      tokenizer=tokenizer, device=device)
    else:
        controller = AttentionRefine(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps, local_blend=lb,
                                     tokenizer=tokenizer, device=device)
    
    if equilizer_params is not None:
        # Pass the tokenizer to get_equalizer
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer)
        controller = AttentionReweight(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq,
                                       local_blend=lb, controller=controller,
                                       tokenizer=tokenizer, device=device)
    return controller

# def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
#     tokens = tokenizer.encode(prompts[select])
#     decoder = tokenizer.decode
#     attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
#     images = []
#     for i in range(len(tokens)):
#         image = attention_maps[:, :, i]
#         image = 255 * image / image.max()
#         image = image.unsqueeze(-1).expand(*image.shape, 3)
#         image = image.numpy().astype(np.uint8)
#         image = np.array(Image.fromarray(image).resize((256, 256)))
#         image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
#         images.append(image)
#     ptp_utils.view_images(np.stack(images, axis=0))


# TO MODIFIY!
def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    view_images(np.concatenate(images, axis=1))