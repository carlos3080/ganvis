import sys
sys.path.append('./AudioCLIP')
sys.path.append('./stylegan3')
sys.path.append('.')

import io
import os, time, glob
import pickle
import shutil
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from audioclip import AudioCLIP
import unicodedata
import re
from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from IPython.display import display
import IPython.display as disp
from einops import rearrange
from google.colab import files
import random
import math
import time
from abc import ABC, abstractmethod
import json
import pyaudio
import wave
import dill
from torch.multiprocessing import Process, set_start_method
from threading import Thread
# import multiprocessing as mp
import torch.multiprocessing as mp

def encode_audio_and_make_image(input_queue, output_queue, estimator_path):
    stylegan = StyleGan3('Wikiart').model
    print("post style gan")
    make_cutouts = MakeCutouts(224, 32, 0.5)
    print("post cutouts")
    audio_clip = Audio_CLIP()
    print("post audio clip")
    gan_vis = GanVis(stylegan, audio_clip, make_cutouts)
    print("post gan vis")
    q_estimator = torch.load(estimator_path)
    print("post estimator")
    while True:
        audio = input_queue.get()
        audio = np.concatenate([np.frombuffer(chunk, dtype='int16') for chunk in audio])
        audio = np.float32(audio)
        audio_enc = gan_vis.audio_clip.embed_audio_from_raw(audio)
        q = q_estimator(audio_enc)
        img = gan_vis.gen_img(q, 'const').detach()
        im = TF.to_pil_image(tf(img)[0])
        im.show()
        # output_queue.put(img)

device = torch.device('cuda:1')
cpu = torch.device('cpu')
print('Using device:', device, file=sys.stderr)

tf = Compose([
  Resize(224),
  lambda x: torch.clamp((x+1)/2,min=0,max=1),
  ])

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

class StyleGan3():
    def __init__(self, name):
        self.model = self.load_model(name)

    def fetch_model(self, url_or_path):
        if "drive.google" in url_or_path:
            if "18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj" in url_or_path: 
                basename = "wikiart-1024-stylegan3-t-17.2Mimg.pkl"
            elif "14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1" in url_or_path:
                basename = "lhq-256-stylegan3-t-25Mimg.pkl"
        else:
            basename = os.path.basename(url_or_path)
        if os.path.exists(basename):
            return basename
        # else:
        #     if "drive.google" not in url_or_path:
        #     !wget -c '{url_or_path}'
        #     else:
        #     path_id = url_or_path.split("id=")[-1]
        #     !gdown --id '{path_id}'
        #     return basename
        else:
            print("fcked")
            return 

    def load_model(self, name):
        base_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"
        model_name = {
            "FFHQ": base_url + "stylegan3-t-ffhqu-1024x1024.pkl",
            "MetFaces": base_url + "stylegan3-r-metfacesu-1024x1024.pkl",
            "AFHQv2": base_url + "stylegan3-t-afhqv2-512x512.pkl",
            "cosplay": "https://l4rz.net/cosplayface-snapshot-stylegan3t-008000.pkl",
            "Wikiart": "https://archive.org/download/wikiart-1024-stylegan3-t-17.2Mimg/wikiart-1024-stylegan3-t-17.2Mimg.pkl",
            "Landscapes": "https://archive.org/download/lhq-256-stylegan3-t-25Mimg/lhq-256-stylegan3-t-25Mimg.pkl"
        }
        with open(self.fetch_model(model_name[name]), 'rb') as fp:
            model = pickle.load(fp)['G_ema'].to(device)
        return model

class Audio_CLIP():
    def __init__(self):
        torch.set_grad_enabled(False)
        self.model = AudioCLIP(pretrained='AudioCLIP-Partial-Training.pt').to(device)
        self.model.eval()
        torch.set_grad_enabled(True)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    def norm1(self, prompt):
        "Normalize to the unit sphere."
        return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()
    
    @torch.no_grad()
    def embed_audio(self, audio):
        "Normalized clip audio embedding."
        return self.norm1(self.model.create_audio_encoding(audio))

    @torch.no_grad()
    def embed_audio_from_raw(self, audio):
        audio = audio[:, np.newaxis]
        audio = audio.T
        audio = torch.from_numpy(audio).float().to(device)
        with torch.no_grad():
            audio_encoding = self.model.encode_audio(audio).detach()
            # print("audio enc inner shape: ", audio_encoding.shape)
        return self.norm1(audio_encoding)
    
    def embed_cutout(self, image):
        "Normalized clip image embedding."
        return self.norm1(self.model.encode_image(self.normalize(image)))

    @torch.no_grad()    
    def embed_text(self, text):
        return self.norm1(self.model.encode_text(text))
    
################################################################################################
class GanVis(torch.nn.Module):
    def __init__(self, stylegan, audio_clip, make_cutouts):
        super().__init__()
        self.stylegan = stylegan
        self.audio_clip = audio_clip
        self.make_cutouts = make_cutouts
        self.w_stds = torch.load('w_stds.pt')
        
    def forward(self):
        G = self.stylegan
        img = G.synthesis(self.q * self.w_stds + G.mapping.w_avg, noise_mode='const')
        embed = self.embed_image(img.add(1).div(2))
        loss = self.compute_loss(embed)
        return loss
    
    def set_embeddings(self, audios, texts):
        self.audio_embeds = [self.audio_clip.embed_audio(audio) for audio in audios]
        self.text_embeds = [self.audio_clip.embed_text([text]) for text in texts]
    
    def compute_initial_q(self, explore):
        self.explore = explore
        with torch.no_grad():
            qs = []
            losses = []
            for _ in range(self.explore):
                G = self.stylegan
                q = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None, truncation_psi=0.7) - G.mapping.w_avg) / self.w_stds
                #   q = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None))
                images = G.synthesis(q * self.w_stds + G.mapping.w_avg)
                embeds = self.embed_image(images.add(1).div(2))
                loss = self.compute_loss(embeds).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
            qs = torch.stack(qs)
            losses = torch.stack(losses)
            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0).requires_grad_()
            self.q = q
            
    def compute_loss(self, img_embed):
            
        def spherical_dist_loss(x, y):
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
            return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
            
        loss = spherical_dist_loss   
        if len(self.audio_embeds) == 1: 
            return loss(img_embed, self.audio_embeds[0])
        distances = [loss(img_embed, audio_embed) for audio_embed in self.audio_embeds]
        return torch.stack(distances, dim=-1).sum(dim=-1) 
    
    def gen_img(self, latent, noise):
        G = self.stylegan
        return G.synthesis(latent * self.w_stds + G.mapping.w_avg, noise_mode=noise)
    
    def compute_w_stds(self):
        G = self.stylegan
        zs = torch.randn([10000, G.mapping.z_dim], device=device)
        w_stds = G.mapping(zs, None).std(0)
        return w_stds
    
    def embed_image(self, img):
        n = img.shape[0]
        cutouts = self.make_cutouts(img)
        embeds = self.audio_clip.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds
        
    def embed_text(self, text):
        return self.audio_clip.embed_text(text)

class Latent(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.network = self.build_network()

    def forward(self, x):
        logits = self.network(x).reshape((-1, 16, 512))
        return logits
    
    @abstractmethod
    def build_network(self):
        pass
    
class SmallLatent(Latent):
    def __init__(self):
        super().__init__()
        self.name = "small_latent"
    
    def build_network(self):
        network = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 8192),
        )
        return network
    
class NormalLatent(Latent): # one hidden layer
    def __init__(self):
        super().__init__()
        self.name = "normal_latent"
        
    def build_network(self):
        network = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 8192)
        )
        return network
    
class BigLatent(Latent): # two hidden layers
    def __init__(self):
        super().__init__()
        self.name = "big_latent"
    
    def build_network(self):
        network = torch.nn.Sequential(
            torch.nn.Linear(1024, 2816),
            torch.nn.ReLU(),
            torch.nn.Linear(2816, 4608),
            torch.nn.ReLU(),
            torch.nn.Linear(4608, 6400),
            torch.nn.ReLU(),
            torch.nn.Linear(6400, 8192)
        )
        return network

class HugeLatent(Latent): # four hidden layers
    def __init__(self):
        super().__init__()
        self.name = "huge_latent"
    
    def build_network(self):
        network = torch.nn.Sequential(
            torch.nn.Linear(1024, 2218),
            torch.nn.ReLU(),
            torch.nn.Linear(2218, 3412),
            torch.nn.ReLU(),
            torch.nn.Linear(3412, 4606),
            torch.nn.ReLU(),
            torch.nn.Linear(4606, 5800),
            torch.nn.ReLU(),
            torch.nn.Linear(5800, 6994),
            torch.nn.ReLU(),
            torch.nn.Linear(6994, 8192)
        )
        return network
    
class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_samples = self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return self.n_samples



if __name__ == '__main__':
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    LOOP_HZ = RATE / CHUNK
    WINDOW_SECONDS = 10
    WINDOW_SAMPLE_HZ = 1
    WINDOW_SIZE = int(RATE / CHUNK * WINDOW_SECONDS)
    SHORT_NORMALIZE = 1.0 / 32768.0
    print(LOOP_HZ)

    
    p = pyaudio.PyAudio()
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    estimator_path = "models/big_latent_0.005_300_32_4.pt"
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    proc = mp.Process(target=encode_audio_and_make_image, args=(input_queue, output_queue, estimator_path))             
    proc.start()
    print("after start")
    time.sleep(10)
    print("after sleep")
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = [[]]
    i = 0
    start = None
    while stream.is_active():
        data = stream.read(CHUNK) # this is the limiting step in terms of time
        for frame in frames:
            frame.append(data)
        if len(frames[0]) == WINDOW_SIZE:
            audio = frames[0]
            input_queue.put(audio)
            # img = output_queue.get()
            # disp.clear_output(wait=True)
            # display(TF.to_pil_image(tf(img)[0]))
            # im = TF.to_pil_image(tf(img)[0])
            # im.show()
            frames.pop(0)
        if i % int(LOOP_HZ / WINDOW_SAMPLE_HZ) == 0:
            frames.append([])
        i += 1