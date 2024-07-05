from torchmetrics.functional.multimodal import clip_score
import numpy as np
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.linalg import sqrtm
from PIL import Image

def calculate_clip_score(images, prompts, model_name_or_path="openai/clip-vit-base-patch16"):
    clip_score_fn = partial(clip_score, model_name_or_path=model_name_or_path)

    def calculate_clip_score(images, prompts):
        images_int = (images * 255).astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)

    sd_clip_score = calculate_clip_score(images, prompts)

    return sd_clip_score

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_fid(model, real_images, fake_images):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    real_images = [preprocess(Image.fromarray(img)).unsqueeze(0) for img in real_images]
    fake_images = [preprocess(Image.fromarray(img)).unsqueeze(0) for img in fake_images]

    real_images = torch.cat(real_images)
    fake_images = torch.cat(fake_images)
    
    def get_activations(images):
        images = images.to("cpu")
        with torch.no_grad():
            pred = model(images)
        pred = F.adaptive_avg_pool2d(pred, (1, 1))
        return pred.squeeze(3).squeeze(2).cpu().numpy()

    real_activations = get_activations(real_images)
    fake_activations = get_activations(fake_images)
    
    mu1, sigma1 = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid