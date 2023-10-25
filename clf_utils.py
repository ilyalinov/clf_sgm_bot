import torch
import gdown
import numpy as np
from torchvision import models, transforms
from PIL import Image

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_clf_model():
    model_clf = models.resnet18()
    num_features = model_clf.fc.in_features
    model_clf.fc = torch.nn.Linear(num_features, 2)

    # pretrained weights url
    url = 'https://drive.google.com/uc?id=12tmF8f8HXLcv4eX5J3-q17JOARPe8vKK'
    output = 'resnet.pth'
    gdown.download(url, output, quiet=False)
    DEVICE = get_device()
    model_clf.load_state_dict(torch.load(output, map_location=DEVICE))
    model_clf.to(DEVICE)
    return model_clf

def predict_one_sample(model, img, device):
    """Предсказание, для одной картинки"""
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    model.eval()
    logit = model(img).cpu()
    probs = torch.nn.functional.softmax(logit, dim=-1).detach().numpy()
    return probs

def classify_one_image(model, img, device):
    probs = predict_one_sample(model, img, device)
    probs = np.squeeze(probs, axis=0)
    res_dict = {}
    score = probs[0] if probs[0] > 0.5 else probs[1]
    res_dict['score'] = str(round(score, 5))
    if probs[0] > 0.5:
        score = probs[0]
        res_dict['type'] = 'nevus'
        return res_dict
    else:
        res_dict['type'] = 'melanoma'
        return res_dict
    
def dict_to_str(d):
    res_str = ''
    # print(d)
    for k in d:
        res_str += f'{k}:  {d[k]}\n'

    return res_str