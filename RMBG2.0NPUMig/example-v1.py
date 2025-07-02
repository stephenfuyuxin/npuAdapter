from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch_npu
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from torch_npu.contrib import transfer_to_npu

#model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
model = AutoModelForImageSegmentation.from_pretrained('/data/fuyuxin/zhaohang/RMBG2.0/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
#model.to('cuda')
model.to('npu')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#image = Image.open(input_image_path)
image = Image.open("/data/fuyuxin/zhaohang/RMBG2.0/RMBG-2.0/p1.jpg")
#input_images = transform_image(image).unsqueeze(0).to('cuda')
input_images = transform_image(image).unsqueeze(0).to('npu')

# Prediction
with torch.no_grad():
    #preds = model(input_images)[-1].sigmoid().cpu()
    preds = model(input_images)[-1].sigmoid().npu()
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
image.putalpha(mask)

image.save("no_bg_image.png")
