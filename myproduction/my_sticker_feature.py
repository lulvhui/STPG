import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import clip
from transformers import ViTModel, ViTConfig, ViTImageProcessor

import os


def get_sticker_text(img_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch") # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    result = ocr.ocr(img_path, cls=True)
    result = result[0]
    txts = [line[1][0] for line in result]
    text = ''
    for i in range(len(txts)):
        text += txts[i]
    return text 

def get_sticker_feature_ResNet50(img_path):
    # 加载预训练的ResNet模型
    resnet = models.resnet50(pretrained=True)
    # 移除ResNet模型的全连接层
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    # 设置输入图像的预处理转换
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 加载图像
    image = Image.open(img_path).convert('RGB')
    # 对图像进行预处理
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    # 将图像输入ResNet模型获取特征
    with torch.no_grad():
        features = resnet(input_batch)
    # 将特征转换为一维向量
    feature_vector = torch.flatten(features, start_dim=1)
    # 打印特征向量的形状
    print(feature_vector.shape)#[1, 2048]
    return feature_vector
    
def get_sticker_feature_ViT(img_path):
    model_path = "/data/yyang/Temp_Model/vit-base-patch16-224"
    model = ViTModel.from_pretrained(model_path)
    feature_extractor = ViTImageProcessor.from_pretrained(model_path)
    image = Image.open(img_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors='pt')
    
    # 提取特征
    with torch.no_grad():
        features = model(**inputs).last_hidden_state.squeeze(0)
    print(features.shape)
    return features  # torch.Size([197, 768])
    
    
def get_sticker_feature_clip(img_path):
    # 加载预训练的CLIP模型
    model, preprocess = clip.load('ViT-B/32', device='cpu')
    # 图像预处理
    image = Image.open(img_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to('cpu')
    # 提取特征
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    # 打印特征向量的形状
    print(image_features.shape)#[1, 512]
    return image_features


if __name__ == '__main__':
    import os
    imgs_pth = '/data/yyang/M^4Comment_Detection/meme_dataset/meme'
    # 遍历imgs_pth下的所有图片
    for img_name in os.listdir(imgs_pth):
        features = get_sticker_feature_clip(os.path.join(imgs_pth, img_name))
        # 将特征保存到文件夹
        torch.save(features, os.path.join('/data/yyang/Temp_Model/M^4_clip_features', img_name.split('.')[0]+'.pt'))


'''
f2 = get_sticker_feature_ViT('/data/yyang/MSAIRS/1274.png')
torch.save(f2, '/data/yyang/Temp_Model/Clip_Features/1274.pt')
'''