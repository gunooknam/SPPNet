import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from sppnet import SPPNet

model_path = './data/model_single.pth'
image_path = './data/test.jpg'

if __name__ == '__main__':

    image=Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image = transform(image)

    # the second way to load model
    model = SPPNet()
    model = torch.load(model_path) #
    model = model.cpu()
    #print(image.unsqueeze(0).shape) # [1, 3, 500, 667] 이다.
    output = model(image.unsqueeze(0))
    pred = torch.max(output, 1)[1]  # get the index of the max log-probability
    print("Prediction : ",int(pred)+1)
    test=cv2.imread(image_path)
    cv2.imshow("test.jpg", test)
    cv2.waitKey(0)
