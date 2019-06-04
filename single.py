import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

import numpy as np
import scipy.io
from PIL import Image
from sppnet import SPPNet

image_path = './data/jpg'
label_path = './data/imagelabels.mat'
# labels 1x8189
setid_path = './data/setid.mat'
# tstid 1x6149
# trnid 1x1020
# valid 1x1020
save_path = './data/model_single.pth'

BATCH = 1
EPOCH = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, image_path, label_path, setid_path, train=True, transform=None):
        """
        image_00001.jpg
        image_00002.jpg
        image_00003.jpg
        """
        setid = scipy.io.loadmat(setid_path)
        labels = scipy.io.loadmat(label_path)['labels'][0] # 1x8189

        if train :
            trnid = setid['tstid'][0] # tstid는 train set의 label
            self.labels = [labels[i-1] -1 for i in trnid]  #trnid -> 1x6149 trnid에는 그 이미지가 몇 번째 라벨인지가 들어있다. 
            self.images = ['%s/image_%05d.jpg' % (image_path, i) for i in trnid] # 1에서 6149까지의 데이터셋
        else :
            tstid = np.append(setid['valid'][0], setid['trnid'][0]) # tr
            self.labels = [labels[i-1] -1 for i in tstid] # 한 줄로 처리
            self.images = ['%s/image_%05d.jpg' % (image_path, i) for i in tstid] # 한 줄로 처리
        self.transform= transform



    def __getitem__(self, index):
        label = self.labels[index] # 배열 형태로 만든 것을 index화 시킨다. 
        image = self.images[index] # index화 시킴-> 얻는 것은 filename임
        print("before",image)
        if self.transform is not None:
            image = self.transform(Image.open(image)) # transform은 이미지를 변형할 때 쓴다. /// filename을 open하여 이미지를 얻는다.
        print("after",image)
        return image, label

    def __len__(self):
        return len(self.labels)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    # batch_idx는 여기서 나온다.
    for batch_idx, (image, label) in enumerate(train_loader):
        image , label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        label=label.type(torch.LongTensor).cuda()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() # 200개까지의 loss를 다 더한다. 
        if (batch_idx + 1) % 200 ==0 :
            train_loss /= 200

            print('Train Epoch: %d [%d/%d (%.4f%%)\tLoss: %.4f]'% (
                epoch, (batch_idx + 1) * len(image), len(train_loader.dataset),
                100. * (batch_idx + 1) * len(image) / len(train_loader.dataset), train_loss)) # 그리고 loss 출력
            train_loss = 0


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    total_true =0
    total_loss =0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            label=label.type(torch.LongTensor).cuda()
            loss = criterion(output, label)
            pred = torch.max(output, 1)[1]
            total_loss += (pred.view(label.size()).data == label.data).sum().item()
            total_loss += loss.item()

    accuracy = total_true / len(test_loader.dataset)
    loss = total_loss / len(test_loader.dataset)
    print('\nTest Epoch: %d ====> Accuracy: [%d/%d (%.4f%%)]\tAverage loss: %.4f\n' % (epoch, total_true, len(test_loader.dataset), 100. * accuracy, loss))


if __name__ == '__main__':

    # dataset의 크기는 각기 다르다.
    train_dataset = MyDataset(image_path, label_path, setid_path,
                              train=True, transform=
                              transforms.Compose([
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(30),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])) # imagenet mean, std

    train_loader = DataLoader(train_dataset, batch_size = BATCH, shuffle=True)
    print('Train size:', len(train_loader))

    test_dataset = MyDataset(image_path, label_path, setid_path,
                             train=False, transform=
                             transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    print('Test size:', len(test_loader))

    model = SPPNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCH +1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        torch.save(model, save_path)
        test(model, device, test_loader, criterion, epoch)

