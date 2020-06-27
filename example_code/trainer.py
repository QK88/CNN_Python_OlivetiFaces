import torch
from torch import nn, optim
import numpy as np
from dataset import FaceDataSet
from classifier import SimpleConvNN,Neuralnetwork
import cv2
# configs
im_train_set = 'FaceIdenfier/datasets/att_faces_train'
im_val_set = 'FaceIdenfier/datasets/att_faces_val'
is_train = True
batch_size = 80
max_epochs = 50
im_height = 112
im_width = 92
def train():
    # build model
    print('Loading model...')
    model = Neuralnetwork(3,40)
    model.train()
    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # prepare dataset
    print('Loading dataset...')
    
    db = FaceDataSet(im_train_set, is_train, batch_size)
#TODO: 
    # training
    epoch = 0
    num_iters = db.num_samples // batch_size
    print('start training')
    while epoch != max_epochs:
        running_loss = 0.0
        running_acc = 0.0
        for iter in range(num_iters):
            data_x, data_y = db.get_batch_data()
            data_x = torch.from_numpy(data_x)
            data_y = torch.from_numpy(data_y)
            # 根据dim 与class搭建神经网络
            out = model(data_x)
            loss = criterion(out, data_y)

            running_loss += loss.item() * data_y.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == data_y).sum()
            running_acc += num_correct.item()
            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 30 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, epoch, running_loss / (batch_size * (iter+1)),
                    running_acc / (batch_size * (iter+1))))
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (db.num_samples), running_acc /
                db.num_samples))
        epoch += 1
    print('validating...')
    validate(model)
    print('Done')

def validate(model):
    print('Loading dataset...')
    model.eval()
    db = FaceDataSet(im_val_set, False, 1)
    num_iters = db.num_samples
    running_acc = 0.0
    for iter in range(num_iters):
        data_x, data_y = db.get_batch_data()
        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)
        out = model(data_x)
        _, pred = torch.max(out, 1)
        num_correct = (pred == data_y).sum()
        running_acc += num_correct.item()
    print('Finish, Acc: {:.6f}'.format(running_acc /db.num_samples))
if __name__ == '__main__':
    train()


