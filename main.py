import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
#from unet import Unet
from unet_model import Unet
#from unet_edit import Unet
from network import R2U_Net,AttU_Net,R2AttU_Net,U_Net
from unet_resnet18 import ResNetUNet
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from dataset import LiverDataset
import numpy as np
from eval import eval_net , eval_net1

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    #transforms.Resize(256),
    #transforms.RandomCrop(64),
    #transforms.RandomRotation((-10,10)),
    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize([0.5], [0.5])
])

# mask只需要转换为tensor

#y_transforms = transforms.ToTensor()
y_transforms = transforms.Compose([
    #transforms.Resize(256),
    #transforms.RandomCrop(64),
    transforms.ToTensor()
])

# mask只需要转换为tensor
def val(epoch,model, criterion, optimizer, dataload):
    model.eval()
    liver_dataset = LiverDataset("/home/cvlab04/Desktop/Code/Medical/u_net_liver/data/val/", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=6)
    step=0
    epoch_loss = 0
    print("Validation...")
    with torch.no_grad():
        for x, y in dataloaders:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
        print("epoch %d Val_loss:%0.5f " % (epoch, epoch_loss/step))
    return epoch_loss/step

def train_model(model, criterion, optimizer, dataload, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        #num_correct = 0
        for x, y in dataload:
            num_correct = 0
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
                       

            #print(num_correct)
            #val_score = eval_net1(model, inputs,labels,6 )
            #print(val_score)
            print("%d/%d,train_loss:%0.5f " % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
            #val_score = eval_net(model, dataload,6 )
        
        print("epoch %d loss:%0.5f " % (epoch, epoch_loss/step))
        val_loss = val(epoch,model, criterion, optimizer, dataload)
        fp = open("Log_weights_24_unet_Bi2.txt", "w")
        fp.write("epoch %d loss:%0.5f Val_loss:%0.5f\n" % (epoch, epoch_loss/step,val_loss))
        fp.close()
        
    torch.save(model.state_dict(), 'weights_%d_unet_Bi2.pth' % epoch)
    return model

#训练模型
def train(args):
    #vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    #model = FCN8s(pretrained_net=vgg_model, n_class=1).to(device)
    model = Unet(1, 1).to(device)
    #model = R2AttU_Net().to(device)
    #model = AttU_Net().to(device)
    #model = R2U_Net().to(device)
    #model = U_Net().to(device)
    #model = ResNetUNet(1).to(device)
    batch_size = args.batch_size
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0, weight_decay=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    liver_dataset = LiverDataset("/home/cvlab04/Desktop/Code/Medical/u_net_liver/data/train/",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    train_model(model, criterion, optimizer, dataloaders)

#显示模型的输出结果
def test(args):
    model = Unet(1, 1)
    #model = R2AttU_Net()
    #model = U_Net()
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    #liver_dataset = LiverDataset("/home/cvlab04/Desktop/Code/Medical/u_net_liver/data/val/", transform=x_transforms,target_transform=y_transforms)
    liver_dataset = LiverDataset("/home/cvlab04/Desktop/Code/Medical/u_net_liver/data/train/", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=6)
    model.eval()
    import matplotlib.pyplot as plt
    #plt.ion()
    count=0
    count_sum=0.
    dice_loss=0.
    with torch.no_grad():
        for x, labels in dataloaders:
            count+=1
            print("batch:",count)
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            img_y = (img_y> 0.3).astype(np.uint8)
            img_y = img_y.flatten()
            count_predict=np.count_nonzero(img_y > 0)
            #print("predict pixel:   ",count_predict)
            true =torch.squeeze(labels).numpy()
            true  = true .flatten()
            count_true=np.count_nonzero(true > 0)
            #print("true pixel:   ",count_true)
            ans=0
            '''
            for i in range(len(img_y)):
                for j in range(len(img_y)):
                    if img_y[i][j]>0 and true[i][j]>0:
                        ans+=1
            '''
            ans = np.count_nonzero(img_y*true>0)
            dice_loss = (2*ans+0.0001)/(count_predict+count_true + 0.0001)
            print("dice_loss:",dice_loss)
            
            count_sum += (dice_loss)
            
            
            #plt.imshow(img_y)
            #plt.pause(1)
        #plt.show()
        print("Final_Dice_Loss:",count_sum/count)

def check(args):
    model = Unet(1, 1)
    #model = R2AttU_Net()
    #model = AttU_Net()
    #model = U_Net()
    #vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    #model = FCN8s(pretrained_net=vgg_model, n_class=1)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    #liver_dataset = LiverDataset("/home/cvlab04/Desktop/Code/Medical/u_net_liver/data/val/", transform=x_transforms,target_transform=y_transforms)
    #dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import PIL.Image as Image
    img = Image.open('/home/cvlab04/Desktop/Code/Medical/u_net_liver/check/train/A001-2_instance-47.jpeg')
    #img = Image.open('/home/cvlab04/Desktop/Code/Medical/u_net_liver/A001-23230277-27.jpeg').convert('RGB')
    img = x_transforms(img)
    img = img.view(1,1,512,512)
    #img = img.view(1,1,64,64)
    #img = img.view(1,3,512,512)
    #img = img.to(device=device, dtype=torch.float32)
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        y=model(img)
        y = torch.sigmoid(y)
        y = y.squeeze(0)
        print(y)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(512),
                transforms.ToTensor()
            ]
        )
        
        
        #y = tf(y.cpu())
        #print(y)
        img_y = y.squeeze().cpu().numpy()
        print(img_y)
        img_y = (img_y> 0.3).astype(np.uint8)   #0.3 / 0.01 for unet #3e-4 for r2attunet #3e-1 for unet-trans
        print(img_y)
        im = Image.fromarray((img_y*255).astype(np.uint8))
        im.save("/home/cvlab04/Desktop/Code/Medical/u_net_liver/check/result/Threshold03_U_Net_transpose_25_epoch_A001-2_instance-47.png")
        
        plt.imshow(img_y, plt.cm.gray)
        plt.pause(2)
        plt.show()

        
    

if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
    elif args.action=="check":
        check(args)
