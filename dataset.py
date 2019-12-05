from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root):
    '''
    imgs=[]
    n=len(os.listdir(root))//2

    for i in range(n):
        img=os.path.join(root',"%03d.png"%i)
        mask=os.path.join(root',"%03d_mask.png"%i)
        imgs.append((img,mask))

    return imgs
    '''
    imgs=[]
    masks=[]
    for filename in os.listdir(root):
        for filename2 in os.listdir(root+filename):
            if filename == 'annotations':
                mask=os.path.join(root+'annotations/'+filename2)
                masks.append(mask)
            if filename == 'COCO2019':
                img=os.path.join(root+'COCO2019/'+filename2)
                imgs.append(img)
    masks.sort()
    imgs.sort()  
    conbine=list(zip(imgs,masks))
    #print(conbine)
    return conbine


class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        #img_x = Image.open(x_path).convert('RGB')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
