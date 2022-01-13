from PIL import Image
import cv2
import scipy.ndimage as ndimage
import os

folder = 'C:/Users/jl_sa/Desktop/images and landmarks/UTKFaceNew'
names = [f for f in os.listdir(folder)]
new_folder = 'C:/Users/jl_sa/Desktop/images and landmarks/UTKFace_reshape'
path = os.path.isdir(new_folder)
if not path:
    os.makedirs(new_folder)
    print('Created folder in {0}'.format(new_folder))
else:
    print('There is a folder already named {0}'.format(new_folder))

img = 0
x=0
final = []
pth = 0
for i in names:
    imagem = folder+'/'+i
    x = Image.open(imagem)
    img = ndimage.rotate(x,90,reshape=True)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pth = new_folder + '/'+i
    save = cv2.imwrite(pth,img)
    x.close()
print('[INFO] Everything is done, check your folder now!')