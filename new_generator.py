import numpy as np
import matplotlib.pyplot as plt
from time import time
import skimage.io as skio
from scipy.ndimage import gaussian_filter
import skimage
import argparse 
import os
from skimage.transform import pyramid_reduce
## SETTING PARAMETERS
parser = argparse.ArgumentParser(description='Dead leaves image generator')
## general arguments
parser.add_argument('--size', type=int, default=1999, metavar='S',
                    help='size of the square side in the image')
parser.add_argument('--color_mode', type=str, default="color", metavar='C',
                    help='RGB if color, greyscale if grey')
parser.add_argument('--alpha', type=float, default=3.0, metavar='A',
                    help='exponent of the distribution of the radius')
parser.add_argument('--rmin', type=int, default=1, metavar='RMIN',
                    help='minimal size of the radius')
parser.add_argument('--rmax', type=int, default=1999, metavar='RMAX',
                    help='maximal size of the radius')
# parser.add_argument('--anti_aliasing', type=int, default=2, metavar='AA',
#                     help='anti aliasing mode 0: no A-A, 1: gaussian filtering 2: mean filter on the neighboors')
parser.add_argument('--number', type=int, default=10, metavar='N',
                    help='Number of image to generate with those parameters')
parser.add_argument('--downscaling','--ds',action='store_true',\
                    help="dowscales the image with a factor 5")
args = parser.parse_args()

#loading params
width = args.size
length = args.size
r_min = args.rmin
r_max = args.rmax
t0 = time()
disk_dict = np.load('dict.npy',allow_pickle=True)
def hor_grad(I,alpha,n,angle):
    result = np.zeros((int(n+0.4*n),int(n+0.4*n)))
    for k in range(int(n+0.4*n)):
        result[:,k] = (I-alpha) + (k/n)*(2*alpha)
    return(skimage.transform.rotate(result,angle)[int(0.2*n):n+int(0.2*n),int(0.2*n):n+int(0.2*n)])

def dead_leaves_image(alpha):
    ind = np.random.randint(0,4000)
    img_source = skio.imread("/home/raphael/Travail/ffdnet/datasets/data/{:05d}.bmp".format((ind)))
    print("{:05d}".format(ind))
    w,l = img_source.shape[0],img_source.shape[1]
    img = np.ones((width+2*r_max+1,length+2*r_max+1,3))
    binary_image = np.ones((width+2*r_max+1,length+2*r_max+1))
    disk_values = []
    k = 0
    # interval for the random radius
    vamin = 1/(r_max**(alpha-1))
    vamax = 1/(r_min**(alpha-1))
    n = width*length
    n_grad = 0
    t0 = time()
    while n >100:
        #get the random values and store them

        # defining the random radius
        r = vamin + (vamax-vamin)*np.random.random()
        r = int(1/(r**(1./(alpha-1))))


        color = (1./255)*img_source[np.random.randint(0,w),np.random.randint(0,l),:]
        pos = [np.random.randint(0,width),np.random.randint(0,length)]
        disk_values.append([r,pos,color])
        #update the binary mask
        # L = np.arange(-r,r + 1)
        # X, Y = np.meshgrid(L, L)
        
        # disk_1d = np.array((X ** 2 + Y ** 2) <= r ** 2)
        disk_1d = disk_dict[()][str(r)]
        disk_mask = np.zeros((2*r+1,2*r+1,3))

        disk_mask_1d = binary_image[r_max+pos[0]-r:r_max+1+pos[0]+r,r_max-r+pos[1]:r_max+pos[1]+r+1].copy()
        disk_mask_1d *=  disk_1d
        disk_mask= np.repeat(disk_mask_1d[:, :, np.newaxis], 3, axis=2)
        # add the color value to the disk
        disk_mask_grad = disk_mask.copy()
        
        if r>200:
            x = np.random.random()
            if x>0.3:
                print(k)
                n_grad+=1
                angle = np.random.randint(0,360)
                d = hor_grad(0.5,0.3,2*r+1,angle)
                disk_mask_grad*= np.repeat(d[:, :, np.newaxis], 3, axis=2)
            
        disk = color*disk_mask_grad
        # add the disk at the right place
        img[r_max+pos[0]-r:r_max+1+pos[0]+r,r_max-r+pos[1]:r_max+pos[1]+r+1,:]*=1-disk_mask
        img[r_max+pos[0]-r:r_max+1+pos[0]+r,r_max-r+pos[1]:r_max+pos[1]+r+1,:]+=disk
        binary_image[r_max+pos[0]-r:r_max+1+pos[0]+r,r_max-r+pos[1]:r_max+pos[1]+r+1]*=1-disk_mask_1d       
        
        #update counter
        #n = binary_image[r_max:-r_max,r_max:-r_max].sum()
        k+=1

        if k%2000 ==0:
            print(time()-t0)
            print("number of disks : {:07d}".format(k))
        if k%50000 ==0:
            n = binary_image[r_max:-r_max,r_max:-r_max].sum()
            print("percentage covered :{}%".format((100*n)/(width*length)))
    print(n_grad)
    return(img[r_max:-r_max,r_max:-r_max,:])

direct = 'datasets/dead_leaves_big_alpha_{}/'.format(args.alpha)
if not os.path.exists(direct):
    os.makedirs(direct)
for i in range(args.number):
    image = dead_leaves_image(args.alpha)
    if args.downscaling:
        img = pyramid_reduce(image,5)
        skio.imsave('datasets/dead_leaves_big_alpha_{}/im_ds_{:02d}.png'.format(args.alpha,i), img)
    else:
        skio.imsave('datasets/dead_leaves_big_alpha_{}/im_{:02d}.png'.format(args.alpha,i), image)
