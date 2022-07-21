

import sys
from model import *


# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('fi')  # first image
    sys.argv.append('li')  # last image

# Compute
pathFchamberR = "/n/groups/patel/Alan/Aging/Medical_Images/images/Heart/MRI/4chambersRaw"
imgs_FchamberR = []
f_imgs = []
for f in os.listdir(pathFchamberR):
    f_imgs.append(f)
    imgs_FchamberR.append(pathFchamberR+'/'+f)



xf, xl = 50, 170
yf, yl = 30, 150
bd1 = [np.nan for i in range(xf)] + [60 for i in range(xf,xl)] + [np.nan for i in range(xl,200)]
bd2 = [np.nan for i in range(yf)] + [140 for i in range(yf,yl)] + [np.nan for i in range(yl,200)]

for i in range(int(sys.argv[1]), int(sys.argv[2])):
    img = load_img(imgs_FchamberR[i])
    image,im2 = second_seg(img,bd1,bd2,xf,xl,yf,yl,100,75)
    final = cover(im2, img)
    im_final = array_to_img(final)

    n = count_pxls_seg(final)
    if n >= 12000 or n < 1500:
        im_final_f, y_label = metric(img, 100, 75)
        im_final_f.save('/n/groups/patel/Hamza/Images_Segmented_f/'+f_imgs[i], 'JPEG')
    else: 
        im_final.save('/n/groups/patel/Hamza/Images_Segmented_f/'+f_imgs[i], 'JPEG')
   	

    im_final.save('/n/groups/patel/Hamza/Images_Segmented_f_int/'+f_imgs[i], 'JPEG')
   



print('Done.')
sys.exit(0)
