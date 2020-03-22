import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
im = cv2.imread('./images/train/samoyed/0312.jpg')
gray_image = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
smaller_image = cv2.resize(im, (264,264)) #, interpolation='linear')

#print("size:"+type(im.shape))

print("size:"+str(im.shape))
print(type(im.shape))

#plt.figure()
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(im)
axarr[0,1].imshow(smaller_image)
axarr[1,0].imshow(gray_image)
plt.show()

i =3


