import hopfield
import matplotlib.pyplot as plt
import numpy as np
from vizualization import *
import cv2


def main():
    N = 70
    img_v = cv2.imread("image2.png",cv2.IMREAD_GRAYSCALE)
    img_v = np.array(cv2.resize(img_v,[N,N]),dtype = float)
    img_v = cv2.normalize(img_v,None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
    
    img_3 = cv2.imread("image3.png",cv2.IMREAD_GRAYSCALE)
    img_3 = np.array(cv2.resize(img_3,[N,N]),dtype = float)
    img_3 = cv2.normalize(img_3,None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)

    img_4 = cv2.imread("image.png",cv2.IMREAD_GRAYSCALE)
    img_4 = np.array(cv2.resize(img_4,[N,N]),dtype = float)
    img_4 = cv2.normalize(img_4,None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)

    img_2 = cv2.imread("image4.jpg",cv2.IMREAD_GRAYSCALE)
    img_2 = np.array(cv2.resize(img_2,[N,N]),dtype = float)
    img_2 = cv2.normalize(img_2  ,None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
    N = len(img_2)
    hop = hopfield.Hopfield_NN(N*N,0.1)
    # hop.set_weights(np.array([np.array(img_3),0.995*np.array(img_v)]))#,,np.array(img_4),np.array(img_3)
    for i in range(5):
        hop.teach_one(img_v)
        print(i)
    for i in range(6):
        hop.teach_one(img_4)
        print(i)
    win = MainWindow(hop,N,N)
    win.run()
    # for i in range(5):
    #     img = hop.get_out(np.array([np.random.uniform()*2 - 1 for i in range(N*N)]))[0].reshape([N,N])
    #     print(img)
    #     plt.imshow(img)
    #     plt.show()
    # pass


if __name__ == '__main__':
    main()