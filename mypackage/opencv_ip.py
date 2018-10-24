import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class ImageProcessing:
    def __init__(self, parent = None):
        pass

    def open_img(self, file):
        img = cv2.imread(file)
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #R<->B
        return img, img_color

    def save_img(self, file, _img):
        cv2.imwrite(file, _img)

    # Grayscale
    def grayscale(self, src):
        if len(src.shape) == 3:
            return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            return src
    # Flip
    def flip(self, src):
        return cv2.flip(src, 1)

    # Affine
    def translation(self, src):
        mat = np.float32([[1, 0, -100], [0, 1, 50]])
        dst = cv2.warpAffine(src, mat, (src.shape[1], src.shape[0]))
        return dst

    # Edge Detection
    def sobelX(self, src):
        return cv2.Sobel(src, cv2.CV_8U, 1, 0, ksize=3)
    def sobelY(self, src):
        return cv2.Sobel(src, cv2.CV_8U, 0, 1, ksize=3)
    def laplacian(self, src):
        return cv2.Laplacian(src, cv2.CV_8U, ksize=3)

    # Canny Edge Detection
    def canny(self, src):
        if len(src.shape) == 3:
            img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img, 100, 200)
            edges2 = np.zeros_like(src)
            for i in (0,1,2):
                edges2[:,:,i] = edges
            dst = cv2.addWeighted(src, 1, edges2, 0.4,0)
            return dst

        else:
            edges = cv2.Canny(src, 100, 200)
            dst = cv2.addWeighted(src, 1, edges, 0.4,0)
            return dst

if __name__ == '__main__':
    file = "../data/before_set1_test//Image15.jpg"
    #ip = ImageProcessing()
    #pic,cpic = ip.open_img(file)
    #dst = ip.canny(pic)
    img = cv2.imread(file)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    imgY = imgYUV[:, :, 0]
    result = cv2.Laplacian(imgY, cv2.CV_64F)

    kernel = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
    dx = cv2.filter2D(imgY, cv2.CV_64F, kernel)
    kernel2 = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
    dy = cv2.filter2D(imgY, cv2.CV_64F, kernel2)

    dx = cv2.Sobel(imgY, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(imgY, cv2.CV_64F, 0, 1, ksize=3)
    #grad = np.sqrt(dx ** 2 + dy ** 2)

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    axes[0].imshow(imgY, cmap=cm.Greys_r, vmin=0, vmax=255)
    axes[0].set_title('Y')
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[1].imshow(dx, cmap=cm.Greys_r, vmin=-128, vmax=128)
    axes[1].set_title('dx')
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[2].imshow(dy, cmap=cm.Greys_r, vmin=-128, vmax=128)
    axes[2].set_title('dy')
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)
    print(dy)

    plt.show()
    #cv2.imshow("imgY",imgY)
    #cv2.imshow("imgY",result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()