# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

color = ['b', 'g', 'r']


def getColorHist(image):
    """
    This function produces channelwise histogram distribution.
    """
    if len(image.shape) == 3:
        hists = []
        for i, col in enumerate(color):
            hists.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
        return hists
    else:
        hists = []
        i = 0
        hists.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
        return hists


def applyCLAHE(image, display: bool = False):
    """
    CLAHE implementation.
    image: 3-channel grayscale image.
    returns the image applied CLAHE.
    display: if True, it will display the input and output.
    """
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw)  # + 30
    final_img = np.stack((final_img,) * 3, axis=-1)

    # Ordinary thresholding the same image
    _, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)

    if display:
        fig = plt.figure(figsize=(9, 3), dpi=300)
        rows, cols = 1, 3

        # Display the original image
        fig.add_subplot(rows, cols, 1)
        plt.imshow(image, cmap=plt.cm.bone);
        plt.axis('off')
        plt.title("Input Image")

        # Display the thresholded image
        fig.add_subplot(rows, cols, 2)
        plt.imshow(ordinary_img, cmap=plt.cm.gray);
        plt.axis('off')
        plt.title("Binary Thresholded Image")

        # Display the CLAHE processed image
        fig.add_subplot(rows, cols, 3)
        plt.imshow(final_img, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title("CLAHE Image")
        plt.show()
    # print(final_img.shape)
    return final_img


def applyHistogramEqualization(image, display: bool = False):
    """
    Applies the histogram equalization to a 3 channel grayscale image.
    If display is true, it wills show the comparison.
    """
    # https://123machinelearn.wordpress.com/2017/12/25/image-enhancement-using-high-frequency-emphasis-filtering-and-histogram-equalization/
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)

    hist, bins = np.histogram(image_bw.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_norm = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img_enhanced = cdf[image]

    if display:
        fig = plt.figure(figsize=(10, 5), dpi=300)
        rows, cols = 2, 2

        # Display the original image
        fig.add_subplot(rows, cols, 1)
        plt.imshow(image, cmap=plt.cm.gray);
        plt.axis('off');
        plt.title("Input Image")

        hists = getColorHist(image)
        fig.add_subplot(rows, cols, 3)
        plt.plot(hists[0], color="b")
        plt.plot(hists[1], color="g")
        plt.plot(hists[2], color="r")
        plt.xlim([0, 256])
        plt.title("Original Histogram")

        # Display the thresholded image
        fig.add_subplot(rows, cols, 2)
        plt.imshow(img_enhanced, cmap=plt.cm.gray);plt.axis('off')
        plt.title("Histogram Equalized Image")

        hists = getColorHist(img_enhanced)
        fig.add_subplot(rows, cols, 4)
        plt.plot(hists[0], color="b")
        plt.plot(hists[1], color="g")
        plt.plot(hists[2], color="r")
        plt.xlim([0, 256])
        plt.title("Equalized Histogram")

        plt.show()

        plt.plot

    print("final shape", img_enhanced.shape)
    return img_enhanced


def applyHFEFilter(image, display: bool = False):
    """
    This function applies the High Frequency Emphasis Filter on an Image.
    if display is true, it will show the comparison before and after the filter operation.
    """
    # https://123machinelearn.wordpress.com/2017/12/25/image-enhancement-using-high-frequency-emphasis-filtering-and-histogram-equalization/
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    npFFT = np.fft.fft2(image_bw)
    npFFTS = np.fft.fftshift(npFFT)

    # High-pass Gaussian filter
    (P, Q) = npFFTS.shape
    H = np.zeros((P, Q))
    D0 = 40
    for u in range(P):
        for v in range(Q):
            H[u, v] = 1.0 - np.exp(- ((u - P / 2.0) ** 2 + (v - Q / 2.0) ** 2) / (2 * (D0 ** 2)))
    k1 = 0.5;
    k2 = 0.80
    HFEfilt = k1 + k2 * H  # Apply High-frequency emphasis

    # Apply HFE filter to FFT of original image
    HFE = HFEfilt * npFFTS

    """
    Implement 2D-FFT algorithm

    Input : Input Image
    Output : 2D-FFT of input image
    """

    def fft2d(image):
        # 1) compute 1d-fft on columns
        fftcols = np.array([np.fft.fft(row) for row in image]).transpose()

        # 2) next, compute 1d-fft on in the opposite direction (for each row) on the resulting values
        return np.array([np.fft.fft(row) for row in fftcols]).transpose()

    # Perform IFFT (implemented here using the np.fft function)
    HFEfinal = (np.conjugate(fft2d(np.conjugate(HFE)))) / (P * Q)

    output = np.sqrt((HFEfinal.real) ** 2 + (HFEfinal.imag) ** 2)
    output = np.array(np.stack((output,) * 3, axis=-1), dtype=np.uint8)

    if display:
        fig = plt.figure(figsize=(10, 5), dpi=300)
        rows, cols = 1, 2

        # Display the original image
        fig.add_subplot(rows, cols, 1)
        plt.imshow(image, cmap=plt.cm.gray);
        plt.axis('off');
        plt.title("Input Image")

        # Display the thresholded image
        fig.add_subplot(rows, cols, 2)
        plt.imshow(output, cmap=plt.cm.gray);
        plt.axis('off')
        plt.title("HF Enhanced Image")
        plt.show()

    print(output.shape)
    return output


if __name__ == "__main__":
    img = cv2.imread("/home/sonymd/Downloads/ChestXray14Data/subset/00000003_005.png")
    applyCLAHE(img, display=True)
    #

    applyHistogramEqualization(img, display=True)

    applyHFEFilter(img, display=True)

    # applyHistogramEqualization(output, display=True)
