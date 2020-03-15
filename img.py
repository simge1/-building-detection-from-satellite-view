#görüntünün okunması ve griye çevrilmesi
image = cv2.imread('resim.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#görüntünün geliştirilmesi
kernel = np.ones((3,3), np.uint8)
img_dilate = cv2.dilate(gray, kernel, iterations=1)

#IGV
depth = cv2.CV_16S
kernel_size = 5
source = image
source2 = cv2.medianBlur(source,3)
source_gray = cv2.cvtColor(source2, cv2.COLOR_BGR2GRAY)
destination = cv2.Laplacian(source_gray,depth, kernel_size)
abs_dst = cv2.convertScaleAbs(destination)

#çekirdek nokta tespiti ve kümeleme
Z = image.reshape(-3,3)
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
seed = center[label.flatten()]
cluster = seed.reshape((image.shape))

#binarizasyon teniği
ret,thresh=cv2.threshold(cluster,125,255,cv2.THRESH_BINARY)

#morfoloji
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

#erosion filtresi
erosion = cv2.erode(morph,kernel,iterations = 1)

#segmentasyon tekniği
BGS = cv2.createBackgroundSubtractorMOG2()
gmask = BGS.apply(erosion)
ret,otsu=cv2.threshold(gmask,125,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("result",otsu)
