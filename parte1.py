from convolucao import apply_kernel_RGB
import numpy as np
import cv2
import time

temple_img = cv2.imread('./templo.jpg', cv2.IMREAD_COLOR)
temple_img= cv2.resize(temple_img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA) #img, tam final da imagem, fator de escala x, fator de escala e método de interpolação

#BLUR EXAMPLE------------------------------------------------------------------
blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]]) / 9
#calculando tempo
start = time.time()
blur_temple = apply_kernel_RGB(temple_img, blur_kernel)
elapsed = time.time() - start
cv2.putText(blur_temple, f"{elapsed:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#mostra as imagens
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo com Blur', blur_temple)
cv2.waitKey()
cv2.destroyAllWindows()

#EMBOSS EXAMPLE------------------------------------------------------------------
emboss_kernel = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]])
#calculando tempo
start = time.time()
emboss_temple = apply_kernel_RGB(temple_img, emboss_kernel)
elapsed = time.time() - start
cv2.putText(emboss_temple, f"{elapsed:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#mostra as imagens
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo com Emboss', emboss_temple)
cv2.waitKey()
cv2.destroyAllWindows()

#SHARPEN EXAMPLE------------------------------------------------------------------
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]])
#calculando tempo
start = time.time()
sharpen_temple = apply_kernel_RGB(temple_img, sharpen_kernel)
elapsed = time.time() - start
cv2.putText(sharpen_temple, f"{elapsed:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#mostra as imagens
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo com Sharpen', sharpen_temple)
cv2.waitKey()
cv2.destroyAllWindows()

#MOTION BLUR EXAMPLE------------------------------------------------------------------
motion_blur_kernel = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]) / 3
#calculando tempo
start = time.time()
motion_blur_temple = apply_kernel_RGB(temple_img, motion_blur_kernel)
elapsed = time.time() - start
cv2.putText(motion_blur_temple, f"{elapsed:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#mostra as imagens
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo com Motion Blur', motion_blur_temple)
cv2.waitKey()
cv2.destroyAllWindows()

#BORDER DETECTION EXAMPLE------------------------------------------------------------------
border_detection_kernel = np.array([
    [-1, -1, -1], 
    [-1,  8, -1], 
    [-1, -1, -1]])
#calculando tempo
start = time.time()
laplacian_temple = apply_kernel_RGB(temple_img, border_detection_kernel)
elapsed = time.time() - start
cv2.putText(laplacian_temple, f"{elapsed:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#mostra as imagens
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo Laplacian', laplacian_temple)
cv2.waitKey()
cv2.destroyAllWindows()