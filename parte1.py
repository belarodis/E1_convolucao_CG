from convolucao import apply_kernel_RGB
import numpy as np
import cv2

temple_img = cv2.imread('./templo.jpg', cv2.IMREAD_COLOR)
temple_img= cv2.resize(temple_img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA) #img, tam final da imagem, fator de escala x, fator de escala e método de interpolação

#BLUR EXAMPLE
blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]]) / 9
blur_temple = apply_kernel_RGB(temple_img, blur_kernel)
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo com Blur', blur_temple)
cv2.waitKey()
cv2.destroyAllWindows()

#EMBOSS EXAMPLE
emboss_kernel = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]])
emboss_temple = apply_kernel_RGB(temple_img, emboss_kernel)
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo com Emboss', emboss_temple)
cv2.waitKey()
cv2.destroyAllWindows()

#SHARPEN EXAMPLE
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]])
sharpen_temple = apply_kernel_RGB(temple_img, sharpen_kernel)
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo com Sharpen', sharpen_temple)
cv2.waitKey()
cv2.destroyAllWindows()

#MOTION BLUR EXAMPLE
motion_blur_kernel = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]) / 3
motion_blur_temple = apply_kernel_RGB(temple_img, motion_blur_kernel)
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo com Motion Blur', motion_blur_temple)
cv2.waitKey()
cv2.destroyAllWindows()

#LAPLACIAN EXAMPLE
laplacian_kernel = np.array([
    [-1, -1, -1], 
    [-1,  8, -1], 
    [-1, -1, -1]])
laplacian_temple = apply_kernel_RGB(temple_img, laplacian_kernel)
cv2.imshow('Templo sem filtro', temple_img)
cv2.imshow('Templo Laplacian', laplacian_temple)
cv2.waitKey()
cv2.destroyAllWindows()
