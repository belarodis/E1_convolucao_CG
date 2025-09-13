import numpy as np;
import cv2;

img_path = './imagens/frog.jpg'
image_grayscale = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image_RGB = cv2.imread(img_path, cv2.IMREAD_COLOR)

blur_kernel = np.ones((7,7)) / 49 #blurr

def apply_kernel_grayscale(image, kernel):
    h_image, w_image = image.shape 
    h_kernel, w_kernel = kernel.shape
    middle_h_kernel = h_kernel//2 
    middle_w_kernel = w_kernel//2

    new_array_image = np.array(image)
    zeros_image = np.zeros_like(new_array_image)

    for row in range(middle_h_kernel, h_image - middle_h_kernel):
        for column in range(middle_w_kernel, w_image - middle_w_kernel):
            sum = 0
            for row_kernel in range(h_kernel):
                for column_kernel in range(w_kernel):
                    pixel = new_array_image[row + row_kernel - middle_h_kernel, column + column_kernel - middle_w_kernel]
                    value_kernel = kernel[row_kernel, column_kernel]
                    sum += pixel * value_kernel

            zeros_image[row, column] = np.clip(sum, 0, 255)

    return zeros_image.astype(np.uint8)

def apply_kernel_RGB(image, kernel):
    h_image, w_image, c_image = image.shape 
    h_kernel, w_kernel = kernel.shape
    middle_h_kernel = h_kernel//2
    middle_w_kernel = w_kernel//2

    new_array_image = np.array(image)
    zeros_image = np.zeros_like(new_array_image)

    for row in range(middle_h_kernel, h_image - middle_h_kernel):
        for column in range(middle_w_kernel, w_image - middle_w_kernel):
            for color_canal in range(c_image):
                sum = 0
                for row_kernel in range(h_kernel):
                    for column_kernel in range(w_kernel):
                        pixel = new_array_image[row + row_kernel - middle_h_kernel, column + column_kernel - middle_w_kernel, color_canal]
                        value_kernel = kernel[row_kernel, column_kernel]
                        sum += pixel * value_kernel

                zeros_image[row, column, color_canal] = np.clip(sum, 0, 255)

    return zeros_image.astype(np.uint8)

if __name__ == "__main__":
    blur_kernel_image_grayscale = apply_kernel_grayscale(image_grayscale, blur_kernel)
    cv2.imshow('Original grayscale image', image_grayscale)
    cv2.imshow('Grayscale image with blur', blur_kernel_image_grayscale)

    blur_kernel_image_RGB = apply_kernel_RGB(image_RGB, blur_kernel)
    cv2.imshow('Original RGB image', image_RGB)
    cv2.imshow('RGB image with blur', blur_kernel_image_RGB)

    cv2.waitKey()
    cv2.destroyAllWindows()