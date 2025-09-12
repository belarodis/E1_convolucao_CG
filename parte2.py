import numpy as np
import cv2

gbr_frutas_e_cebola = cv2.imread('./imagens/6frutas1vegetal.png')
gs_frutas_e_cebola = cv2.cvtColor(gbr_frutas_e_cebola, cv2.COLOR_BGR2GRAY)

#ler imagem e converter p/ grayscale
gbr_frutas = cv2.imread('./imagens/5frutas.png')
gs_frutas = cv2.cvtColor(gbr_frutas, cv2.COLOR_BGR2GRAY)
kernel = np.array([
	[2, 2, 1],
	[2, 2, 1],
	[1, 1, 0]
], dtype=np.uint8)

#aplicar threshold binário
_, binary_frutas = cv2.threshold(gs_frutas, 198, 255, cv2.THRESH_BINARY_INV)
#_, binary_frutas = cv2.threshold(gs_frutas_e_cebola, 160, 255, cv2.THRESH_BINARY)

#limpando imagem p/ detecção de componentes
for i in range(10):
    binary_frutas = cv2.morphologyEx(binary_frutas, cv2.MORPH_CLOSE, kernel)

for i in range(5):
    binary_frutas = cv2.erode(binary_frutas, kernel, iterations=2)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frutas)

#contorno
contours, _ = cv2.findContours(binary_frutas, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area < 500:
        continue

    #cv2.drawContours(gbr_frutas, [c], -1, (0,255,0), 2)
    cv2.rectangle(gbr_frutas, (x, y), (x+w, y+h), (0,255,0), 2)

    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        cv2.circle(gbr_frutas, (cx, cy), 5, (0,0,255), -1)


print("Número de componentes:", len(contours))

cv2.imshow('Frutas dev', binary_frutas)
cv2.imshow('Frutas contorno e keypoints', gbr_frutas)
cv2.waitKey()
cv2.destroyAllWindows()