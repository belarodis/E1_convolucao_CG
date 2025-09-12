import numpy as np
import cv2

#5 FRUTAS ----------------------------------------------------------------------
#ler imagem e converter p/ grayscale
gbr_frutas = cv2.imread('./imagens/5frutas.png')
gs_frutas = cv2.cvtColor(gbr_frutas, cv2.COLOR_BGR2GRAY)

#aplicar threshold binário
_, binary_frutas = cv2.threshold(gs_frutas, 190, 255, cv2.THRESH_BINARY_INV)

#limpando imagem p/ detecção de componentes
primary_kernel = np.array([
	[1, 1, 1],
	[0, 1, 0],
	[0, 1, 0]
], dtype=np.uint8)
binary_frutas = cv2.morphologyEx(binary_frutas, cv2.MORPH_CLOSE, primary_kernel)
binary_frutas = cv2.dilate(binary_frutas, primary_kernel, iterations=4)
binary_frutas = cv2.morphologyEx(binary_frutas, cv2.MORPH_CLOSE, primary_kernel)
binary_frutas = cv2.erode(binary_frutas, primary_kernel, iterations=2)

secondary_kernel = np.array([
	[0, 0, 0],
	[0, 0, 0],
	[0, 1, 0]
], dtype=np.uint8)
binary_frutas = cv2.morphologyEx(binary_frutas, cv2.MORPH_CLOSE, secondary_kernel)
binary_frutas = cv2.erode(binary_frutas, secondary_kernel, iterations=4)
binary_frutas = cv2.morphologyEx(binary_frutas, cv2.MORPH_CLOSE, secondary_kernel)

#parametros
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frutas)

#desenhando bounding box, keypoints e texto de área
count = 0
for c in range(1, num_labels):
    x, y, w, h, area = stats[c]
    cx, cy = centroids[c]

    if area < 200 or w > 500:
        continue

    count += 1
    cv2.circle(gbr_frutas, (int(cx), int(cy)), 4, (0,0,255), -1)
    cv2.rectangle(gbr_frutas, (x, y), (x+w, y+h), (0,255,0), 2)

    text = f"Area: {area}"
    text_pos = (x, y - 10 if y - 10 > 10 else y + 20)
    cv2.putText(gbr_frutas, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

print("Componentes detectados: " + str(count))
cv2.imshow('Frutas binary', binary_frutas)
cv2.imshow('Frutas com bounding box e keypoints', gbr_frutas)
cv2.waitKey()
cv2.destroyAllWindows()

#6 FRUTAS E UMA CEBOLA ---------------------------------------------------------
gbr_frutas_e_cebola = cv2.imread('./imagens/6frutas1vegetal.png')
gs_frutas_e_cebola = cv2.cvtColor(gbr_frutas_e_cebola, cv2.COLOR_BGR2GRAY)

#aplicar threshold binário

#limpando imagem p/ detecção de componentes

#parametros

#desenhando bounding box, keypoints e texto de área

cv2.waitKey()
cv2.destroyAllWindows()