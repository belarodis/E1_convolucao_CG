import numpy as np
import cv2

#5 FRUTAS ----------------------------------------------------------------------
#ler imagem e converter p/ grayscale
bgr_frutas = cv2.imread('./imagens/5frutas.png')
gs_frutas = cv2.cvtColor(bgr_frutas, cv2.COLOR_BGR2GRAY)

#aplicar threshold binário
_, binary_frutas = cv2.threshold(gs_frutas, 190, 255, cv2.THRESH_BINARY_INV)

# limpando imagem p/ detecção de componentes
primary_kernel = np.array([
	[1, 1, 1],
	[0, 1, 0],
	[0, 1, 0]
], dtype=np.uint8)
binary_frutas = cv2.morphologyEx(binary_frutas, cv2.MORPH_CLOSE, primary_kernel)
binary_frutas = cv2.dilate(binary_frutas, primary_kernel, iterations=4)
binary_frutas = cv2.morphologyEx(binary_frutas, cv2.MORPH_CLOSE, primary_kernel)
binary_frutas = cv2.erode(binary_frutas, primary_kernel, iterations=2)

# subindo a máscara para centralizar melhor as frutas na imagem (por conta das suas sombras)
secondary_kernel = np.array([
	[0, 0, 0],
	[0, 0, 0],
	[0, 1, 0]
], dtype=np.uint8)
binary_frutas = cv2.morphologyEx(binary_frutas, cv2.MORPH_CLOSE, secondary_kernel)
binary_frutas = cv2.erode(binary_frutas, primary_kernel, iterations=2)
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
    cv2.circle(bgr_frutas, (int(cx), int(cy)), 4, (0,0,255), -1)
    cv2.rectangle(bgr_frutas, (x, y), (x+w, y+h), (0,255,0), 2)

    text = f"Area: {area}px"
    text_pos = (x, y - 10 if y - 10 > 10 else y + 20)
    cv2.putText(bgr_frutas, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

print("Componentes detectados: " + str(count))
cv2.imshow('Frutas binary', binary_frutas)
cv2.imshow('Frutas com bounding box e keypoints', bgr_frutas)
cv2.waitKey()
cv2.destroyAllWindows()

#6 FRUTAS E UMA CEBOLA ---------------------------------------------------------
bgr_frutas_e_cebola = cv2.imread('./imagens/6frutas1vegetal.png')
gs_frutas_e_cebola = cv2.cvtColor(bgr_frutas_e_cebola, cv2.COLOR_BGR2GRAY)

#aplicar threshold binário
_, binary_frutas_e_cebola = cv2.threshold(gs_frutas_e_cebola, 145, 255, cv2.THRESH_BINARY_INV)

#limpando imagem p/ detecção de componentes
primary_kernel = np.array([
	[1, 1, 1],
	[0, 1, 0],
	[0, 1, 0]
], dtype=np.uint8)
binary_frutas_e_cebola = cv2.morphologyEx(binary_frutas_e_cebola, cv2.MORPH_CLOSE, primary_kernel)
binary_frutas_e_cebola = cv2.erode(binary_frutas_e_cebola, primary_kernel, iterations=2)
binary_frutas_e_cebola = cv2.morphologyEx(binary_frutas_e_cebola, cv2.MORPH_CLOSE, primary_kernel)

#parametros
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frutas_e_cebola)

#desenhando bounding box, keypoints e texto de área
count = 0
for c in range(1, num_labels):
    x, y, w, h, area = stats[c]
    cx, cy = centroids[c]
    
    """ #tentativa de redefinir os parametros da cebola
    if area == 240:
        cebola_x = x
        cebola_y = y
    if area == 4018:
        x = cebola_x
        y = cebola_y
        cx = ((x + w) - x) / 2 + x
        cy = ((y + h) - y) / 2 + y """

    if area < 2000:
        continue

    count += 1
    cv2.circle(bgr_frutas_e_cebola, (int(cx), int(cy)), 4, (0,0,255), -1)
    cv2.rectangle(bgr_frutas_e_cebola, (x, y), (x+w, y+h), (0,255,0), 2)

    text = f"Area: {area}px"
    text_pos = (x, y - 10 if y - 10 > 10 else y + 20)
    cv2.putText(bgr_frutas_e_cebola, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

cv2.imshow('Frutas binary', binary_frutas_e_cebola)
cv2.imshow('Frutas com bounding box e keypoints', bgr_frutas_e_cebola)
cv2.waitKey()
cv2.destroyAllWindows()