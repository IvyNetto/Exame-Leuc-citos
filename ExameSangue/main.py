# https://www.kaggle.com/datasets/paultimothymooney/blood-cells dataset novo

import cv2 as cv

# Define a escala de mm por px
mm_por_px = 0.0075/286

img = cv.imread('teste.jpg')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, limite = cv.threshold(imgray, 127, 255, 0)
contornos, hierarquia = cv.findContours(limite, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contornos, -1, (0, 255, 0), 3)
cnt = contornos[0]

# Início cálculo do perímetro
perimetro = cv.arcLength(cnt, True)
perimetro_mm = perimetro * mm_por_px
# Fim cálculo do perímetro

# Início cálculo da área
area = cv.contourArea(cnt)
area_mm2 = area*(mm_por_px ** 2)
# Fim cálculo da área

print(f"\nPerímetro -> {perimetro_mm:.6f} mm")
print(f"\nÁrea -> {area_mm2:.6f} mm2")

# print("Probabilidade de HIV ")
# print("Probabilidade de dengue")
cv.imshow(f"Imagem Contornada P{perimetro_mm:.6f}, A{area_mm2:.6f}", img)
cv.waitKey()