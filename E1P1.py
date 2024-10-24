import cv2
import numpy as np
import matplotlib.pyplot as plt

#EJ 1.1
#---------------------
#a
img: np.ndarray = cv2.imread('U1/img_calculadora.tif',cv2.IMREAD_GRAYSCALE)
plt.imshow(img)
plt.show()
#print(img.shape)
print(f'Las dimensiones de la imagen son: {img.shape[0]} filas x {img.shape[1]} columnas')
print(f'El tipo de dato de la imagen es: {type(img)} y el tipo de dato de cada pixel es {img.dtype}')

#b
print(f' el valor máximo de la imagen es {img.max()} y el valor mínimo es {img.min()}')

print(np.unique(img))

#c
valores, counts_ = np.unique(img, return_counts=True)

min_counts = counts_.min()
max_counts = counts_.max()
#d
print(f'Los valores de grises que tienen la imagen son un total de: {len(valores)}')

#e
valores_menor_repetitividad = valores[counts_ == min_counts]
valores_mayor_repetitividad = valores[counts_ == max_counts]
print(f"el valor más repetido es {valores_mayor_repetitividad}, los valores menos repetidos son {valores_menor_repetitividad}")

#EJ 1.2
#f
sin_cropped: np.ndarray = img[341:427,745:860]
cos_cropped: np.ndarray = img[341:427,970:1091]
tan_cropped: np.ndarray = img[341:427, 1197:1324]
# Create subplots with 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Sin cropped image
im1 = axs[0].imshow(sin_cropped, cmap='gray')
axs[0].set_title('Sin Cropped')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_xticks([])  # Hide x-axis ticks
axs[0].set_yticks([])  # Hide y-axis ticks
plt.colorbar(im1, ax=axs[0])

# Cos cropped image
im2 = axs[1].imshow(cos_cropped, cmap='gray')
axs[1].set_title('Cos Cropped')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_xticks([])  # Hide x-axis ticks
axs[1].set_yticks([])  # Hide y-axis ticks
plt.colorbar(im2, ax=axs[1])

# Tan cropped image
im3 = axs[2].imshow(tan_cropped, cmap='gray')
axs[2].set_title('Tan Cropped')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
axs[2].set_xticks([])  # Hide x-axis ticks
axs[2].set_yticks([])  # Hide y-axis ticks
plt.colorbar(im3, ax=axs[2])

# Display the plots
plt.tight_layout()
plt.show()

#g
"""
sin_cropped: np.ndarray = img[341:427,745:860]
cos_cropped: np.ndarray = img[341:427,970:1091] # 427-341 1091-970
tan_cropped: np.ndarray = img[341:427, 1197:1324]
"""
img_replaced = img.copy()

img_replaced[341:341+tan_cropped.shape[0], 745:745+tan_cropped.shape[1]] = tan_cropped
img_replaced[341:341+sin_cropped.shape[0], 1197:1197+sin_cropped.shape[1]] = sin_cropped
plt.figure(), plt.imshow(img_replaced, cmap='gray'), plt.show(block=False)

enter_cropped: np.ndarray = img[571:630,69:406]
plt.figure(), plt.imshow(enter_cropped, cmap='gray'), plt.show(block=False)


enter_cropped_resized: np.ndarray = cv2.resize(enter_cropped, (121,86))
plt.figure(), plt.imshow(enter_cropped_resized, cmap='gray'), plt.show(block=False)

img_replaced[enter_cropped_resized.shape[0]+341:341, enter_cropped_resized.shape[1]+970:970] = enter_cropped_resized