import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import IPython.display as ipd
from tqdm import tqdm
import subprocess

plt.style.use('ggplot')

# Convertir el video de entrada a formato mp4
input_file = 'entrada.mov'
subprocess.run(['ffmpeg',
                '-i',
                input_file,
                '-qscale',
                '0',
                'salida.mp4',
                '-loglevel',
                'quiet'])

# Mostrar el video de salida
ipd.Video('salida.mp4', width=700)

# Cargar el video capturado
cap = cv2.VideoCapture('salida.mp4')

# Obtener información del video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)

cap.release()

print(f'Total de fotogramas: {total_frames}')
print(f'Altura: {height}, Ancho: {width}')
print(f'FPS: {fps:0.2f}')

# Mostrar la primera imagen del video
cap = cv2.VideoCapture('salida.mp4')
ret, img = cap.read()
print(f'Retornado {ret} e imagen de forma {img.shape}')

## Función auxiliar para mostrar imágenes de OpenCV en el notebook
def display_cv2_img(img, figsize=(10, 10)):
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_)
    ax.axis("off")

display_cv2_img(img)

cap.release()

# Mostrar una cuadrícula de imágenes del video
fig, axs = plt.subplots(5, 5, figsize=(30, 20))
axs = axs.flatten()

cap = cv2.VideoCapture("salida.mp4")

img_idx = 0
for frame in range(total_frames):
    ret, img = cap.read()
    if ret == False:
        break
    if frame % 100 == 0:
        axs[img_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[img_idx].set_title(f'Frame: {frame}')
        axs[img_idx].axis('off')
        img_idx += 1
        if img_idx >= 25:
            break

plt.tight_layout()
plt.show()
cap.release()

# Leer las etiquetas del video
labels = pd.read_csv('../input/driving-video-with-object-tracking/mot_labels.csv', low_memory=False)
video_labels = labels.query('videoName == "026c7465-309f6d33"').reset_index(drop=True).copy()
video_labels["video_frame"] = (video_labels["frameIndex"] * 11.9).round().astype("int")

video_labels["category"].value_counts()

# Mostrar la imagen en el fotograma 1035 con las etiquetas
cap = cv2.VideoCapture("salida.mp4")

frame = 0
while frame < 1035:
    ret, img = cap.read()
    if ret == False:
        break
    frame += 1

cap.release()

display_cv2_img(img)

img_example = img.copy()
frame_labels = video_labels[video_labels["video_frame"] == 1035]