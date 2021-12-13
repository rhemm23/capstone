from PIL import Image

import numpy as np

CONSTRAST = 128

def bcau(sub_image):
  factor = (259 * (255 + CONSTRAST)) / (255 * (259 - CONSTRAST))
  for i in range(400):
    temp = int(factor * (sub_image[i] - CONSTRAST) + CONSTRAST)
    if temp > 255:
      temp = 255
    elif temp < 0:
      temp = 0
    sub_image[i] = temp

def heu(sub_image):
  cnt = [0 for _ in range(256)]
  for pixel in sub_image:
    cnt[pixel] += 1
  cdf = [cnt[0]]
  for i in range(1, 256):
    cdf.append(cdf[-1] + cnt[i])
  for i in range(400):
    sub_image[i] = int(cdf[sub_image[i]] * 256 / 400)

def bcau(sub_image):
  for y in range(5):
    for x in range(5):
      avg = 0
      for i in range(4):
        for j in range(4):
          avg += sub_image[(y * 4) + i][(x * 4) + j]
      avg = avg // 16
      for i in range(4):
        for j in range(4):
          value = sub_image[(y * 4) + i][(x * 4) + j]
          new_value = (value + 32) if (value > avg) else (value - 32)
          new_value = max(0, min(255, new_value))
          sub_image[(y * 4) + i][(x * 4) + j] = new_value

pixels = None
with open('./image.bin', 'rb') as file:
  pixels = np.frombuffer(file.read(), dtype=np.uint8).tolist()
# arr = np.array(pixels, dtype=np.uint8).reshape((20, 20))
# image = Image.fromarray(arr, 'L').resize((100, 100))
# image.show()
# bcau(pixels)
# heu(pixels)
arr2 = np.array(pixels, dtype=np.uint8).reshape((20, 20))
image1 = Image.fromarray(arr2, 'L').resize((100, 100))
image1.show()
