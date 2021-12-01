
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