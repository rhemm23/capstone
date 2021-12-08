from math import radians, cos, sin

import numpy as np

def iru(sub_image, rnn_out):
  theta = 0
  for i in range(36):
    if rnn_out[i]:
      theta = radians(i * 10)
      break
  res_image = [0 for _ in range(400)]
  for y in range(20):
    for x in range(20):
      nx = int(cos(theta) * (x - 10) - sin(theta) * (y - 10)) + 10
      ny = int(sin(theta) * (x - 10) + cos(theta) * (y - 10)) + 10
      if nx >= 0 and ny >= 0 and nx < 20 and ny < 20:
        res_image[(ny * 20) + nx] = sub_image[(y * 20) + x]
  for i in range(400):
    sub_image[i] = res_image[i]

if __name__ == '__main__':
  rnn_out = [False for _ in range(36)]
  rnn_out[35] = True
  data = None
  with open("./image.bin", "rb") as file:
    data = np.frombuffer(file.read(), dtype=np.uint8).tolist()
  iru(data, rnn_out)
  with open("./result.bin", "wb+") as result:
    result.write(np.array(data, dtype=np.uint8).tobytes())
