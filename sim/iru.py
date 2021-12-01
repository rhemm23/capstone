from math import radians, cos, sin

def iru(sub_image, rnn_out):
  theta = 0
  for i in range(36):
    if rnn_out[i]:
      theta = radians(i * 10)
      break
  res_image = [0 for _ in range(400)]
  for y in range(20):
    for x in range(20):
      nx = int(cos(theta) * x - sin(theta) * y)
      ny = int(sin(theta) * x + cos(theta) * y)
      if nx >= 0 and ny >= 0 and nx < 20 and ny < 20:
        res_image[(ny * 20) + nx] = sub_image[(y * 20) + x]
  for i in range(400):
    sub_image[i] = res_image[i]
