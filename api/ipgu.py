
SCALES = {
  300: 240,
  240: 180,
  180: 120,
  120: 60,
  60: 20
}

def ipgu(image):
  size = len(image)
  sub_images_bb = []
  sub_images = []
  while True:
    for y in range(0, size - 10, 10):
      for x in range(0, size - 10, 10):
        sub_image = []
        for i in range(20):
          for j in range(20):
            sub_image.append(image[y + i][x + j])
        rscl = 300 / size
        sub_images_bb.append([
          int(rscl * x),
          int(rscl * y),
          int(rscl * (x + 20)),
          int(rscl * (y + 20))
        ])
        sub_images.append(sub_image)
    if size in SCALES:
      scale = SCALES[size] / size
      next_img = [[0 for _ in range(SCALES[size])] for _ in range(SCALES[size])]
      next_img_cnts = [[0 for _ in range(SCALES[size])] for _ in range(SCALES[size])]
      for y in range(size):
        for x in range(size):
          ny = int(y * scale)
          nx = int(x * scale)
          next_img_cnts[ny][nx] += 1
          next_img[ny][nx] += image[y][x]
      for y in range(SCALES[size]):
        for x in range(SCALES[size]):
          next_img[y][x] = int(next_img[y][x] / next_img_cnts[y][x])
      image = next_img
      size = SCALES[size]
    else:
      break
  return sub_images, sub_images_bb
