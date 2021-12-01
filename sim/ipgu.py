def ipgu(image):
  sub_images = []
  for y in range(0, 290, 10):
    for x in range(0, 290, 10):
      sub_image = []
      for i in range(20):
        row = []
        for j in range(20):
          row.append(image[y + i][x + j])
        sub_image.append(row)
      sub_images.append(sub_image)
  return sub_images
