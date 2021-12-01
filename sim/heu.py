def heu(sub_image):
  cnt = [0 for _ in range(256)]
  for pixel in sub_image:
    cnt[pixel] += 1
  cdf = [cnt[0]]
  for i in range(1, 256):
    cdf.append(cdf[-1] + cnt[i])
  for i in range(400):
    sub_image[i] = int(cdf[sub_image[i]] * 256 / 400)
