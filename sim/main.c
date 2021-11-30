#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define IMAGE_SIZE 90000

typedef struct image {
  int size;
  uint8_t pixels[300][300];
} image_t;

typedef struct sub_image {
  uint8_t pixels[20][20];
} sub_image_t;

static void exit_with_error(char *error) {
  fprintf(stderr, "Error: %s\n", error);
  exit(EXIT_FAILURE);
}

static void heu(sub_image_t *sub_image) {
  uint32_t cnt[256];
  uint32_t cdf[256];
  for (int i = 0; i < 256; i++) {
    cnt[i] = 0;
  }
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 20; j++) {
      cnt[sub_image->pixels[i][j]]++;
    }
  }
  cdf[0] = cnt[0];
  for (int i = 1; i < 256; i++) {
    cdf[i] = cdf[i - 1] + cnt[i];
  }
  for (int i = 0; i < 20; i++) {
    for (int j = 0; j < 20; j++) {
      sub_image->pixels[i][j] = (uint8_t)roundf(((float)cnt[sub_image->pixels[i][j]] / (float)400) * (float)256);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    exit_with_error("Expected one argument, a filepath to image binary");
  }
  FILE *file = fopen(argv[1], "rb");
  if (file == NULL) {
    exit_with_error("Could not open image file");
  }
  fseek(file, 0, SEEK_END);
  if (ftell(file) != IMAGE_SIZE) {
    exit_with_error("Invalid image file");
  }
  rewind(file);
  uint8_t *buffer = (uint8_t*)malloc(IMAGE_SIZE);
  if (fread(buffer, sizeof(uint8_t), IMAGE_SIZE, file) != IMAGE_SIZE) {
    exit_with_error("Failed to read image file");
  }
  fclose(file);
  image_t image;
  for (int i = 0; i < 300; i++) {
    for (int j = 0; j < 300; j++) {
      image.pixels[i][j] = buffer[(i * 300) + j];
    }
  }
  image.size = 300;
  free(buffer);

  // IPGU
  image_t all_images[6];
  all_images[0] = image;
  int image_sizes[] = { 240, 180, 120, 60, 20 };
  for (int i = 0; i < 5; i++) {
    image_t scaled_image;
    scaled_image.size = image_sizes[i];
    uint8_t counts[image_sizes[i]][image_sizes[i]];
    uint16_t temp_pixels[image_sizes[i]][image_sizes[i]];
    for (int j = 0; j < image_sizes[i]; j++) {
      for (int k = 0; k < image_sizes[i]; k++) {
        counts[j][k] = 0;
        temp_pixels[j][k] = 0;
      }
    }
    float scale = (float)image_sizes[i] / (float)300;
    for (int j = 0; j < 300; j++) {
      for (int k = 0; k < 300; k++) {
        int scl_r = (int)roundf(scale * j);
        int scl_c = (int)roundf(scale * k);
        temp_pixels[scl_r][scl_c] += image.pixels[j][k];
        counts[scl_r][scl_c]++;
      }
    }
    for (int j = 0; j < image_sizes[i]; j++) {
      for (int k = 0; k < image_sizes[i]; k++) {
        scaled_image.pixels[j][k] = (uint8_t)roundf((float)temp_pixels[j][k] / (float)counts[j][k]);
      }
    }
    all_images[i + 1] = scaled_image;
  }
  sub_image_t sub_images[496];
  int sub_image_cnt = 0;
  for (int i = 0; i < 6; i++) {
    image_t cur = all_images[i];
    for (int j = 0; j < cur.size; j += 20) {
      for (int k = 0; k < cur.size; k += 20) {
        sub_image_t sub_image;
        for (int l = 0; l < 20; l++) {
          for (int m = 0; m < 20; m++) {
            sub_image.pixels[l][m] = cur.pixels[(j * 20) + l][(k * 20) + m];
          }
        }
        sub_images[sub_image_cnt++] = sub_image;
      }
    }
  }

  // HEU
  for (int i = 0; i < 496; i++) {
    heu(&sub_images[i]);
  }

  return 0;
}
