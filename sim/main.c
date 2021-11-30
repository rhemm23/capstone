#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define IMAGE_SIZE 90000

typedef struct image {
  uint8_t pixels[300][300];
} image_t;

typedef struct sub_image {
  uint8_t pixels[20][20];
} sub_image_t;

static void exit_with_error(char *error) {
  fprintf(stderr, "Error: %s\n", error);
  exit(EXIT_FAILURE);
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
  free(buffer);

  // IPGU
  sub_image_t sub_images[496];

  int image_sizes[] = { 300, 240, 180, 120, 60 };

  return 0;
}
