#include <stdbool.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "afu.h"
#include "compiler.h"
#include "afu_json_info.h"

ROT_WEIGHT_BYTES = 62016

static void exit_with_error(char *error) {
  fprintf(stderr, "Error: %s\n", error);
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  /*
   * Progress command line args
   */
  char *program_path = NULL;
  char *rot_weights_path = NULL;
  char *det_weights_path = NULL;

  for (int i = 1; i < argc; i++) {
    if (strcmp("--help", argv[i]) == 0 || strcmp("-h", argv[i]) == 0) {
      printf("\nCommand line arguments:\n\n");
      printf("\t<--help, -h>                          -> Print command line args\n");
      printf("\t<--program, -p> <path to program>     -> Specify the CPU program to execute\n");
      printf("\t<--rot-weights -rw> <path to weights> -> Specify the rotation weights to use\n");
      printf("\t<--det-weights -dw> <path to weights> -> Specify the detection weights to use\n\n");
      exit(0);
    }
    if (strcmp("--program", argv[i]) == 0 || strcmp("-p", argv[i]) == 0) {
      if (i == argc - 1) {
        exit_with_error("Expected program path after option");
      } else {
        program_path = argv[++i];
      }
    }
    if (strcmp("--rot-weights", argv[i]) == 0 || strcmp("-rw", argv[i]) == 0) {
      if (i == argc - 1) {
        exit_with_error("Expected weights path after option");
      } else {
        rot_weights_path = argv[++i];
      }
    }
    if (strcmp("--det-weights", argv[i]) == 0 || strcmp("-dw", argv[i]) == 0) {
      if (i == argc - 1) {
        exit_with_error("Expected weights path after option");
      } else {
        det_weights_path = argv[++i];
      }
    }
  }

  /*
   * Assure required args are specified
   */
  if (program_path == NULL) {
    exit_with_error("Missing program path parameter, use --help or -h");
  }
  if (rot_weights_path == NULL) {
    exit_with_error("Missing rotational weights path parameter, use --help or -h");
  }
  if (det_weights_path == NULL) {
    exit_with_error("Missing detection weights parameter, use --help or -h");
  }

  uint32_t *compiled_program;
  compile_program(program_path, &compiled_program);

  // Size of entire shared memory buffer
  uint64_t buffer_size = (MAX_INSTRUCTIONS * 4) + ROT_WEIGHT_BYTES;

  void *buffer = (void*)malloc(buffer_size);
  volatile uint32_t *program_buffer = (volatile uint32_t*)buffer;

  // Copy program instructions
  for (int i = 0; i < MAX_INSTRUCTIONS; i++) {
    program_buffer[i] = compiled_program[i];
  }

  FILE *rot_weight_file = fopen(rot_weights_path, "rb");
  if (rot_weight_file == NULL) {
    exit_with_error("Could not open rotation weight file");
  }

  volatile uint8_t *rot_weights = (volatile uint8_t*)&program_buffer[MAX_INSTRUCTIONS];
  for (int i = 0; i < ROT_WEIGHT_BYTES; i++) {
    uint8_t byte;
    if (fread(&byte, sizeof(uint8_t), 1, rot_weight_file) != 1) {
      exit_with_error("Failed to read from rotation weight file");
    }
    rot_weights[i] = byte;
  }

  afu_t afu;
  setup_afu(&afu, AFU_ACCEL_UUID);
  setup_afu_buffer(&afu, &buffer, buffer_size);

  // Allow AFU to execute, replace with mmio poll in future
  sleep(10);

  close_afu(&afu);
  free(compiled_program);

  return 0;
}
