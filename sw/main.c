#include <stdbool.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "afu.h"
#include "compiler.h"
#include "afu_json_info.h"

static void exit_with_error(char *error) {
  fprintf(stderr, "Error: %s\n", error);
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  /*
   * Progress command line args
   */
  char *program_path = NULL;

  for (int i = 1; i < argc; i++) {
    if (strcmp("--help", argv[i]) == 0 || strcmp("-h", argv[i]) == 0) {
      printf("\nCommand line arguments:\n\n");
      printf("\t<--help, -h>                      -> Print command line args\n");
      printf("\t<--program, -p> <path to program> -> Specify the CPU program to execute\n\n");
      exit(0);
    }
    if (strcmp("--program", argv[i]) == 0 || strcmp("-p", argv[i]) == 0) {
      if (i == argc - 1) {
        exit_with_error("Expected program path after option");
      } else {
        program_path = argv[++i];
      }
    }
  }

  /*
   * Assure required args are specified
   */
  if (program_path == NULL) {
    exit_with_error("Missing program path parameter, use --help or -h");
  }

  uint32_t *compiled_program;
  compile_program(program_path, &compiled_program);

  // Size of entire shared memory buffer
  uint64_t buffer_size = MAX_INSTRUCTIONS * 4;

  afu_t afu;
  setup_afu(&afu, AFU_ACCEL_UUID);

  // Setup buffer
  volatile uint32_t *program_buffer = (volatile uint32_t*)create_afu_buffer(&afu, buffer_size);

  // Copy program instructions
  for (int i = 0; i < MAX_INSTRUCTIONS; i++) {
    program_buffer[i] = compiled_program[i];
  }

  // Allow AFU to execute, replace with events in future
  sleep(10);

  for (int i = 0; i < MAX_INSTRUCTIONS; i++) {
    if (program_buffer[i] != 0xffffffff) {
      printf("invalid\n");
      break;
    }
  }

  close_afu(&afu);
  free(compiled_program);

  return 0;
}
