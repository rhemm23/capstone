#include <inttypes.h>

#include "afu.h"
#include "afu_json_info.h"

int main() {

  afu_t afu;

  setup_afu(&afu, AFU_ACCEL_UUID);

  uint32_t instructions[64];

  for (int i = 0; i < 64; i++) {
    instructions[i] = 0;
  }

  set_afu_buffer(&afu, (void**)&&instructions[0], 512);

  close_afu(&afu);

  return 0;
}
