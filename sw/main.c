#include <inttypes.h>
#include <stdlib.h>

#include "afu.h"
#include "afu_json_info.h"

int main() {

  afu_t afu;

  setup_afu(&afu, AFU_ACCEL_UUID);

  uint32_t *instructions = calloc(64, sizeof(uint32_t));

  set_afu_buffer(&afu, (void**)&instructions, 64 * sizeof(uint32_t));

  close_afu(&afu);

  return 0;
}
