#include <inttypes.h>

#include "afu.h"
#include "afu_json_info.h"

int main() {

  afu_t afu;

  setup_afu(&afu, AFU_ACCEL_UUID);

  uint64_t uuid_l = read_afu_csr(&afu, AFU_ID_L);
  uint64_t uuid_h = read_afu_csr(&afu, AFU_ID_H);

  printf("id_l: %" PRIx64 ", id_h: %" PRIx64 "\n", uuid_l, uuid_h);

  close_afu(&afu);

  return 0;
}
