#ifndef AFU_H
#define AFU_H

#include <opae/fpga.h>
#include <uuid/uuid.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

typedef enum afu_csr {
  DHF = 0,
  AFU_ID_L = 1,
  AFU_ID_H = 2,
  BUFFER_ADDR = 5
} afu_csr_t;

typedef struct afu_buffer {
  uint64_t iova;
  uint64_t wsid;
} afu_buffer_t;

typedef struct afu {
  afu_buffer_t shared_buffer;
  fpga_properties filter;
  bool is_mmio_mapped;
  fpga_handle handle;
  fpga_token token;
  fpga_guid uuid;
} afu_t;

/*
 * Sets up a new AFU object using a string UUID
 */
void setup_afu(afu_t *afu, char *uuid);

/*
 * Read from an AFU control status register
 */
uint64_t read_afu_csr(afu_t *afu, afu_csr_t csr);

/*
 * Write to an AFU control status register
 */
void write_afu_csr(afu_t *afu, afu_csr_t csr, uint64_t value);

/*
 * Create a new AFU shared memory buffer of a specified size,
 * and return the pointer to the buffer
 */
void * create_afu_buffer(afu_t *afu, uint64_t size);

/*
 * Release all AFU resources
 */
void close_afu(afu_t *afu);

#endif
