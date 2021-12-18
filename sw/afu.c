#include "afu.h"

static void exit_with_error(char *error) {
  fprintf(stderr, "AFU Error: %s\n", error);
  exit(EXIT_FAILURE);
}

static void close_with_error(afu_t *afu, char *error) {
  close_afu(afu);
  exit_with_error(error);
}

void close_afu(afu_t *afu) {
  if (afu->shared_buffer.wsid) {
    if (fpgaReleaseBuffer(afu->handle, afu->shared_buffer.wsid) != FPGA_OK) {
      exit_with_error("Failed to release shared memory buffer");
    }
  }
  if (afu->is_mmio_mapped) {
    if (fpgaUnmapMMIO(afu->handle, 0) != FPGA_OK) {
      exit_with_error("Failed to unmap mmio space");
    }
  }
  if (afu->handle) {
    if (fpgaClose(afu->handle) != FPGA_OK) {
      exit_with_error("Failed to close fpga handle");
    }
  }
  if (afu->token) {
    if (fpgaDestroyToken(&afu->token) != FPGA_OK) {
      exit_with_error("Failed to destroy token");
    }
  }
  if (afu->filter) {
    if (fpgaDestroyProperties(&afu->filter) != FPGA_OK) {
      exit_with_error("Failed to destroy properties object");
    }
  }
}

void setup_afu(afu_t *afu, char *uuid) {

  /*
   * Default fields
   */
  afu->shared_buffer.iova = 0;
  afu->shared_buffer.wsid = 0;

  afu->is_mmio_mapped = false;

  afu->filter = NULL;
  afu->handle = NULL;
  afu->token = NULL;

  uint32_t num_matches = 0;

  if (uuid_parse(uuid, afu->uuid) < 0) {
    exit_with_error("Failed to parse uuid");
  }
  if (fpgaGetProperties(NULL, &afu->filter) != FPGA_OK) {
    close_with_error(afu, "Failed to get properties object");
  }
  if (fpgaPropertiesSetObjectType(afu->filter, FPGA_ACCELERATOR) != FPGA_OK) {
    close_with_error(afu, "Failed to set filter object type");
  }
  if (fpgaPropertiesSetGUID(afu->filter, afu->uuid) != FPGA_OK) {
    close_with_error(afu, "Failed to set filter uuid");
  }
  if (fpgaEnumerate(&afu->filter, 1, &afu->token, 1, &num_matches) != FPGA_OK) {
    close_with_error(afu, "Failed to enumerate available fpgas");
  }
  if (fpgaOpen(afu->token, &afu->handle, 0) != FPGA_OK) {
    close_with_error(afu, "Failed to open available fpga");
  }
  if (fpgaMapMMIO(afu->handle, 0, NULL) != FPGA_OK) {
    close_with_error(afu, "Failed to map fpga mmio space");
  }
  afu->is_mmio_mapped = true;
}

uint64_t read_afu_csr(afu_t *afu, afu_csr_t csr) {
  if (!afu->is_mmio_mapped) {
    close_with_error(afu, "MMIO space must be mapped before reading from it");
  }
  uint64_t value;
  if (fpgaReadMMIO64(afu->handle, 0, csr * 8, &value) != FPGA_OK) {
    close_with_error(afu, "Failed to read from CSR");
  }
  return value;
}

void write_afu_csr(afu_t *afu, afu_csr_t csr, uint64_t value) {
  if (!afu->is_mmio_mapped) {
    close_with_error(afu, "MMIO space must be mapped before writing to it");
  }
  if (fpgaWriteMMIO64(afu->handle, 0, csr * 8, value) != FPGA_OK) {
    close_with_error(afu, "Failed to write to CSR");
  }
}

void * create_afu_buffer(afu_t *afu, uint64_t size) {

  // Page align size
  int page_size = getpagesize();
  if (size % page_size != 0) {
    size += page_size - (size % page_size);
  }

  uint64_t iova;
  void *buffer;

  // Prepare new buffer
  if (fpgaPrepareBuffer(afu->handle, size, &buffer, &afu->shared_buffer.wsid, 0) != FPGA_OK) {
    close_with_error(afu, "Failed to create shared memory buffer");
  }

  // Get physical address and send to AFU
  if (fpgaGetIOAddress(afu->handle, afu->shared_buffer.wsid, &iova) != FPGA_OK) {
    close_with_error(afu, "Failed to get shared memory buffer IO address");
  }

  // Cache align iova
  iova >>= 6;

  // Notify fpga of buffer
  write_afu_csr(afu, BUFFER_ADDR, iova);

  // Store buffer info
  afu->shared_buffer.iova = iova;

  return buffer;
}
