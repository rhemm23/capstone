#include "compiler.h"

static void compiler_error(char *error, int line_num) {
  fprintf(stderr, "Program Error on line %d: %s\n", line_num, error);
  exit(EXIT_FAILURE);
}

void compile_program(char *program_path, uint32_t **compiled_program) {

  // Open program file
  FILE *program = fopen(program_path, "r");
  if (program == NULL) {
    exit_with_error("Could not open specified program");
  }

  char *line;
  size_t len;
  ssize_t cnt;

  int line_num = 0;
  int instruction_cnt = 0;
  uint32_t instructions[MAX_INSTRUCTIONS];

  while ((cnt = getline(&line, &len, program)) != -1) {

    char *temp = line;
    line_num++;

    // Read leading whitespace or comment
    bool is_cmt_line = false;
    while (*temp != '\n' && (*temp == ' ' || *temp == '\t' || *temp == '#')) {
      if (*temp == '#') {
        is_cmt_line = true;
        break;
      } else {
        temp++;
      }
    }
    if (is_cmt_line || *temp == '\n') {
      continue;
    }

    // Read opcode
    uint8_t opcode;
    if (strncmp("HLT", temp, 3) == 0) {
      temp += 3;
      opcode = 0x00u;
    } else if (strncmp("SET_RESULT_ADDR", temp, 15) == 0) {
      temp += 15;
      opcode = 0x01u;
    } else if (strncmp("LOAD_RNW", temp, 8) == 0) {
      temp += 8;
      opcode = 0x02u;
    } else if (strncmp("LOAD_DNW", temp, 8) == 0) {
      temp += 8;
      opcode = 0x03u;
    } else if (strncmp("SET_IMG_NUM", temp, 11) == 0) {
      temp += 11;
      opcode = 0x05u;
    } else if (strncmp("BEGIN_PROC", temp, 10) == 0) {
      temp += 10;
      opcode = 0x04u;
    } else {
      compiler_error("Invalid instruction", line_num);
    }

    // Read immediate
    uint32_t immediate;
    if (opcode != 0x00u) {
      if (*(temp++) != ' ') {
        compiler_error("Expected an immediate value", line_num);
      }

      char *end;
      immediate = strtoul(temp, &end, 0);

      if (temp == end) {
        compiler_error("Invalid immediate value", line_num);
      } else if (opcode == 0x05u && immediate > 0xFFFFu) {
        compiler_error("Immediate value too large for 16 bits", line_num);
      } else if (opcode != 0x05u && immediate > 0x0FFFFFFFu) {
        compiler_error("Immediate value too large for 28 bits", line_num);
      } else {
        temp = end;
      }
    }

    // Read trailing whitespace or comment
    while (*temp != '\n') {
      if (*temp == ' ' || *temp == '\t') {
        temp++;
      } else if (*temp == '#') {
        break;
      } else {
        compiler_error("Invalid character", line_num);
      }
    }

    // Store formatted instruction
    if (instruction_cnt == MAX_INSTRUCTIONS) {
      compiler_error("Only 4096 instructions are supported", line_num);
    } else {
      uint32_t instruction;
      if (opcode == 0x00u) {
        instruction = 0x00000000u;
      } else if (opcode == 0x05u) {
        instruction = 0x50000000u | ((uint16_t)immediate);
      } else {
        instruction = (((uint32_t)opcode) << 28) | immediate;
      }
      instructions[instruction_cnt++] = instruction;
    }
  }

  /*
   * Allocate buffer for compiled program
   */
  *compiled_program = (uint32_t*)calloc(MAX_INSTRUCTIONS, sizeof(uint32_t));
  for (int i = 0; i < instruction_cnt; i++) {
    (*compiled_program)[i] = instructions[i];
  }
  fclose(program);
}
