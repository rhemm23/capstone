#ifndef COMPILER_H
#define COMPILER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void compile_program(char *program_path, uint32_t **compiled_program);

#endif
