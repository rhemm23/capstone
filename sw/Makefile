C_LINK = -luuid -ljson-c -lpthread

BIN_DIR = ./bin
AFU_JSON_INFO = ../hw/afu.json
AFU_JSON_INFO_HDR = ./afu_json_info.h

SRCS = main.c compiler.c afu.c
OBJS = $(addprefix $(BIN_DIR)/,$(patsubst %.c,%.o,$(SRCS)))

build: $(BIN_DIR)/afu

ase: $(BIN_DIR)/afu_ase

$(BIN_DIR)/afu_ase: $(OBJS)
	mkdir -p $(BIN_DIR)
	gcc -o $@ $^ $(C_LINK) -lopae-c-ase

$(BIN_DIR)/afu: $(OBJS)
	mkdir -p $(BIN_DIR)
	gcc -o $@ $^ $(C_LINK) -lopae-c

clean:
	rm -rf $(BIN_DIR)

$(AFU_JSON_INFO_HDR): $(AFU_JSON_INFO)
	afu_json_mgr json-info --afu-json=$< --c-hdr=$@

$(BIN_DIR)/%.o: %.c $(AFU_JSON_INFO_HDR)
	mkdir -p $(BIN_DIR)
	gcc -c $< -o $@ $(C_LINK)
