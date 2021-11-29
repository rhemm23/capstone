C_LINK = -lopae-c -luuid -ljson-c -lpthread

BIN_DIR = ./bin
AFU_JSON_INFO = ../hw/afu.json
AFU_JSON_INFO_HDR = $(BIN_DIR)/afu_json_info.h

SRCS = main.c afu.c
OBJS = $(addprefix $(BIN_DIR)/,$(patsubst %.c,%.o,$(SRCS)))

afu: $(OBJS)
	gcc -o $@ $^ $(C_LINK)

clean:
	rm -rf $(BIN_DIR)

$(OBJS): $(AFU_JSON_INFO_HDR)

$(AFU_JSON_INFO_HDR): $(AFU_JSON_INFO)
	mkdir -p $(BIN_DIR)
	afu_json_mgr json-info --afu-json=$< --c-hdr=$@

$(BIN_DIR)/%.o: %.c
	mkdir -p $(BIN_DIR)
	gcc -c $< -o $@ $(C_LINK)
