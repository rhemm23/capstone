C_LINK = -luuid -ljson-c -lpthread

BIN_DIR = ./bin
AFU_JSON_INFO = ../hw/afu.json
AFU_JSON_INFO_HDR = ./afu_json_info.h

BBB_DIR = ${FPGA_BBB_CCI_SRC}/BBB_cci_mpf/sw/include/

# Build directory

prefix = /usr/local
OBJDIR = obj
CFLAGS += -I./$(OBJDIR) -I$(BBB_DIR) -I/filespace/s/sjain75/opae/include/
CPPFLAGS += -I./$(OBJDIR) -I$(BBB_DIR) -I/filespace/s/sjain75/opae/include/
CFLAGS   += -I$(DESTDIR)$(prefix)/include
CPPFLAGS += -I$(DESTDIR)$(prefix)/include
LDFLAGS  += -L$(DESTDIR)$(prefix)/lib -Wl,-rpath-link -Wl,$(prefix)/lib -Wl,-rpath -Wl,$(DESTDIR)$(prefix)/lib \
            -L$(DESTDIR)$(prefix)/lib64 -Wl,-rpath-link -Wl,$(prefix)/lib64 -Wl,-rpath -Wl,$(DESTDIR)$(prefix)/lib64


SRCS = main.c compiler.c afu.c
OBJS = $(addprefix $(BIN_DIR)/,$(patsubst %.c,%.o,$(SRCS)))

build: $(BIN_DIR)/afu

ase: $(BIN_DIR)/afu_ase

$(BIN_DIR)/afu_ase: $(AFU_JSON_INFO_HDR) $(SRCS)
	mkdir -p $(BIN_DIR)
	gcc $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(C_LINK) -lopae-c-ase

$(BIN_DIR)/afu: $(AFU_JSON_INFO_HDR) $(SRCS)
	mkdir -p $(BIN_DIR)
	gcc $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(C_LINK) -lopae-c

clean:
	rm -rf $(BIN_DIR)

$(AFU_JSON_INFO_HDR): $(AFU_JSON_INFO)
	afu_json_mgr json-info --afu-json=$< --c-hdr=$@

$(BIN_DIR)/%.o: %.c $(AFU_JSON_INFO_HDR)
	mkdir -p $(BIN_DIR)
	gcc $(CFLAGS) $(CPPFLAGS) -c $< -o $@ $(LDFLAGS) 
