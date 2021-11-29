
`include "data_types.vh"

package mem_types;

  typedef logic [63:0] t_mem_addr;

  typedef enum logic [2:0] {
    NONE = 3'b000,
    INSTR = 3'b001,
    RNN_W = 3'b010,
    DNN_W = 3'b011,
    IMAGE = 3'b100
  } t_mem_rd_req_type;

  typedef enum logic [2:0] {
    NONE_VALID = 3'b000,
    INSTR_VALID = 3'b001,
    RNN_W_VALID = 3'b010,
    DNN_W_VALID = 3'b011,
    IMAGE_VALID = 3'b100
  } t_mem_rx_status;

  typedef struct {
    t_mem_rd_req_type req_type;
    t_mem_addr addr;
  } t_mem_tx;

  typedef struct {
    t_rnn_weights rnn_weights;
    t_dnn_weights dnn_weights;
    t_mem_rx_status status;
    t_program cpu_program;
    t_image image_data;
  } t_mem_rx;

endpackage
