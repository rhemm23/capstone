package data_types;

  /*
   * Instruction types
   */
  typedef logic [31:0] t_instruction;

  typedef t_instruction [63:0] t_program;

  /*
   * Image types
   */
  typedef logic [7:0] t_pixel;

  typedef t_pixel [299:0][299:0] t_image;

  /*
   * Neural net types
   */
  typedef logic [8:0] t_weight;

  typedef struct packed {
    t_weight bias;
    t_weight [399:0] weights;
  } t_neuron_400_w;

  typedef struct packed {
    t_weight bias;
    t_weight [99:0] weights;
  } t_neuron_100_w;

  typedef struct packed {
    t_weight bias;
    t_weight [79:0] weights;
  } t_neuron_80_w;

  typedef struct packed {
    t_weight bias;
    t_weight [24:0] weights;
  } t_neuron_25_w;

  typedef struct packed {
    t_weight bias;
    t_weight [15:0] weights;
  } t_neuron_16_w;

  typedef struct packed {
    t_weight bias;
    t_weight [14:0] weights;
  } t_neuron_15_w;

  typedef struct packed {
    t_weight bias;
    t_weight [4:0] weights;
  } t_neuron_5_w;

  typedef struct packed {
    t_weight bias;
    t_weight [3:0] weights;
  } t_neuron_4_w;

  typedef struct packed {
    t_weight bias;
    t_weight [2:0] weights;
  } t_neuron_3_w;

  typedef struct packed {
    t_neuron_400_w [14:0] a_layer;
    t_neuron_15_w [14:0] b_layer;
    t_neuron_15_w [35:0] c_layer;
  } t_rnn_weights;

  typedef struct packed {
    t_neuron_100_w [1:0][1:0] a_layer_type1;
    t_neuron_25_w [3:0][3:0] a_layer_type2;
    t_neuron_80_w [4:0] a_layer_type3;
    t_neuron_4_w b_neuron_type1;
    t_neuron_16_w b_neuron_type2;
    t_neuron_5_w b_neuron_type3;
    t_neuron_3_w c_neuron;
  } t_dnn_weights;

  /*
   * Memory types
   */
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

  typedef struct packed {
    t_mem_rd_req_type req_type;
    t_mem_addr addr;
  } t_mem_tx;

  typedef struct packed {
    t_rnn_weights rnn_weights;
    t_dnn_weights dnn_weights;
    t_mem_rx_status status;
    t_program cpu_program;
    t_image image_data;
  } t_mem_rx;

endpackage
