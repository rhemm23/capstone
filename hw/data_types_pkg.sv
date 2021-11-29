package data_types;

  /*
   * Instruction types
   */
  typedef logic [31:0] t_instruction;

  typedef t_instruction t_program [63:0];

  /*
   * Image types
   */
  typedef logic [7:0] t_pixel;

  typedef t_pixel t_image [299:0][299:0];

  /*
   * Neural net types
   */
  typedef logic [8:0] t_weight;

  typedef struct {
    t_weight bias;
    t_weight weights [399:0];
  } t_neuron_400_w;

  typedef struct {
    t_weight bias;
    t_weight weights [99:0];
  } t_neuron_100_w;

  typedef struct {
    t_weight bias;
    t_weight weights [79:0];
  } t_neuron_80_w;

  typedef struct {
    t_weight bias;
    t_weight weights [24:0];
  } t_neuron_25_w;

  typedef struct {
    t_weight bias;
    t_weight weights [15:0];
  } t_neuron_16_w;

  typedef struct {
    t_weight bias;
    t_weight weights [14:0];
  } t_neuron_15_w;

  typedef struct {
    t_weight bias;
    t_weight weights [4:0];
  } t_neuron_5_w;

  typedef struct {
    t_weight bias;
    t_weight weights [3:0];
  } t_neuron_4_w;

  typedef struct {
    t_weight bias;
    t_weight weights [2:0];
  } t_neuron_3_w;

  typedef struct {
    t_neuron_400_w a_layer [14:0];
    t_neuron_15_w b_layer [14:0];
    t_neuron_15_w c_layer [35:0];
  } t_rnn_weights;

  typedef struct {
    t_neuron_100_w a_layer_type1 [1:0][1:0];
    t_neuron_25_w a_layer_type2 [3:0][3:0];
    t_neuron_80_w a_layer_type3 [4:0];
    t_neuron_4_w b_neuron_type1;
    t_neuron_16_w b_neuron_type2;
    t_neuron_5_w b_neuron_type3;
    t_neuron_3_w c_neuron;
  } t_dnn_weights;

endpackage
