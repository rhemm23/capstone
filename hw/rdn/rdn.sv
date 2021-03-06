module rdn
  #(
    parameter NUM_A_NEURONS = 15,
    parameter NUM_B_NEURONS = 15,
    parameter NUM_C_NEURONS = 36
  )
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input load_weights,
    input mem_ready,
    input heu_out_ready,
    input iru_in_ready,
    input [7:0] d [4:0][79:0],
    input [15:0] mem_data [31:0],

    /*
     * Outputs
     */
    output in_ready,
    output out_ready,
    output weight_valid,
    output req_mem,
    output [NUM_C_NEURONS-1:0] net_out,
    output [7:0] q [4:0][79:0]
  );

  wire [15:0] a_layer_q [NUM_A_NEURONS-1:0];
  wire [15:0] b_layer_q [NUM_B_NEURONS-1:0];

  wire [7:0] in_buffer_q [4:0];

  wire shift_out;
  wire rotate_in;
  wire write_in;

  wire z_a_layer;
  wire z_b_layer;
  wire z_c_layer;
  wire en_a_layer;
  wire en_b_layer;
  wire en_c_layer;

  wire write_a;
  wire write_b;
  wire write_c;

  wire [3:0] a_sel;
  wire [3:0] b_sel;
  wire [5:0] c_sel;

  wire [15:0] b_layer_in;
  wire [15:0] c_layer_in;

  wire signed [15:0] a_weight_bus;
  wire signed [15:0] b_weight_bus;
  wire signed [15:0] c_weight_bus;

  wire [8:0] a_weight_sel;
  wire [$clog2(15)-1:0] b_weight_sel;
  wire [$clog2(15)-1:0] c_weight_sel;

  rdn_weight_ld weight_loader (
    .a_sel(a_sel),
    .b_sel(b_sel),
    .c_sel(c_sel),
    .write_a(write_a),
    .write_b(write_b),
    .write_c(write_c),
    .weight_valid(weight_valid)
  );
  (
  /*
    * Inputs
    */
  .clk(clk),
  .rst_n(rst_n),
  .go(load_weights),
  .mem_ready(mem_ready),
  .mem_data(mem_data),

  /*
    * logics
    */
  .a_weight_bus(a_weight_bus),
  .b_weight_bus(b_weight_bus),
  .c_weight_bus(c_weight_bus),
  .a_sel(a_sel),
  .b_sel(b_sel),
  .c_sel(c_sel),
  .write_a(write_a),
  .write_b(write_b),
  .write_c(write_c),
  .a_weight_sel(a_weight_sel),
  .b_weight_sel(b_weight_sel),
  .c_weight_sel(c_weight_sel),
  .weight_valid(weight_valid),
  .req_mem(req_mem)
  );

  rdn_ctrl_unit ctrl_unit (
    .clk(clk),
    .rst_n(rst_n),
    .heu_out_ready(heu_out_ready),
    .iru_in_ready(iru_in_ready),
    .a_layer_out(a_layer_q),
    .b_layer_out(b_layer_q),
    .rotate_in(rotate_in),
    .write_in(write_in),
    .shift_out(shift_out),
    .z_a_layer(z_a_layer),
    .z_b_layer(z_b_layer),
    .z_c_layer(z_c_layer),
    .en_a_layer(en_a_layer),
    .en_b_layer(en_b_layer),
    .en_c_layer(en_c_layer),
    .b_layer_in(b_layer_in),
    .c_layer_in(c_layer_in),
    .in_ready(in_ready),
    .out_ready(out_ready)
  );

  in_buffer in_buf (
    .clk(clk),
    .rst_n(rst_n),
    .wr(write_in),
    .en(rotate_in),
    .d(d),
    .q(in_buffer_q)
  );

  out_buffer out_buf (
    .clk(clk),
    .rst_n(rst_n),
    .en(shift_out),
    .d(in_buffer_q),
    .q(q)
  );

  generate
    genvar i;
    for (i = 0; i < NUM_A_NEURONS; i++) begin : a_layer
      a_neuron a (
        .clk(clk),
        .rst_n(rst_n),
        .z(z_a_layer),
        .en(en_a_layer),
        .wr_weights((a_sel == i) && write_a),
        .a_weight_sel(a_weight_sel), //TODO add
        .d(in_buffer_q),
        .bias_d(a_weight_bus), // TODO fix
        .weights_d(a_weight_bus), // TODO fix
        .q(a_layer_q[i])
      );
    end
    for (i = 0; i < NUM_B_NEURONS; i++) begin : b_layer
      b_neuron #(.INPUTS(NUM_A_NEURONS)) b (
        .clk(clk),
        .rst_n(rst_n),
        .z(z_b_layer),
        .en(en_b_layer),
        .wr_weights((b_sel == i) && write_b),
        .b_weight_sel(b_weight_sel), // TODO add
        .d(b_layer_in),
        .bias_d(b_weight_bus),// TODO fix
        .weights_d(b_weight_bus),// TODO fix
        .q(b_layer_q[i])
      );
    end
    for (i = 0; i < NUM_C_NEURONS; i++) begin : c_layer
      c_neuron #(.INPUTS(NUM_B_NEURONS)) c (
        .clk(clk),
        .rst_n(rst_n),
        .z(z_c_layer),
        .en(en_c_layer),
        .wr_weights((c_sel == i) && write_c),
        .c_weight_sel(c_weight_sel), // TODO add
        .d(c_layer_in),
        .bias_d(c_weight_bus),// TODO fix
        .weights_d(c_weight_bus),// TODO fix
        .q(net_out[i])
      );
    end
  endgenerate

endmodule
