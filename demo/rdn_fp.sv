module rdn_fp
  #(
    parameter NUM_A_NEURONS = 15,
    parameter NUM_B_NEURONS = 30,
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
    input heu_out_valid,
    input iru_in_ready,
    input [7:0] d [4:0][79:0],
    input [63:0] mem_data [7:0],

    /*
     * Outputs
     */
    output in_ready,
    output out_ready,
    output weight_valid, //all weights have been received
    output req_mem,
    output [NUM_C_NEURONS-1:0] angle_out,
    output [7:0] q [4:0][79:0]
  );

  wire write_weight;
  wire start_rnn;
  wire rot_nn_done;

  wire [1:0] layer_sel;
  wire [5:0] neuron_sel;

  wire [63:0] weight_bus;

  wire [8:0] weight_sel;
  wire [7:0] data_in [399:0];
  reg  [7:0] out_buffer [4:0][79:0];

  assign data_in[79:0]    = d[0];
  assign data_in[159:80]  = d[1];
  assign data_in[239:160] = d[2];
  assign data_in[319:240] = d[3];
  assign data_in[399:320] = d[4];
  
  // Acts as the out_buffer for rdn
  integer i, j;
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < 5; i = i + 1) begin
        for (j = 0; j < 80; j = j + 1) begin
          out_buffer[i][j] <= 8'h00;
        end
      end
    end else if (start_rnn)
      out_buffer <= d;
  end

  assign q = out_buffer;

  rdn_weight_ld_fp weight_loader (
  .clk(clk),
  .rst_n(rst_n),
  .go(load_weights),
  .mem_ready(mem_ready),
  .mem_data(mem_data),

  .weight_bus(weight_bus),
  .layer_sel(layer_sel),
  .neuron_sel(neuron_sel),
  .write_weight(write_weight),
  .weight_sel(weight_sel),
  .weight_valid(weight_valid),
  .req_mem(req_mem)
  );

  rdn_ctrl_unit_fp ctrl_unit (
    .clk(clk),
    .rst_n(rst_n),
    .heu_out_valid(heu_out_valid),
    .iru_in_ready(iru_in_ready),
    .load_weights(load_weights),
    .weight_valid(weight_valid),
    .rot_nn_done(rot_nn_done),

    .start_rnn(start_rnn),
    .in_ready(in_ready),
    .out_ready(out_ready)
  );

  // Buffer built into rot_nn network
  rot_nn network (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_rnn),
    .data_in(data_in),
    .write_weight(write_weight),
    .layer_sel(layer_sel),
    .neuron_sel(neuron_sel),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),

    .angle(angle_out),
    .done(rot_nn_done)
  );

endmodule
