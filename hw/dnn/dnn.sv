module dnn
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input mem_ready,
    input load_weights,
    input next_in_ready,
    input prev_out_ready,
    input [7:0] d [4:0][79:0],
    input [63:0] mem_data [7:0],

    /*
     * Outputs
     */
    output req_mem,
    output in_ready,
    output out_ready,
    output weight_valid,
    output [511:0] results
  );

  wire [7:0] det_d [399:0];

  wire [63:0] weight_bus;
  wire [2:0] layer_sel;
  wire [3:0] neuron_sel;
  wire [6:0] weight_sel;

  wire write_weight;

  wire det_nn_done;
  wire start_det_nn;
  wire det_nn_result;

  dnn_weight_ld weight_ld (
    .clk(clk),
    .rst_n(rst_n),
    .start(load_weights),
    .mem_ready(mem_ready),
    .mem_data(mem_data),
    .weight_bus(weight_bus),
    .layer_sel(layer_sel),
    .neuron_sel(neuron_sel),
    .weight_sel(weight_sel),
    .write_weight(write_weight),
    .weight_valid(weight_valid),
    .req_mem(req_mem)
  );

  dnn_ctrl_unit ctrl_unit (
    .clk(clk),
    .rst_n(rst_n),
    .det_nn_done(det_nn_done),
    .det_nn_result(det_nn_result),
    .prev_out_ready(prev_out_ready),
    .next_in_ready(next_in_ready),
    .in_ready(in_ready),
    .out_ready(out_ready),
    .start_det_nn(start_det_nn),
    .results(results)
  );

  det_nn det (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_det_nn),
    .data_in(det_d),
    .write_weight(write_weight),
    .layer_sel(layer_sel),
    .neuron_sel(neuron_sel),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .result(det_nn_result),
    .done(det_nn_done)
  );

  generate
    genvar i;
    for (i = 0; i < 5; i++) begin : g_conn
      assign det_d[(i * 80) +: 80] = d[i];
    end
  endgenerate

endmodule
