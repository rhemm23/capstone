module det_nn
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input start,
    input [7:0] data_in [399:0],
    input write_weight,
    input [2:0] layer_sel,
    input [3:0] neuron_sel,
    input [6:0] weight_sel,
    input [63:0] weight_bus,

    /*
     * Outputs
     */
    output result,
    output done
  );

  

endmodule
