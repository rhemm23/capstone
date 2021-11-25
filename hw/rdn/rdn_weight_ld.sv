module rdn_weight_ld
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input go,
    input mem_ready,
    input [7:0] mem_data [63:0],

    /*
     * Outputs
     */
    output [8:0] weight_bus [400:0],
    output [3:0] a_sel,
    output [3:0] b_sel,
    output [5:0] c_sel,
    output write_a,
    output write_b,
    output write_c,
    output weight_valid
  );

  /*
   * TODO
   */

endmodule
