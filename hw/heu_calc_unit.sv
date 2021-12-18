module heu_calc_unit
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input z,
    input en,
    input [7:0] d,

    /*
     * Outputs
     */
    output reg [6:0] q [255:0]
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || z) begin
      for (integer i = 0; i < 256; i++) begin
        q[i] <= 7'h00;
      end
    end else if (en) begin
      q[d] <= q[d] + 1;
    end
  end

endmodule
