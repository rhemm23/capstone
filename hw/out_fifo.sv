module out_fifo
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input en,
    input [7:0] d,

    /*
     * Outputs
     */
    output reg [7:0] q [79:0]
  );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      q <= '{ 80 { 8'h00 } };
    end else if (en) begin
      q[0] <= d;
      q[79:1] <= q[78:0];
    end
  end

endmodule
