module out_fifo  #(Q_DEPTH = 80) 
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
    output reg [7:0] q [Q_DEPTH-1:0]
  );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      q <= '{ Q_DEPTH { 8'h00 } };
    end else if (en) begin
      q[0] <= d;
      q[Q_DEPTH-1:1] <= q[Q_DEPTH-2:0];
    end
  end

endmodule
