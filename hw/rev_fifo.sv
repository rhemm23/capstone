
/*
 * Circular FIFO
 */
module rev_fifo
  #(
    SIZE = 80
  )
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input wr,
    input en,
    input [7:0] d [SIZE-1:0],

    /*
     * Outputs
     */
    output [7:0] q
  );

  reg [7:0] data [SIZE-1:0];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data <= '{ SIZE { 8'h00 } };
    end else if (wr) begin
      for (integer i = 0; i < SIZE; i++) begin
        data[i] = d[SIZE - i - 1];
      end
    end else if (en) begin
      data[0] <= data[SIZE-1];
      data[SIZE-1:1] <= data[SIZE-2:0];
    end
  end

  assign q = data[SIZE - 1];

endmodule
