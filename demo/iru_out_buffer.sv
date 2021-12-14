module iru_out_buffer
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input z,
    input wr [4:0],
    input [7:0] d [4:0],
    input [4:0] row [4:0],
    input [4:0] col [4:0],

    /*
     * Outputs
     */
    output [7:0] q [4:0][79:0]
  );

  reg [7:0] data [19:0][19:0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || z) begin
      for (integer r = 0; r < 20; r++) begin
        for (integer c = 0; c < 20; c++) begin
          data[r][c] <= '0;
        end
      end
    end else begin
      for (integer i = 0; i < 5; i++) begin
        if (wr[i]) begin
          data[row[i]][col[i]] <= d[i]; // TODO: Make sure there is no contention
        end
      end
    end
  end

  generate
    genvar buffer, r;
    for (buffer = 0; buffer < 5; buffer++) begin : buffer_g
      for (r = 0; r < 4; r++) begin : r_g
        assign q[buffer][r * 20 +: 20] = data[(buffer * 4) + r];
      end
    end
  endgenerate

endmodule
