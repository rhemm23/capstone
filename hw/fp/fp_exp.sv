
/*
 * Calculate e^x using the taylor series
 */
module fp_exp
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input [63:0] fp_in,
    input start,

    /*
     * Outputs
     */
    output [63:0] fp_out,
    output done
  );

  typedef enum logic [] {
    IDLE,
    ADD_X,
    WAIT_ADD_X,
    MULT_X,
    DONE
  } exp_state;

  exp_state state;

  reg [63:0] x;
  reg [63:0] sum;

  reg [63:0] add_a;
  reg [63:0] add_b;

  reg [63:0] mult_a;
  reg [63:0] mult_b;

  reg start_adder;
  reg start_mult;

  wire add_done;
  wire [63:0] add_c;

  wire mult_done;
  wire [63:0] mult_c;

  fp_mult mult (
    .clk(clk),
    .rst_n(rst_n),
    .a_in(mult_a),
    .b_in(mult_b),
    .start(start_mult),

  );

  fp_adder add (
    .clk(clk),
    .rst_n(rst_n),
    .a_in(add_a),
    .b_in(add_b),
    .start(start_adder),
    .c_out(add_c),
    .done(add_done)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      start_adder <= 1'b0;
      start_mult <= 1'b0;
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (start) begin
          sum <= 64'h3ff0000000000000;
          x <= fp_in;
          state <= ADD_X;
        end
        ADD_X: begin
          start_adder <= 1'b1;
          add_a <= sum;
          add_b <= x;
          state <= WAIT_ADD_X;
        end
        WAIT_ADD_X: begin
          if (add_done) begin
            sum <= add_c;
            state <= 
          end
          add_start <= 1'b0;
        end
      endcase
    end
  end

  assign start_product = (state == UPDATE_PRODUCT);

endmodule
