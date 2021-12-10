
/*
 * Calculate e^x using the taylor series
 */
module fp_exp
  #(
    ITERATIONS = 2
  )
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

  typedef enum logic [3:0] {
    IDLE = 4'b0000,
    ADD_X = 4'b0001,
    WAIT_ADD_X = 4'b0010,
    MULT_X = 4'b0011,
    WAIT_MULT_X = 4'b0100,
    CONV_FACT = 4'b0101,
    WAIT_CONV_FACT = 4'b0110,
    DIV_FACT = 4'b0111,
    WAIT_DIV_FACT = 4'b1000,
    DONE = 4'b1001
  } exp_state;

  exp_state state;

  reg [5:0] cnt;

  reg [63:0] fp_factorial;
  reg [63:0] factorial;

  reg [63:0] x;
  reg [63:0] sum;
  reg [63:0] prod;
  reg [63:0] prod_div;

  reg [63:0] div_a;
  reg [63:0] div_b;

  reg [63:0] add_a;
  reg [63:0] add_b;

  reg [63:0] mult_a;
  reg [63:0] mult_b;

  reg start_adder;
  reg start_mult;
  reg start_conv;
  reg start_div;

  wire div_done;
  wire [63:0] div_c;

  wire add_done;
  wire [63:0] add_c;

  wire mult_done;
  wire [63:0] mult_c;

  wire conv_done;
  wire [63:0] conv_out;

  long_to_fp conv (
    .clk(clk),
    .rst_n(rst_n),
    .long_in(factorial),
    .start(start_conv),
    .fp_out(conv_out),
    .done(conv_done)
  );

  fp_mult mult (
    .clk(clk),
    .rst_n(rst_n),
    .a_in(mult_a),
    .b_in(mult_b),
    .start(start_mult),
    .c_out(mult_c),
    .done(mult_done)
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

  fp_divider div (
    .clk(clk),
    .rst_n(rst_n),
    .a_in(div_a),
    .b_in(div_b),
    .start(start_div),
    .c_out(div_c),
    .done(div_done)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      start_adder <= 1'b0;
      start_mult <= 1'b0;
      start_conv <= 1'b0;
      start_div <= 1'b0;
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (start) begin
          sum <= 64'h3ff0000000000000;
          x <= fp_in;
          cnt <= 1;
          prod <= fp_in;
          prod_div <= fp_in;
          factorial <= 1;
          state <= ADD_X;
        end
        ADD_X: begin
          start_adder <= 1'b1;
          add_a <= sum;
          add_b <= prod_div;
          state <= WAIT_ADD_X;
        end
        WAIT_ADD_X: begin
          if (add_done) begin
            sum <= add_c;
            if (cnt == ITERATIONS) begin
              state <= DONE;
            end else begin
              state <= MULT_X;
              cnt <= cnt + 1;
              factorial <= factorial * (cnt + 1);
            end
          end
          start_adder <= 1'b0;
        end
        MULT_X: begin
          start_mult <= 1'b1;
          mult_a <= prod;
          mult_b <= x;
          state <= WAIT_MULT_X;
        end
        WAIT_MULT_X: begin
          if (mult_done) begin
            prod <= mult_c;
            state <= CONV_FACT;
          end
          start_mult <= 1'b0;
        end
        CONV_FACT: begin
          start_conv <= 1'b1;
          state <= WAIT_CONV_FACT;
        end
        WAIT_CONV_FACT: begin
          if (conv_done) begin
            fp_factorial <= conv_out;
            state <= DIV_FACT;
          end
          start_conv <= 1'b0;
        end
        DIV_FACT: begin
          start_div <= 1'b1;
          div_a <= prod;
          div_b <= fp_factorial;
          state <= WAIT_DIV_FACT;
        end
        WAIT_DIV_FACT: begin
          if (div_done) begin
            prod_div <= div_c;
            state <= ADD_X;
          end
          start_div <= 1'b0;
        end
        DONE: begin
          state <= IDLE;
        end
      endcase
    end
  end

  assign fp_out = sum;
  assign done = (state == DONE);

endmodule
