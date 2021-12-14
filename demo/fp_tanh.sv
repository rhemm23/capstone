module fp_tanh
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
  
  localparam NEG1 = 64'hbff0000000000000;
  localparam NEG2 = 64'hc000000000000000;
  localparam TWO = 64'h4000000000000000;
  localparam ONE = 64'h3ff0000000000000;

  typedef enum logic [3:0] {
    IDLE = 4'b0000,
    WAIT_NEG2_MULT = 4'b0001,
    EXP_NEG2_X = 4'b0010,
    WAIT_EXP_NEG2_X = 4'b0011,
    ADD_ONE_DENOM = 4'b0100,
    WAIT_ADD_ONE_DENOM = 4'b0101,
    TWO_OVER_DENOM = 4'b0110,
    WAIT_TWO_OVER_DENOM = 4'b0111,
    MINUS_ONE = 4'b1000,
    WAIT_MINUS_ONE = 4'b1001,
    DONE = 4'b1010
  } tanh_state;
  
  tanh_state state;

  reg [63:0] res;
  reg [63:0] denom;

  reg [63:0] exp_in;

  reg [63:0] div_a;
  reg [63:0] div_b;

  reg [63:0] add_a;
  reg [63:0] add_b;

  reg [63:0] mult_a;
  reg [63:0] mult_b;

  reg start_mult;
  reg start_exp;
  reg start_add;
  reg start_div;

  wire div_done;
  wire [63:0] div_c;

  wire add_done;
  wire [63:0] add_c;

  wire exp_done;
  wire [63:0] exp_out;

  wire mult_done;
  wire [63:0] mult_c;

  fp_divider div (
    .clk(clk),
    .rst_n(rst_n),
    .a_in(div_a),
    .b_in(div_b),
    .start(start_div),
    .c_out(div_c),
    .done(div_done)
  );

  fp_adder add (
    .clk(clk),
    .rst_n(rst_n),
    .a_in(add_a),
    .b_in(add_b),
    .start(start_add),
    .c_out(add_c),
    .done(add_done)
  );

  fp_exp exp (
    .clk(clk),
    .rst_n(rst_n),
    .fp_in(exp_in),
    .start(start_exp),
    .fp_out(exp_out),
    .done(exp_done)
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

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      start_mult <= 1'b0;
      start_exp <= 1'b0;
      start_add <= 1'b0;
      start_div <= 1'b0;
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (start) begin
          mult_a <= fp_in;
          mult_b <= NEG2;
          start_mult <= 1'b1;
          state <= WAIT_NEG2_MULT;
        end
        WAIT_NEG2_MULT: begin
          if (mult_done) begin
            denom <= mult_c;
            state <= EXP_NEG2_X;
          end
          start_mult <= 1'b0;
        end
        EXP_NEG2_X: begin
          exp_in <= denom;
          start_exp <= 1'b1;
          state <= WAIT_EXP_NEG2_X;
        end
        WAIT_EXP_NEG2_X: begin
          if (exp_done) begin
            denom <= exp_out;
            state <= ADD_ONE_DENOM;
          end
          start_exp <= 1'b0;
        end
        ADD_ONE_DENOM: begin
          add_a <= denom;
          add_b <= ONE;
          start_add <= 1'b1;
          state <= WAIT_ADD_ONE_DENOM;
        end
        WAIT_ADD_ONE_DENOM: begin
          if (add_done) begin
            denom <= add_c;
            state <= TWO_OVER_DENOM;
          end
          start_add <= 1'b0;
        end
        TWO_OVER_DENOM: begin
          div_a <= TWO;
          div_b <= denom;
          start_div <= 1'b1;
          state <= WAIT_TWO_OVER_DENOM;
        end
        WAIT_TWO_OVER_DENOM: begin
          if (div_done) begin
            res <= div_c;
            state <= MINUS_ONE;
          end
          start_div <= 1'b0;
        end
        MINUS_ONE: begin
          add_a <= res;
          add_b <= NEG1;
          start_add <= 1'b1;
          state <= WAIT_MINUS_ONE;
        end
        WAIT_MINUS_ONE: begin
          if (add_done) begin
            res <= add_c;
            state <= DONE;
          end
          start_add <= 1'b0;
        end
        DONE: begin
          state <= IDLE;
        end
      endcase
    end
  end

  assign fp_out = res;
  assign done = (state == DONE);

endmodule
