module neuron
  #(
    INPUTS = 400,
    USE_TANH = 1
  )
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input start,
    input [63:0] data [INPUTS - 1:0],
    input write_weight,
    input [8:0] weight_sel,
    input [63:0] weight_bus,

    /*
     * Outputs
     */
    output [63:0] activation,
    output done
  );

  typedef enum logic [2:0] {
    IDLE = 3'b000,
    WEIGHT_PRODUCT = 3'b001,
    WAIT_WEIGHT_PRODUCT = 3'b010,
    ACCUMULATE = 3'b011,
    WAIT_ACCUMULATE = 3'b100,
    CALC_TANH = 3'b101,
    WAIT_CALC_TANH = 3'b110,
    DONE = 3'b111
  } neuron_state;

  neuron_state state;

  reg [$clog2(INPUTS)-1:0] cnt;

  reg [63:0] accum;
  reg [63:0] product;

  reg [63:0] bias;
  reg [63:0] weights [INPUTS-1:0];

  reg [63:0] tanh_in;

  reg [63:0] add_a;
  reg [63:0] add_b;

  reg [63:0] mult_a;
  reg [63:0] mult_b;

  reg start_mult;
  reg start_tanh;
  reg start_add;

  wire tanh_done;
  wire [63:0] tanh_out;

  wire add_done;
  wire [63:0] add_c;

  wire mult_done;
  wire [63:0] mult_c;

  fp_tanh tanh (
    .clk(clk),
    .rst_n(rst_n),
    .fp_in(tanh_in),
    .start(start_tanh),
    .fp_out(tanh_out),
    .done(tanh_done)
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
    .start(start_add),
    .c_out(add_c),
    .done(add_done)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (integer i = 0; i < INPUTS; i++) begin
        weights[i] <= '0;
      end
      state <= IDLE;
      start_mult <= 1'b0;
      start_tanh <= 1'b0;
      start_add <= 1'b0;
      accum <= '0;
      bias <= '0;
    end else begin
      case (state)
        IDLE: if (start) begin
          state <= WEIGHT_PRODUCT;
          accum <= bias;
          cnt <= '0;
        end
        WEIGHT_PRODUCT: begin
          state <= WAIT_WEIGHT_PRODUCT;
          mult_a <= data[cnt];
          mult_b <= weights[cnt];
          start_mult <= 1'b1;
        end
        WAIT_WEIGHT_PRODUCT: begin
          if (mult_done) begin
            product <= mult_c;
            state <= ACCUMULATE;
          end
          start_mult <= 1'b0;
        end
        ACCUMULATE: begin
          start_add <= 1'b1;
          add_a <= accum;
          add_b <= product;
          state <= WAIT_ACCUMULATE;
        end
        WAIT_ACCUMULATE: begin
          if (add_done) begin
            accum <= add_c;
            if (cnt + 1 == INPUTS) begin
              if (USE_TANH) begin
                state <= CALC_TANH;
              end else begin
                state <= DONE;
              end
            end else begin
              cnt <= cnt + 1;
              state <= WEIGHT_PRODUCT;
            end
          end
          start_add <= 1'b0;
        end
        CALC_TANH: begin
          state <= WAIT_CALC_TANH;
          start_tanh <= 1'b1;
          tanh_in <= accum;
        end
        WAIT_CALC_TANH: begin
          if (tanh_done) begin
            state <= DONE;
            accum <= tanh_out;
          end
          start_tanh <= 1'b0;
        end
        DONE: begin
          state <= IDLE;
        end
      endcase
      if (write_weight) begin
        if (weight_sel == 0) begin
          bias <= weight_bus;
        end else begin
          weights[weight_sel - 1] <= weight_bus;
        end
      end
    end
  end

  assign activation = accum;
  assign done = (state == DONE);

endmodule
