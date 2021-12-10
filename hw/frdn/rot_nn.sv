module rot_nn
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input start,
    input [7:0] data_in [399:0],
    input write_weight,
    input [1:0] layer_sel,
    input [5:0] neuron_sel,
    input [8:0] weight_sel,
    input [63:0] weight_bus,

    /*
     * Outputs
     */
    output [35:0] angle,
    output done
  );

  localparam TWO_FIFTY_SIX = 64'h4070000000000000;

  typedef enum logic [3:0] {
    IDLE = 4'b0000,
    NORM_CONV = 4'b0001,
    WAIT_NORM_CONV = 4'b0010,
    NORM_DIV = 4'b0011,
    WAIT_NORM_DIV = 4'b0100,
    START_A = 4'b0101,
    WAIT_A = 4'b0110,
    START_B = 4'b0111,
    WAIT_B = 4'b1000,
    START_C = 4'b1001,
    WAIT_C = 4'b1010,
    COMPARE_OUT = 4'b1011,
    WAIT_COMPARE_OUT = 4'b1100,
    DONE = 4'b1101
  } nn_state;

  nn_state state;

  reg [8:0] cnt;

  reg [63:0] temp_data;

  reg [7:0] data [399:0];
  reg [63:0] normalized_data [399:0];

  reg [63:0] out_max;
  reg [5:0] out_max_index;

  reg [63:0] conv_in;

  reg [63:0] div_a;
  reg [63:0] div_b;

  reg [63:0] add_a;
  reg [63:0] add_b;

  reg [63:0] a_activations [14:0];
  reg [63:0] b_activations [29:0];
  reg [63:0] c_activations [35:0];

  reg start_conv;
  reg start_div;
  reg start_add;
  reg start_a;
  reg start_b;
  reg start_c;

  wire a_done;
  wire [63:0] a_activations_out [14:0];
  
  wire b_done;
  wire [63:0] b_activations_out [29:0];

  wire c_done;
  wire [63:0] c_activations_out [35:0];

  wire conv_done;
  wire [63:0] conv_out;

  wire div_done;
  wire [63:0] div_c;

  wire add_done;
  wire [63:0] add_c;

  layer a (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_a),
    .write_weight((layer_sel == 0) && write_weight),
    .data(normalized_data),
    .neuron_sel(neuron_sel),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .activations(a_activations_out),
    .done(a_done)
  );

  layer #(
    .INPUTS(15),
    .NEURONS(30)
  ) b (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_b),
    .write_weight((layer_sel == 1) && write_weight),
    .data(a_activations),
    .neuron_sel(neuron_sel),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .activations(b_activations_out),
    .done(b_done)
  );

  layer #(
    .INPUTS(30),
    .NEURONS(36),
    .USE_TANH(0)
  ) c (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_c),
    .write_weight((layer_sel == 2) && write_weight),
    .data(b_activations),
    .neuron_sel(neuron_sel),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .activations(c_activations_out),
    .done(c_done)
  );

  long_to_fp conv (
    .clk(clk),
    .rst_n(rst_n),
    .long_in(conv_in),
    .start(start_conv),
    .fp_out(conv_out),
    .done(conv_done)
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
      start_conv <= 1'b0;
      start_div <= 1'b0;
      start_add <= 1'b0;
      start_a <= 1'b0;
      start_b <= 1'b0;
      start_c <= 1'b0;
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (start) begin
          state <= NORM_CONV;
          data <= data_in;
          cnt <= '0;
        end
        NORM_CONV: begin
          state <= WAIT_NORM_CONV;
          start_conv <= 1'b1;
          conv_in <= { 56'h0, data[cnt] };
        end
        WAIT_NORM_CONV: begin
          if (conv_done) begin
            state <= NORM_DIV;
            normalized_data[cnt] <= conv_out;
          end
          start_conv <= 1'b0;
        end
        NORM_DIV: begin
          state <= WAIT_NORM_DIV;
          div_a <= normalized_data[cnt];
          div_b <= TWO_FIFTY_SIX;
          start_div <= 1'b1;
        end
        WAIT_NORM_DIV: begin
          if (div_done) begin
            normalized_data[cnt] <= div_c;
            if (cnt + 1 == 400) begin
              state <= START_A;
            end else begin
              cnt <= cnt + 1;
              state <= NORM_CONV;
            end
          end
          start_div <= 1'b0;
        end
        START_A: begin
          state <= WAIT_A;
          start_a <= 1'b1;
        end
        WAIT_A: begin
          if (a_done) begin
            state <= START_B;
            a_activations <= a_activations_out;
          end
          start_a <= 1'b0;
        end
        START_B: begin
          state <= WAIT_B;
          start_b <= 1'b1;
        end
        WAIT_B: begin
          if (b_done) begin
            state <= START_C;
            b_activations <= b_activations_out;
          end
          start_b <= 1'b0;
        end
        START_C: begin
          state <= WAIT_C;
          start_c <= 1'b1;
        end
        WAIT_C: begin
          if (c_done) begin
            cnt <= '0;
            out_max <= c_activations_out[0];
            out_max_index <= 0;
            state <= COMPARE_OUT;
            c_activations <= c_activations_out;
          end
          start_c <= 1'b0;
        end
        COMPARE_OUT: begin
          start_add <= 1'b1;
          add_a <= c_activations[cnt];
          add_b <= { ~out_max[63], out_max[62:0] };
          state <= WAIT_COMPARE_OUT;
        end
        WAIT_COMPARE_OUT: begin
          if (add_done) begin
            if (~add_c[63]) begin
              out_max <= c_activations[cnt];
              out_max_index <= cnt;
            end
            if (cnt == 35) begin
              state <= DONE;
            end else begin
              state <= COMPARE_OUT;
              cnt <= cnt + 1;
            end
          end
          start_add <= 1'b0;
        end
        DONE: begin
          state <= IDLE;
        end
      endcase
    end
  end

  assign angle = (36'd1 << out_max_index);
  assign done = (state == DONE);

endmodule
