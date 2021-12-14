module det_nn
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input start,
    input [7:0] data_in [399:0],
    input write_weight,
    input [2:0] layer_sel,
    input [3:0] neuron_sel,
    input [6:0] weight_sel,
    input [63:0] weight_bus,

    /*
     * Outputs
     */
    output reg result,
    output done
  );

  localparam TWO_FIFTY_SIX = 64'h4070000000000000;
  localparam SIG_THRESHOLD = 64'hbff193ea2d2fe3f3;

  typedef enum logic [4:0] {
    IDLE = 5'b00000,
    NORM_CONV = 5'b00001,
    WAIT_NORM_CONV = 5'b00010,
    NORM_DIV = 5'b00011,
    WAIT_NORM_DIV = 5'b00100,
    START_A1 = 5'b00101,
    WAIT_A1 = 5'b00110,
    START_A2 = 5'b00111,
    WAIT_A2 = 5'b01000,
    START_A3 = 5'b01001,
    WAIT_A3 = 5'b01010,
    START_B1 = 5'b01011,
    WAIT_B1 = 5'b01100,
    START_B2 = 5'b01101,
    WAIT_B2 = 5'b01110,
    START_B3 = 5'b01111,
    WAIT_B3 = 5'b10000,
    START_C = 5'b10001,
    WAIT_C = 5'b10010,
    SUB_THRESHOLD = 5'b10011,
    WAIT_SUB_THRESHOLD = 5'b10100,
    DONE = 5'b10101
  } nn_state;

  nn_state state;

  reg [8:0] cnt;

  reg [7:0] data [399:0];
  reg [63:0] normalized_data [399:0];

  reg [63:0] conv_in;

  reg [63:0] div_a;
  reg [63:0] div_b;

  reg [63:0] add_a;
  reg [63:0] add_b;

  reg start_conv;
  reg start_div;
  reg start_add;
  reg start_a1;
  reg start_a2;
  reg start_a3;
  reg start_b1;
  reg start_b2;
  reg start_b3;
  reg start_c;

  reg [63:0] a1_activations [3:0];
  reg [63:0] a2_activations [15:0];
  reg [63:0] a3_activations [4:0];

  reg [63:0] b1_activation;
  reg [63:0] b2_activation;
  reg [63:0] b3_activation;

  reg [63:0] c_activation;

  wire conv_done;
  wire [63:0] conv_out;

  wire div_done;
  wire [63:0] div_c;

  wire add_done;
  wire [63:0] add_c;

  wire [63:0] a1_inputs [3:0][99:0];
  wire [63:0] a2_inputs [15:0][24:0];
  wire [63:0] a3_inputs [4:0][79:0];

  wire a1_neurons_done [3:0];
  wire a2_neurons_done [15:0];
  wire a3_neurons_done [4:0];

  wire [63:0] a1_activations_out [3:0];
  wire [63:0] a2_activations_out [15:0];
  wire [63:0] a3_activations_out [4:0];

  wire [63:0] b1_activation_out;
  wire [63:0] b2_activation_out;
  wire [63:0] b3_activation_out;

  wire [63:0] c_activation_out;

  wire [63:0] b_activations [2:0];

  wire b1_done;
  wire b2_done;
  wire b3_done;

  wire c_done;

  fp_adder add (
    .clk(clk),
    .rst_n(rst_n),
    .a_in(add_a),
    .b_in(add_b),
    .start(start_add),
    .c_out(add_c),
    .done(add_done)
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

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      start_conv <= 1'b0;
      start_div <= 1'b0;
      start_add <= 1'b0;
      start_a1 <= 1'b0;
      start_a2 <= 1'b0;
      start_a3 <= 1'b0;
      start_b1 <= 1'b0;
      start_b2 <= 1'b0;
      start_b3 <= 1'b0;
      start_c <= 1'b0;
      result <= 1'b0;
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
              state <= START_A1;
              cnt <= '0;
            end else begin
              cnt <= cnt + 1;
              state <= NORM_CONV;
            end
          end
          start_div <= 1'b0;
        end
        START_A1: begin
          start_a1 <= 1'b1;
          state <= WAIT_A1;
        end
        WAIT_A1: begin
          if (a1_neurons_done[cnt]) begin
            if (cnt + 1 == 4) begin
              state <= START_A2;
              cnt <= '0;
            end else begin
              state <= START_A1;
              cnt <= cnt + 1;
            end
            a1_activations[cnt] <= a1_activations_out[cnt];
          end
          start_a1 <= 1'b0;
        end
        START_A2: begin
          start_a2 <= 1'b1;
          state <= WAIT_A2;
        end
        WAIT_A2: begin
          if (a2_neurons_done[cnt]) begin
            if (cnt + 1 == 16) begin
              state <= START_A3;
              cnt <= '0;
            end else begin
              state <= START_A2;
              cnt <= cnt + 1;
            end
            a2_activations[cnt] <= a2_activations_out[cnt];
          end
          start_a2 <= 1'b0;
        end
        START_A3: begin
          start_a3 <= 1'b1;
          state <= WAIT_A3;
        end
        WAIT_A3: begin
          if (a3_neurons_done[cnt]) begin
            if (cnt + 1 == 5) begin
              state <= START_B1;
              cnt <= '0;
            end else begin
              state <= START_A3;
              cnt <= cnt + 1;
            end
            a3_activations[cnt] <= a3_activations_out[cnt];
          end
          start_a3 <= 1'b0;
        end
        START_B1: begin
          state <= WAIT_B1;
          start_b1 <= 1'b1;
        end
        WAIT_B1: begin
          if (b1_done) begin
            state <= START_B2;
            b1_activation <= b1_activation_out;
          end
          start_b1 <= 1'b0;
        end
        START_B2: begin
          state <= WAIT_B2;
          start_b2 <= 1'b1;
        end
        WAIT_B2: begin
          if (b2_done) begin
            state <= START_B3;
            b2_activation <= b2_activation_out;
          end
          start_b2 <= 1'b0;
        end
        START_B3: begin
          state <= WAIT_B3;
          start_b3 <= 1'b1;
        end
        WAIT_B3: begin
          if (b3_done) begin
            state <= START_C;
            b3_activation <= b3_activation_out;
          end
          start_b3 <= 1'b0;
        end
        START_C: begin
          state <= WAIT_C;
          start_c <= 1'b1;
        end
        WAIT_C: begin
          if (c_done) begin
            state <= SUB_THRESHOLD;
            c_activation <= c_activation_out;
          end
          start_c <= 1'b0;
        end
        SUB_THRESHOLD: begin
          state <= WAIT_SUB_THRESHOLD;
          add_a <= c_activation;
          add_b <= SIG_THRESHOLD;
          start_add <= 1'b1;
        end
        WAIT_SUB_THRESHOLD: begin
          if (add_done) begin
            result <= ~add_c[63];
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

  generate
    for (genvar i = 0; i < 4; i++) begin : a1_g
      for (genvar j = 0; j < 10; j++) begin : a1i_g
        assign a1_inputs[i][(j * 10) +: 10] = normalized_data[((i / 2) * 200 + (i % 2) * 10) +: 10];
      end
    end
  endgenerate

  generate
    for (genvar i = 0; i < 16; i++) begin : a2_g
      for (genvar j = 0; j < 5; j++) begin : a2i_g
        assign a2_inputs[i][(j * 5) +: 5] = normalized_data[((i / 4) * 100 + (i % 4) * 5) +: 5];
      end
    end
  endgenerate

  generate
    for (genvar i = 0; i < 5; i++) begin : a3_g
      assign a3_inputs[i] = normalized_data[(i * 80) +: 80];
    end
  endgenerate

  assign b_activations[0] = b1_activation;
  assign b_activations[1] = b2_activation;
  assign b_activations[2] = b3_activation;

  assign done = (state == DONE);

  generate
    for (genvar i = 0; i < 4; i++) begin : g_a1
      neuron #(
        .INPUTS(100)
      ) a1 (
        .clk(clk),
        .rst_n(rst_n),
        .start((cnt == i) && start_a1),
        .data(a1_inputs[i]),
        .write_weight((layer_sel == 0) && (neuron_sel == i) && write_weight),
        .weight_sel(weight_sel),
        .weight_bus(weight_bus),
        .activation(a1_activations_out[i]),
        .done(a1_neurons_done[i])
      );
    end
  endgenerate

  generate
    for (genvar i = 0; i < 16; i++) begin : g_a2
      neuron #(
        .INPUTS(25)
      ) a2 (
        .clk(clk),
        .rst_n(rst_n),
        .start((cnt == i) && start_a2),
        .data(a2_inputs[i]),
        .write_weight((layer_sel == 1) && (neuron_sel == i) && write_weight),
        .weight_sel(weight_sel),
        .weight_bus(weight_bus),
        .activation(a2_activations_out[i]),
        .done(a2_neurons_done[i])
      );
    end
  endgenerate

  generate
    for (genvar i = 0; i < 5; i++) begin : g_a3
      neuron #(
        .INPUTS(80)
      ) a3 (
        .clk(clk),
        .rst_n(rst_n),
        .start((cnt == i) && start_a3),
        .data(a3_inputs[i]),
        .write_weight((layer_sel == 2) && (neuron_sel == i) && write_weight),
        .weight_sel(weight_sel),
        .weight_bus(weight_bus),
        .activation(a3_activations_out[i]),
        .done(a3_neurons_done[i])
      );
    end
  endgenerate

  neuron #(
    .INPUTS(4)
  ) b1 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_b1),
    .data(a1_activations),
    .write_weight((layer_sel == 3) && write_weight),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .activation(b1_activation_out),
    .done(b1_done)
  );

  neuron #(
    .INPUTS(16)
  ) b2 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_b2),
    .data(a2_activations),
    .write_weight((layer_sel == 4) && write_weight),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .activation(b2_activation_out),
    .done(b2_done)
  );

  neuron #(
    .INPUTS(5)
  ) b3 (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_b3),
    .data(a3_activations),
    .write_weight((layer_sel == 5) && write_weight),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .activation(b3_activation_out),
    .done(b3_done)
  );

  neuron #(
    .INPUTS(3),
    .USE_TANH(0)
  ) c (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_c),
    .data(b_activations),
    .write_weight((layer_sel == 6) && write_weight),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .activation(c_activation_out),
    .done(c_done)
  );

endmodule
