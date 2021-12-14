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

  wire [99:0] a1_inputs [3:0];
  wire [24:0] a2_inputs [15:0];
  wire [79:0] a3_inputs [4:0];

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

  assign a1_inputs[0] = {
    data[0 +: 10],
    data[20 +: 10],
    data[40 +: 10],
    data[60 +: 10],
    data[80 +: 10],
    data[100 +: 10],
    data[120 +: 10],
    data[140 +: 10],
    data[160 +: 10],
    data[180 +: 10]
  };
  assign a1_inputs[1] = {
    data[10 +: 10],
    data[30 +: 10],
    data[50 +: 10],
    data[70 +: 10],
    data[90 +: 10],
    data[110 +: 10],
    data[130 +: 10],
    data[150 +: 10],
    data[170 +: 10],
    data[190 +: 10]
  };
  assign a1_inputs[2] = {
    data[200 +: 10],
    data[220 +: 10],
    data[240 +: 10],
    data[260 +: 10],
    data[280 +: 10],
    data[300 +: 10],
    data[320 +: 10],
    data[340 +: 10],
    data[360 +: 10],
    data[380 +: 10]
  };
  assign a1_inputs[3] = {
    data[210 +: 10],
    data[230 +: 10],
    data[250 +: 10],
    data[270 +: 10],
    data[290 +: 10],
    data[310 +: 10],
    data[330 +: 10],
    data[350 +: 10],
    data[370 +: 10],
    data[390 +: 10]
  };

  assign a2_inputs[0] = {
    data[0 +: 5],
    data[20 +: 5],
    data[40 +: 5],
    data[60 +: 5],
    data[80 +: 5]
  };
  assign a2_inputs[1] = {
    data[5 +: 5],
    data[25 +: 5],
    data[45 +: 5],
    data[65 +: 5],
    data[85 +: 5]
  };
  assign a2_inputs[2] = {
    data[10 +: 5],
    data[30 +: 5],
    data[50 +: 5],
    data[70 +: 5],
    data[90 +: 5]
  };
  assign a2_inputs[3] = {
    data[15 +: 5],
    data[35 +: 5],
    data[55 +: 5],
    data[75 +: 5],
    data[95 +: 5]
  };
  assign a2_inputs[4] = {
    data[100 +: 5],
    data[120 +: 5],
    data[140 +: 5],
    data[160 +: 5],
    data[180 +: 5]
  };
  assign a2_inputs[5] = {
    data[105 +: 5],
    data[125 +: 5],
    data[145 +: 5],
    data[165 +: 5],
    data[185 +: 5]
  };
  assign a2_inputs[6] = {
    data[110 +: 5],
    data[130 +: 5],
    data[150 +: 5],
    data[170 +: 5],
    data[190 +: 5]
  };
  assign a2_inputs[7] = {
    data[115 +: 5],
    data[135 +: 5],
    data[155 +: 5],
    data[175 +: 5],
    data[195 +: 5]
  };
  assign a2_inputs[8] = {
    data[200 +: 5],
    data[220 +: 5],
    data[240 +: 5],
    data[260 +: 5],
    data[280 +: 5]
  };
  assign a2_inputs[9] = {
    data[205 +: 5],
    data[225 +: 5],
    data[245 +: 5],
    data[265 +: 5],
    data[285 +: 5]
  };
  assign a2_inputs[10] = {
    data[210 +: 5],
    data[230 +: 5],
    data[250 +: 5],
    data[270 +: 5],
    data[290 +: 5]
  };
  assign a2_inputs[11] = {
    data[215 +: 5],
    data[235 +: 5],
    data[255 +: 5],
    data[275 +: 5],
    data[295 +: 5]
  };
  assign a2_inputs[12] = {
    data[300 +: 5],
    data[320 +: 5],
    data[340 +: 5],
    data[360 +: 5],
    data[380 +: 5]
  };
  assign a2_inputs[13] = {
    data[305 +: 5],
    data[325 +: 5],
    data[345 +: 5],
    data[365 +: 5],
    data[385 +: 5]
  };
  assign a2_inputs[14] = {
    data[310 +: 5],
    data[330 +: 5],
    data[350 +: 5],
    data[370 +: 5],
    data[390 +: 5]
  };
  assign a2_inputs[15] = {
    data[315 +: 5],
    data[335 +: 5],
    data[355 +: 5],
    data[375 +: 5],
    data[395 +: 5]
  };

  assign a3_inputs[0] = data[0 +: 80];
  assign a3_inputs[1] = data[80 +: 80];
  assign a3_inputs[2] = data[160 +: 80];
  assign a3_inputs[3] = data[240 +: 80];
  assign a3_inputs[4] = data[320 +: 80];

  assign b_activations[0] = b1_activation;
  assign b_activations[1] = b2_activation;
  assign b_activations[2] = b3_activation;

  assign done = (state == DONE);

  generate
    genvar i;
    for (i = 0; i < 4; i++) begin : g_a1
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
    genvar i;
    for (i = 0; i < 16; i++) begin : g_a2
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
    genvar i;
    for (i = 0; i < 5; i++) begin : g_a3
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
    .INPUTS(3)
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
