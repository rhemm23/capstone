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
    output result,
    output done
  );

  localparam TWO_FIFTY_SIX = 64'h4070000000000000;

  typedef enum logic [] {
    IDLE = 4'b0000,
    NORM_CONV = 4'b0001,
    WAIT_NORM_CONV = 4'b0010,
    NORM_DIV = 4'b0011,
    WAIT_NORM_DIV = 4'b0100,
    START_A1
  } nn_state;

  nn_state state;

  reg [8:0] cnt;

  reg [7:0] data [399:0];
  reg [63:0] normalized_data [399:0];

  reg [63:0] conv_in;

  reg [63:0] div_a;
  reg [63:0] div_b;

  reg start_conv;
  reg start_div;
  reg start_a1;
  reg start_a2;

  reg [63:0] a1_activations [3:0];
  reg [63:0] a2_activations [15:0];
  reg [63:0] a3_activations [4:0];

  wire conv_done;
  wire [63:0] conv_out;

  wire div_done;
  wire [63:0] div_c;

  wire [99:0] a1_inputs [3:0];
  wire [24:0] a2_inputs [15:0];
  wire [79:0] a3_inputs [4:0];

  wire a1_neurons_done [3:0];
  wire a2_neurons_done [15:0];
  wire a3_neurons_done [4:0];

  wire [63:0] a1_activations_out [3:0];
  wire [63:0] a1_activations_out [15:0];
  wire [63:0] a1_activations_out [4:0];
  
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
      start_a1 <= 1'b0;
      start_a2 <= 1'b0;
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
              state <= ;
              cnt <= '0;
            end else begin
              cnt <= cnt + 1;
              state <= NORM_CONV;
            end
          end
          start_div <= 1'b0;
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
        .data(a1_inputs[i]),
        .write_weight((layer_sel == 0) && (neuron_sel == i) && write_weight),
        .weight_sel(weight_sel),
        .weight_bus(weight_bus),
        .activation(a1_activations_out[i]),
        .done(a1_neurons_done[i])
      );
    end
  endgenerate

endmodule
