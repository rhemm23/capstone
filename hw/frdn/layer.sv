module layer
  #(
    INPUTS = 400,
    NEURONS = 15,
    USE_TANH = 1
  )
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input start,
    input write_weight,
    input [63:0] data [INPUTS - 1:0],
    input [$clog2(NEURONS)-1:0] neuron_sel,
    input [$clog2(INPUTS + 1) - 1:0] weight_sel,
    input [63:0] weight_bus,

    /*
     * Outputs
     */
    output [63:0] activations [NEURONS-1:0],
    output done
  );

  typedef enum logic [1:0] {
    IDLE = 2'b00,
    START_NEURON = 2'b01,
    WAIT_NEURON = 2'b10,
    DONE = 2'b11
  } layer_state;

  layer_state state;

  wire [NEURONS-1:0] neurons_done;

  reg [8:0] cnt;
  reg [NEURONS-1:0] neuron_starts;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      neuron_starts <= '0;
      state <= IDLE;
    end else begin
      case (state)
        IDLE: if (start) begin
          state <= START_NEURON;
          neuron_starts <= '0;
          cnt <= '0;
        end
        START_NEURON: begin
          neuron_starts[cnt] <= 1'b1;
          state <= WAIT_NEURON;
        end
        WAIT_NEURON: begin
          if (neurons_done[cnt]) begin
            if (cnt + 1 == NEURONS) begin
              state <= DONE;
            end else begin
              cnt <= cnt + 1;
              state <= START_NEURON;
            end
          end
          neuron_starts[cnt] <= 1'b0;
        end
        DONE: begin
          state <= IDLE;
        end
      endcase
    end
  end

  generate
    genvar i;
    for (i = 0; i < NEURONS; i++) begin : neuron_g
      neuron #(
        .INPUTS(INPUTS),
        .USE_TANH(USE_TANH)
      ) neuron (
        .clk(clk),
        .rst_n(rst_n),
        .start(neuron_starts[i]),
        .data(data),
        .write_weight((neuron_sel == i) && write_weight),
        .weight_sel(weight_sel),
        .weight_bus(weight_bus),
        .activation(activations[i]),
        .done(neurons_done[i])
      );
    end
  endgenerate

  assign done = (state == DONE);

endmodule
