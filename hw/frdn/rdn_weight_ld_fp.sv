module rdn_weight_ld_fp
  #(
    parameter NUM_A_NEURONS = 15,
    parameter NUM_B_NEURONS = 30,
    parameter NUM_C_NEURONS = 36
  )
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input go,
    input mem_ready,
    input [63:0] mem_data [7:0],

    /*
     * Outputs
     */
    output [63:0] weight_bus,
    output [1:0] layer_sel,
    output [5:0] neuron_sel,
    output logic write_weight,
    output [8:0] weight_sel,
    output logic weight_valid,
    output logic req_mem
  );

  // Control signals of state machine
  logic new_neuron;
  logic store_weights, write_a, write_b, write_c;
  logic clr_layer_reg, inc_layer;
  logic clr_neuron_cnt;

  // Sequential logic
  reg [1:0] layer_reg;
  reg [2:0] word_cnt;
  reg [8:0] weight_cnt;
  reg [5:0] neuron_cnt; 
  reg [63:0] weight_block [7:0];

  // combinational logic
  assign neuron_sel = neuron_cnt;
  assign write_weight = (write_a || write_b || write_c);
  assign layer_sel = layer_reg;

  // setting output data busses
  assign weight_bus = weight_block[word_cnt];
  assign weight_sel = weight_cnt;


  // Counts what word is being load from the mem block
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n || req_mem)
    word_cnt <= 3'h0;
    else if (write_weight)
    word_cnt <= word_cnt + 1;
  end

  // Counts the number of weights loaded for a single neuron
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n || new_neuron)
    weight_cnt <= 9'h000;
    else if (write_weight)
    weight_cnt <= weight_cnt + 1;
  end

  // Holds which layer is being loaded: A,B, or C 
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n || clr_layer_reg)
    layer_reg <= 2'b00;
    else if (inc_layer)
    layer_reg <= layer_reg + 1;
  end

  // Counts the number of neurons that are done in a layer
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n || clr_neuron_cnt)
    neuron_cnt <= 5'h00;
    else if (new_neuron)
    neuron_cnt <= neuron_cnt + 1;
  end

  integer i;
  // Holds the neuron's weight memory before loading the Neuron
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n)
      for (i = 0; i < 8; i = i + 1) weight_block[i] <= 0;
    else if (store_weights)
      weight_block <= mem_data;
  end

  // Different States
  typedef enum reg [3:0] {
    IDLE = 4'h0,
    GET_A_MEM = 4'h1,
    LD_A_WEIGHTS = 4'h2,
    NEW_A_NEURON = 4'h3,
    GET_B_MEM = 4'h4,
    LD_B_WEIGHTS = 4'h5,
    NEW_B_NEURON = 4'h6,
    GET_C_MEM = 4'h7,
    LD_C_WEIGHTS = 4'h8,
    NEW_C_NEURON = 4'h9
  } weightLD_state;
  
  weightLD_state state, nxt_state;

  // resets the state maching to IDLE and changes the state to nxt_state every clk cycle
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n)
    state <= IDLE;
    else
    state <= nxt_state;
  end

  always_comb begin
    // defaults
    nxt_state = state;
    new_neuron = 1'b0;
    store_weights = 1'b0;
    inc_layer = 1'b0;
    clr_layer_reg = 1'b0;
    clr_neuron_cnt = 1'b0;

    //outputs
    write_a = 1'b0;
    write_b = 1'b0;
    write_c = 1'b0;
    weight_valid = 1'b0;
    req_mem = 1'b0;
    casex(state) 
      IDLE:
        if(go) begin
          nxt_state = GET_A_MEM;
        end
      GET_A_MEM:
        if(mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_A_WEIGHTS;
        end
      LD_A_WEIGHTS:
        if (weight_cnt == 9'h191) begin
          new_neuron = 1'b1;
          nxt_state = NEW_A_NEURON;
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_a = 1'b1;
          nxt_state = GET_A_MEM;
        end else begin
          write_a = 1'b1;
        end
      NEW_A_NEURON:
        if (neuron_cnt == NUM_A_NEURONS) begin
          req_mem = 1'b1;
          inc_layer = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = GET_B_MEM;
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_A_MEM;
        end
      GET_B_MEM:
        if (mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_B_WEIGHTS;
        end
      LD_B_WEIGHTS:
        if (weight_cnt == (NUM_A_NEURONS + 1)) begin
          new_neuron = 1'b1;
          nxt_state = NEW_B_NEURON;
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_b = 1'b1;
          nxt_state = GET_B_MEM;
        end else begin
          write_b = 1'b1;
        end
      NEW_B_NEURON:
        if (neuron_cnt == NUM_B_NEURONS) begin
          req_mem = 1'b1;
          inc_layer = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = GET_C_MEM;
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_B_MEM;
        end
      GET_C_MEM:
        if (mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_C_WEIGHTS;
        end
      LD_C_WEIGHTS:
        if (weight_cnt == (NUM_B_NEURONS + 1)) begin
          new_neuron = 1'b1;
          nxt_state = NEW_C_NEURON;
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_c = 1'b1;
          nxt_state = GET_C_MEM;
        end else begin
          write_c = 1'b1;
        end
      NEW_C_NEURON:
        if (neuron_cnt == NUM_C_NEURONS) begin
          weight_valid = 1'b1;
          clr_layer_reg = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = IDLE;
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_C_MEM;
        end
      default : begin
        nxt_state = IDLE;
      end
    endcase
  end

endmodule
