module dnn_weight_ld
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input start,
    input mem_ready,
    input [63:0] mem_data [7:0],

    /*
     * Outputs
     */
    output [63:0] weight_bus,
    output [2:0] layer_sel,
    output [3:0] neuron_sel,
    output [6:0] weight_sel,
    output logic write_weight,
    output logic weight_valid,
    output logic req_mem
  );

  logic clr_layer_reg, inc_layer;
  logic  clr_neuron_cnt, new_neuron;
  logic store_weights;

  reg [2:0] word_cnt;
  reg [3:0] neuron_cnt;
  reg [6:0] weight_cnt;
  reg [2:0] layer_reg;

  reg [63:0] weight_block [7:0];

  // setting output data busses
  assign weight_bus = weight_block[word_cnt];
  assign weight_sel = weight_cnt;
  assign layer_sel = layer_reg;
  assign neuron_sel = neuron_cnt;

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
      weight_cnt <= 7'h00;
    else if (write_weight)
      weight_cnt <= weight_cnt + 1;
  end

  // Holds which layer is being loaded: A,B, or C 
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n || clr_layer_reg)
      layer_reg <= 3'h0;
    else if (inc_layer)
      layer_reg <= layer_reg + 1;
  end

  // Counts the number of neurons that are done in a layer
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n || clr_neuron_cnt)
      neuron_cnt <= 4'h0;
    else if (new_neuron)
      neuron_cnt <= neuron_cnt + 1;
  end

  // Holds the neuron's weight memory before loading the Neuron
  always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n)
      for (int i = 0; i < 8; i = i + 1) weight_block[i] <= 0;
    else if (store_weights)
      weight_block <= mem_data;
  end

  typedef enum logic [4:0] {
    IDLE = 5'h00,
    GET_A0_MEM = 5'h01,
    LD_A0_WEIGHTS = 5'h02,
    NEW_A0_NEURON = 5'h03,
    GET_A1_MEM = 5'h04,
    LD_A1_WEIGHTS = 5'h05,
    NEW_A1_NEURON = 5'h06,
    GET_A2_MEM = 5'h07,
    LD_A2_WEIGHTS = 5'h08,
    NEW_A2_NEURON = 5'h09,
    GET_B0_MEM = 5'h0a,
    LD_B0_WEIGHTS = 5'h0b,
    NEW_B0_NEURON = 5'h0c,
    GET_B1_MEM = 5'h0d,
    LD_B1_WEIGHTS = 5'h0e,
    NEW_B1_NEURON = 5'h0f,
    GET_B2_MEM = 5'h10,
    LD_B2_WEIGHTS = 5'h11,
    NEW_B2_NEURON = 5'h12,
    GET_C_MEM = 5'h13,
    LD_C_WEIGHTS = 5'h14,
  } ld_state;

  ld_state state, nxt_state;

  // resets the state maching to IDLE and changes the state to nxt_state every clk cycle
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n)
      state <= IDLE;
    else
      state <= nxt_state;
  end

  always_comb begin
    nxt_state = state;
    new_neuron = 1'b0;
    store_weights = 1'b0;
    inc_layer = 1'b0;
    clr_layer_reg = 1'b0;
    clr_neuron_cnt = 1'b0;

    write_weight = 1'b0;
    weight_valid = 1'b0;
    req_mem = 1'b0;
    casex(state)
      IDLE:
        if(start) begin
          clr_layer_reg = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = GET_A0_MEM;
        end
      GET_A0_MEM:
        if(mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_A0_WEIGHTS;
        end
      LD_A0_WEIGHTS:
        if (weight_cnt == 7'h65)  
          new_neuron = 1'b1;
          nxt_state = NEW_A0_NEURON; 
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_weight = 1'b1;
          nxt_state = GET_A0_MEM; 
        end else begin
          write_weight = 1'b1;
        end
      NEW_A0_NEURON: 
        if (neuron_cnt == 4) begin 
          req_mem = 1'b1;
          inc_layer = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = GET_A1_MEM; 
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_A0_MEM; 
        end

      GET_A1_MEM:
        if(mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_A1_WEIGHTS; 
        end
      LD_A1_WEIGHTS: 
        if (weight_cnt == 7'h1A)  
          new_neuron = 1'b1;
          nxt_state = NEW_A1_NEURON; 
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_weight = 1'b1;
          nxt_state = GET_A1_MEM; 
        end else begin
          write_weight = 1'b1;
        end
      NEW_A1_NEURON: 
        if (neuron_cnt == 16) begin 
          req_mem = 1'b1;
          inc_layer = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = GET_A2_MEM; 
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_A1_MEM;
        end

      GET_A2_MEM: 
        if(mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_A2_WEIGHTS; 
        end
      LD_A2_WEIGHTS: 
        if (weight_cnt == 7'h51)  
          new_neuron = 1'b1;
          nxt_state = NEW_A2_NEURON; 
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_weight = 1'b1;
          nxt_state = GET_A2_MEM; 
        end else begin
          write_weight = 1'b1;
        end
      NEW_A2_NEURON: 
        if (neuron_cnt == 5) begin 
          req_mem = 1'b1;
          inc_layer = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = GET_B0_MEM; 
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_A2_MEM;
        end

      GET_B0_MEM: 
        if(mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_B0_WEIGHTS; 
        end
      LD_B0_WEIGHTS: 
        if (weight_cnt == 7'h5)
          new_neuron = 1'b1;
          nxt_state = NEW_B0_NEURON;
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_weight = 1'b1;
          nxt_state = GET_B0_MEM; 
        end else begin
          write_weight = 1'b1;
        end
      NEW_B0_NEURON: 
        if (neuron_cnt == 1) begin
          req_mem = 1'b1;
          inc_layer = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = GET_B1_MEM; 
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_B0_MEM; 
        end

      GET_B1_MEM:
        if(mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_B1_WEIGHTS;
        end
      LD_B1_WEIGHTS:
        if (weight_cnt == 7'h11) 
          new_neuron = 1'b1;
          nxt_state = NEW_B1_NEURON; 
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_weight = 1'b1;
          nxt_state = GET_B1_MEM;
        end else begin
          write_weight = 1'b1;
        end
      NEW_B1_NEURON:
        if (neuron_cnt == 1) begin 
          req_mem = 1'b1;
          inc_layer = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = GET_B2_MEM;
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_B1_MEM;
        end

      GET_B2_MEM:
        if(mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_B2_WEIGHTS;
        end
      LD_B2_WEIGHTS: 
        if (weight_cnt == 7'h06)  
          new_neuron = 1'b1;
          nxt_state = NEW_B2_NEURON; 
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_weight = 1'b1;
          nxt_state = GET_B2_MEM;
        end else begin
          write_weight = 1'b1;
        end
      NEW_B2_NEURON:
        if (neuron_cnt == 1) begin 
          req_mem = 1'b1;
          inc_layer = 1'b1;
          clr_neuron_cnt = 1'b1;
          nxt_state = GET_C_MEM; 
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_B2_MEM; 
        end

      GET_C_MEM: 
        if(mem_ready) begin
          store_weights = 1'b1;
          nxt_state = LD_C_WEIGHTS; 
        end
      LD_C_WEIGHTS: 
        if (weight_cnt == 7'h04)
          new_neuron = 1'b1;
          nxt_state = NEW_C_NEURON;
        end else if(word_cnt == 3'b111) begin
          req_mem = 1'b1;
          write_weight = 1'b1;
          nxt_state = GET_C_MEM;
        end else begin
          write_weight = 1'b1;
        end
      NEW_C_NEURON:
        if (neuron_cnt == 1) begin
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
