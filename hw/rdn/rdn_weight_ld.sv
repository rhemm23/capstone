module rdn_weight_ld
  #(
    parameter NUM_A_NEURONS = 15,
    parameter NUM_B_NEURONS = 15,
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
    input signed [15:0] mem_data [31:0],

    /*
     * Outputs
     */
    output signed [15:0] a_weight_bus,
    output signed [15:0] b_weight_bus,
    output signed [15:0] c_weight_bus,
    output [$clog2(NUM_A_NEURONS):0] a_sel,
    output [$clog2(NUM_B_NEURONS):0] b_sel,
    output [$clog2(NUM_C_NEURONS):0] c_sel,
    output logic write_a,
    output logic write_b,
    output logic write_c,
    output [8:0] a_weight_sel,
    output [$clog2(NUM_A_NEURONS):0] b_weight_sel,
    output [$clog2(NUM_B_NEURONS):0] c_weight_sel,
    output logic weight_valid,
    output logic req_mem
  );

  // Control signals of state machine
  logic ld_weight;
  logic new_neuron, new_a_neuron, new_b_neuron, new_c_neuron;
  logic store_a_weights, store_b_weights, store_c_weights;

  // Sequential logic
  reg [4:0] word_cnt;
  reg [8:0] weight_cnt;
  reg [$clog2(NUM_A_NEURONS):0] a_cnt; 
  reg [$clog2(NUM_B_NEURONS):0] b_cnt;
  reg [$clog2(NUM_C_NEURONS):0] c_cnt;
  reg signed [15:0] a_weight_block [31:0];
  reg signed [15:0] b_weight_block  [NUM_A_NEURONS:0];
  reg signed [15:0] c_weight_block  [NUM_B_NEURONS:0];

  // combinational logic
  assign a_sel = a_cnt;
  assign b_sel = b_cnt;
  assign c_sel = c_cnt;
  assign new_neuron = (new_a_neuron || new_b_neuron || new_c_neuron);
  assign ld_weight = (write_a || write_b || write_c);

  // setting output data busses
  assign a_weight_bus = a_weight_block[word_cnt];
  assign b_weight_bus = b_weight_block[word_cnt];
  assign c_weight_bus = c_weight_block[word_cnt];
  assign a_weight_sel = weight_cnt;
  assign b_weight_sel = weight_cnt[$clog2(NUM_A_NEURONS):0];
  assign c_weight_sel = weight_cnt[$clog2(NUM_B_NEURONS):0];


  // Counts what word is being load from the mem block
  always_ff @(posedge clk, negedge rst_n) begin
      if (!rst_n || req_mem)
      word_cnt <= 5'b00000;
      else if (ld_weight)
      word_cnt <= word_cnt + 1;
  end

  // Counts the number of weights loaded for a single neuron
  always_ff @(posedge clk, negedge rst_n) begin
      if (!rst_n || new_neuron)
      weight_cnt <= 9'b000000000;
      else if (ld_weight)
      weight_cnt <= weight_cnt + 1;
  end

  // Counts the number of A neurons that are done
  always_ff @(posedge clk, negedge rst_n) begin
      if (!rst_n)
      a_cnt <= 0;
      else if (new_a_neuron)
      a_cnt <= a_cnt + 1;
  end

  // Counts the number of B neurons that are done
  always_ff @(posedge clk, negedge rst_n) begin
      if (!rst_n)
      b_cnt <= 0;
      else if (new_b_neuron)
      b_cnt <= b_cnt + 1;
  end

  // Counts the number of C neurons that are done
  always_ff @(posedge clk, negedge rst_n) begin
      if (!rst_n)
      c_cnt <= 0;
      else if (new_c_neuron)
      c_cnt <= c_cnt + 1;
  end

  integer i;
  // Holds the A neuron weight memory before loading the Neuron
  always_ff @(posedge clk, negedge rst_n) begin
      if (!rst_n)
        for (i = 0; i < 32; i = i + 1) a_weight_block[i] <= 0;
      else if (store_a_weights)
      a_weight_block <= mem_data;
  end

  // Holds the B neuron weight memory before loading the Neuron
  always_ff @(posedge clk, negedge rst_n) begin
      if (!rst_n)
      for (i = 0; i < 16; i = i + 1) b_weight_block[i] <= 0;
      else if (store_b_weights)
      b_weight_block <= mem_data[15:0];
  end

  // Holds the C neuron weight memory before loading the Neuron
  always_ff @(posedge clk, negedge rst_n) begin
      if (!rst_n)
      for (i = 0; i < 16; i = i + 1) c_weight_block[i] <= 0;
      else if (store_c_weights)
      c_weight_block <= mem_data[15:0];
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
    new_a_neuron = 1'b0;
    new_b_neuron = 1'b0;
    new_c_neuron = 1'b0;
    store_a_weights = 1'b0;
    store_b_weights = 1'b0;
    store_c_weights = 1'b0;

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
          store_a_weights = 1'b1;
          nxt_state = LD_A_WEIGHTS;
        end
      LD_A_WEIGHTS:
        if (weight_cnt == 9'h191) begin
          new_a_neuron = 1'b1;
          nxt_state = NEW_A_NEURON;
        end else if(word_cnt == 5'b11111) begin
          req_mem = 1'b1;
          write_a = 1'b1;
          nxt_state = GET_A_MEM;
        end else begin
          write_a = 1'b1;
        end
      NEW_A_NEURON:
        if (a_cnt == NUM_A_NEURONS) begin
          req_mem = 1'b1;
          nxt_state = GET_B_MEM;
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_A_MEM;
        end
      GET_B_MEM:
        if (mem_ready) begin
          store_b_weights = 1'b1;
          nxt_state = LD_B_WEIGHTS;
        end
      LD_B_WEIGHTS:
        if (weight_cnt == (NUM_A_NEURONS + 1)) begin
          new_b_neuron = 1'b1;
          nxt_state = NEW_B_NEURON;
        end else if(word_cnt == 5'b11111) begin
          req_mem = 1'b1;
          write_b = 1'b1;
          nxt_state = GET_B_MEM;
        end else begin
          write_b = 1'b1;
        end
      NEW_B_NEURON:
        if (b_cnt == NUM_B_NEURONS) begin
          req_mem = 1'b1;
          nxt_state = GET_C_MEM;
        end else begin
          req_mem = 1'b1;
          nxt_state = GET_B_MEM;
        end
      GET_C_MEM:
        if (mem_ready) begin
          store_c_weights = 1'b1;
          nxt_state = LD_C_WEIGHTS;
        end
      LD_C_WEIGHTS:
        if (weight_cnt == (NUM_B_NEURONS + 1)) begin
          new_c_neuron = 1'b1;
          nxt_state = NEW_C_NEURON;
        end else if(word_cnt == 5'b11111) begin
          req_mem = 1'b1;
          write_c = 1'b1;
          nxt_state = GET_C_MEM;
        end else begin
          write_c = 1'b1;
        end
      NEW_C_NEURON:
        if (c_cnt == NUM_C_NEURONS) begin
          weight_valid = 1'b1;
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
