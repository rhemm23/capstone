module bcau_ctrl_unit
(
  /*
   * Inputs
   */
  input clk,
  input rst_n,
  input iru_valid,
  input heu_ready,

  /*
   * Outputs
   */
  output logic bcau_valid,
  output logic bcau_ready,
  output logic wr_in_all,
  output logic cir_fifo,
  output logic wr_accum,
  output logic set_avg,
  output logic shft_out,
  output logic clr_accum,
  output logic [2:0] block_sel
);

reg [6:0] cnt;
reg [2:0] block_cnt;
reg [1:0] block_row_cnt;
wire inc_block_cnt;
logic clr_cnt, inc_cnt;

assign inc_block_cnt = (inc_cnt && (block_row_cnt == 2'b11));
assign block_sel = block_cnt;

// Counts the pixels either accumulated or processed
always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n || clr_cnt)
    cnt <= 7'h00;
    else if (inc_cnt)
    cnt <= cnt + 1;
end

// Counts the pixels either accumulated or processed
always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n || clr_cnt)
      block_cnt <= 3'h0;
    else if (inc_block_cnt && (block_cnt == 3'h4))
      block_cnt <= 3'h0;
    else if (inc_block_cnt)
      block_cnt <= block_cnt + 1;
end

// Counts the pixels either accumulated or processed
always_ff @(posedge clk, negedge rst_n) begin
    if (!rst_n || clr_cnt)
    block_row_cnt <= 2'h0;
    else if (inc_cnt)
    block_row_cnt <= block_row_cnt + 1;
end

typedef enum reg [1:0] {
  IDLE = 2'b00,
  ACCUM = 2'b01,
  CALC = 2'b10,
  DONE = 2'b11
} bcau_state;

bcau_state state, nxt_state;

always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n)
    state <= IDLE;
    else
    state <= nxt_state;
end

always_comb begin
  // defaults
  nxt_state = state;
  clr_cnt = 1'b0;
  inc_cnt = 1'b0;
  bcau_valid = 1'b0;
  bcau_ready = 1'b0;
  wr_in_all = 1'b0;
  cir_fifo = 1'b0;
  wr_accum = 1'b0;
  set_avg = 1'b0;
  shft_out = 1'b0;
  clr_accum = 1'b0;

  case (state)
    IDLE: 
      if (iru_valid) begin
        clr_cnt = 1'b1;
        wr_in_all = 1'b1;
        nxt_state = ACCUM;
      end else begin
        bcau_ready = 1'b1;
      end
    ACCUM:
      if (cnt == 80) begin
        clr_cnt = 1'b1;
        set_avg = 1'b1;
        nxt_state = CALC;
      end else begin
        inc_cnt = 1'b1;
        cir_fifo = 1'b1;
        wr_accum = 1'b1;
      end
    CALC:
      if (cnt == 80) begin
        clr_cnt = 1'b1;
        bcau_valid = 1'b1;
        nxt_state = DONE;
      end else begin
        inc_cnt = 1'b1;
        shft_out = 1'b1;
        cir_fifo = 1'b1;
      end
    DONE: 
      if (heu_ready) begin
        bcau_ready = 1'b1;
        clr_accum = 1'b1;
        nxt_state = IDLE;
      end else begin
        bcau_valid = 1'b1;
      end
  endcase
end

endmodule