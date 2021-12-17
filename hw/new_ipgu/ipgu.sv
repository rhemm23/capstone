module ipgu (
  input clk,
  input rst_n,
  input data_ready,
  input [7:0] pixel_data [63:0],
  input initIpgu,
  input rdyHeu,

  output req_mem,
  output rdyIpgu,
  output vldIpgu,
  output [7:0] ipgu_q_buffer [4:0][79:0]
  );

  reg [7:0] ipgu_q_buffer [4:0][79:0];
  reg [7:0] data_reg [63:0];
  reg [5:0] pixel_cnt;
  reg [2:0] buffer_sel;
  reg [6:0] buffer_addr;

  typedef enum logic [1:0] {
    IDLE = 2'b00,
    STR_MEM = 2'b01,
    STR_PIXELS = 2'b10,
    DONE = 2'b11
  } ipgu_state;

  ipgu_state state;

  always_ff @(posedge clk, negedge rst_n) begin : ipgu_state_machine
    if (!rst_n) begin
      state <= IDLE;
      for (int i = 0; i < 64; i++) data_reg[i] <= 8'h00;
      pixel_cnt <= 6'b000000;
      buffer_sel <= 3'b000;
      buffer_addr <= 7'b0000000;
    end else begin
      case(state)
        IDLE: begin
          if (initIpgu) begin
            state <= STR_MEM;
          end
        end
        STR_MEM: begin
          if (data_ready) begin
            data_reg <= pixel_data;
            state <= STR_PIXELS;
          end else begin
            state <= STR_MEM;
          end
        end
        STR_PIXELS: begin
          if (buffer_sel == 3'b101) begin
            pixel_cnt <= 6'b000000;
            buffer_sel <= 3'b000;
            buffer_addr <= 7'b0000000;
            state <= DONE;
          end else if (buffer_addr == 80) begin
            buffer_sel <= buffer_sel + 1'b1;
            buffer_addr <= 7'b0000000;
            state <= STR_PIXELS;
          end else if (pixel_cnt == 6'b111111) begin
            ipgu_q_buffer[buffer_sel][buffer_addr] <= pixel_data[pixel_cnt];
            buffer_addr <= buffer_addr + 1'b1;
            pixel_cnt <= 6'b000000;
            state <= STR_MEM;
          end else begin
            ipgu_q_buffer[buffer_sel][buffer_addr] <= pixel_data[pixel_cnt];
            buffer_addr <= buffer_addr + 1'b1;
            pixel_cnt <= pixel_cnt + 1'b1;
            state <= STR_PIXELS;
          end
        end
        DONE: begin
          if (rdyHeu) begin
            state <= IDLE;
          end else begin
            state <= DONE;
          end
        end
      endcase
    end
  end

  assign req_mem = ((pixel_cnt == 6'b111111) && (state == STR_PIXELS)) ? 1'b1 : 1'b0;
  assign rdyIpgu = (state == IDLE) && (!initIpgu);
  assign vldIpgu = (state == DONE) && (!rdyHeu);

endmodule