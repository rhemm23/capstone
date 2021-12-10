
`include "platform_if.vh"

module memory
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input read_request_valid,
    input write_request_valid,
    input [31:0] address,
    input [511:0] data_d,
    input t_if_ccip_Rx rx,

    /*
     * Outputs
     */
    output buffer_addr_valid,
    output data_valid,
    output write_done,
    output [511:0] data_q,
    output t_if_ccip_Tx tx
  );

  typedef enum logic [2:0] {
    IDLE = 3'b000,
    READ_REQUEST = 3'b001,
    WRITE_REQUEST = 3'b010,
    WAIT_READ_RESPONSE = 3'b011,
    WAIT_WRITE_RESPONSE = 3'b100
  } mem_state;

  mem_state state;

  reg [31:0] stored_address;
  reg [511:0] stored_data;

  wire [63:0] buffer_addr;

  csrs ctrl_regs (
    .clk(clk),
    .rst_n(rst_n),
    .rx(rx.c0),
    .buffer_addr(buffer_addr),
    .tx(tx.c2)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stored_address <= '0;
      stored_data <= '0;
      state <= IDLE;
      tx.c0 <= '0;
      tx.c1 <= '0;
    end else begin
      case (state)
        IDLE: begin
          if (read_request_valid) begin
            stored_address <= address;
            state <= READ_REQUEST;
          end else if (write_request_valid) begin
            stored_address <= address;
            stored_data <= data_d;
            state <= WRITE_REQUEST;
          end
        end
        READ_REQUEST: if (!rx.c0TxAlmFull) begin
          tx.c0.hdr <= '{
            eVC_VA,
            2'b00,
            eCL_LEN_1,
            eREQ_RDLINE_I,
            6'h00,
            t_ccip_clAddr'(buffer_addr + stored_address),
            16'h0000
          };
          tx.c0.valid <= 1;
          state <= WAIT_READ_RESPONSE;
        end
        WAIT_READ_RESPONSE: begin
          if (rx.c0.rspValid && rx.c0.hdr.resp_type == eRSP_RDLINE && rx.c0.hdr.mdata == 16'h0000) begin
            state <= IDLE;
          end
          tx.c0.valid <= 0;
        end
        WRITE_REQUEST: if (!rx.c1TxAlmFull) begin
          tx.c1.hdr <= '{
            0,
            eVC_VA,
            1'b1,
            eMOD_CL,
            eCL_LEN_1,
            eREQ_WRLINE_I,
            0,
            t_ccip_clAddr'(buffer_addr + stored_address),
            16'h0000
          };
          tx.c1.valid <= 1;
          tx.c1.data <= stored_data;
          state <= WAIT_WRITE_RESPONSE;
        end
        WAIT_WRITE_RESPONSE: begin
          if (rx.c1.rspValid && rx.c1.hdr.resp_type == eRSP_WRLINE && rx.c1.hdr.mdata == 16'h0000) begin
            state <= IDLE;
          end
          tx.c1.valid <= 0;
        end
      endcase
    end
  end

  assign write_done = (rx.c1.rspValid) &&
                      (rx.c1.hdr.resp_type == eRSP_WRLINE) &&
                      (rx.c1.hdr.mdata == 16'h0000) &&
                      (state == WAIT_WRITE_RESPONSE);

  assign data_valid = (rx.c0.rspValid) &&
                      (rx.c0.hdr.resp_type == eRSP_RDLINE) &&
                      (rx.c0.hdr.mdata == 16'h0000) &&
                      (state == WAIT_READ_RESPONSE);

  assign data = rx.c0.data;
  assign buffer_addr_valid = |buffer_addr;

endmodule
