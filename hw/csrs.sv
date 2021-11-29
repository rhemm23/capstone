
`include "platform_if.vh"
`include "afu_json_info.vh"

module csrs
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input t_if_ccip_c0_Rx rx,

    /*
     * Outputs
     */
    output reg [63:0] buffer_addr,
    output t_if_ccip_c2_Tx tx
  );

  logic [127:0] afu_id = `AFU_ACCEL_UUID;

  t_ccip_c0_ReqMmioHdr mmio_hdr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      buffer_addr <= 0;
      tx <= '0;
    end else begin
      if (rx.mmioRdValid) begin
        tx.mmioRdValid <= 1;
        tx.hdr.tid <= mmio_hdr.tid;
        case (mmio_hdr.address)
          16'h0000: tx.data <= {
            4'b0001, // Feature type = AFU
            8'b0,    // reserved
            4'b0,    // afu minor revision = 0
            7'b0,    // reserved
            1'b1,    // end of DFH list = 1
            24'b0,   // next DFH offset = 0
            4'b0,    // afu major revision = 0
            12'b0    // feature ID = 0
          };
          16'h0002: tx.data <= afu_id[63:0];
          16'h0004: tx.data <= afu_id[127:64];
          16'h0006: tx.data <= 0;
          16'h0008: tx.data <= 0;
          16'h000A: tx.data <= buffer_addr;
          default: tx.data <= 0;
        endcase
      end else begin
        tx.mmioRdValid <= 0;
      end
      if (rx.mmioWrValid) begin
        case (mmio_hdr.address)
          16'h000A: buffer_addr <= rx.data;
        endcase
      end
    end
  end

  assign mmio_hdr = t_ccip_c0_ReqMmioHdr'(rx.hdr);

endmodule
