// ***************************************************************************
// Copyright (c) 2013-2018, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// * Neither the name of Intel Corporation nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ***************************************************************************

// Module Name:  afu.sv
// Project:      ccip_mmio
// Description:  Implements an AFU with a single memory-mapped user register to demonstrate
//               memory-mapped I/O (MMIO) using the Core Cache Interface Protocol (CCI-P).
//
//               This module provides a simplified AFU interface since not all the functionality 
//               of the ccip_std_afu interface is required. Specifically, the afu module provides
//               a single clock, simplified port names, and all I/O has already been registered,
//               which is required by any AFU.
//
// For more information on CCI-P, see the Intel Acceleration Stack for Intel Xeon CPU with 
// FPGAs Core Cache Interface (CCI-P) Reference Manual

`include "platform_if.vh"

module ccip_mmio
  (
    /*
     * Inputs
     */
    input clk,
    input rst
    input t_if_ccip_Rx rx,

    /*
     * Outputs
     */
    output t_if_ccip_Tx tx
  );

  logic [127:0] afu_id = 128'h94dd2b43c2ad4c5c8d6a2098f9277842;

  t_ccip_c0_ReqMmioHdr mmio_hdr;

  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      tx.c0.hdr <= 0;
      tx.c0.valid <= 0;
      tx.c1.hdr <= 0;
      tx.c1.valid <= 0;
      tx.c2.hdr <= 0;
      tx.c2.mmioRdValid <= 0;
    end else begin

      /*
       * MMIO read request
       */
      if (rx.c0.mmioRdValid) begin

        /*
         * Echo TID
         */
        tx.c2.mmioRdValid <= 1'b1;
        tx.c2.hdr.tid <= mmio_hdr.tid;

        /*
         * Set response data
         */
        case (mmio_hdr.address)

          /*
           * AFU Header
           */
          16'h0000: tx.c2.data <= {
            4'b0001, // Feature type = AFU
            8'b0,    // reserved
            4'b0,    // afu minor revision = 0
            7'b0,    // reserved
            1'b1,    // end of DFH list = 1
            24'b0,   // next DFH offset = 0
            4'b0,    // afu major revision = 0
            12'b0    // feature ID = 0
          };

          /*
           * AFU ID lower
           */
          16'h0002: tx.c2.data <= afu_id[63:0];

          /*
           * AFU ID higher
           */
          16'h0004: tx.c2.data <= afu_id[127:64];

          /*
           * Reserved
           */
          16'h0006: tx.c2.data <= 64'h0;
          16'h0008: tx.c2.data <= 64'h0;

          /*
           * Unknown
           */
          default:  tx.c2.data <= 64'h0;
        endcase
      end else begin
        tx.c2.mmioRdValid <= 1'b0;
      end
    end
  end

  assign mmio_hdr = t_ccip_c0_ReqMmioHdr'(rx.c0.hdr);

endmodule
