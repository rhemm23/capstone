// ***************************************************************************
// Copyright (c) 2013-2016, Intel Corporation
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
// Module Name :    ccip_std_afu
// Project :        ccip afu top
// Description :    This module instantiates CCI-P compliant AFU

// ***************************************************************************

`include "platform_if.vh"

module ccip_std_afu (
    /*
     * Clocks
     */
    input logic pClk,
    input logic pClkDiv2,
    input logic pClkDiv4,
    input logic uClk_usr,
    input logic uClk_usrDiv2,

    /*
     * CCI-P signals
     */
    input logic pck_cp2af_softReset,
    input logic [1:0] pck_cp2af_pwrState,
    input logic pck_cp2af_error,

    /*
     * CCI-P TX and RX
     */
    input t_if_ccip_Rx pck_cp2af_sRx,
    output t_if_ccip_Tx pck_af2cp_sTx
  );

  /*
   * Buffer CCI-P input signals
   */
  (* preserve *) logic pck_cp2af_softReset_q;
  (* preserve *) logic [1:0] pck_cp2af_pwrState_q;
  (* preserve *) logic pck_cp2af_error_q;

  (* preserve *) t_if_ccip_Rx pck_cp2af_sRx_q;
  (* preserve *) t_if_ccip_Tx pck_af2cp_sTx_q;

  always @(posedge pClk) begin
    pck_cp2af_softReset_q <= pck_cp2af_softReset;
    pck_cp2af_pwrState_q <= pck_cp2af_pwrState;
    pck_cp2af_error_q <= pck_cp2af_error;
    pck_cp2af_sRx_q <= pck_cp2af_sRx;
    pck_af2cp_sTx <= pck_af2cp_sTx_q;
  end

  /*
   * Instantiate AFU
   */
  afu afu (
    .clk(pClk),
    .rst_n(~pck_cp2af_softReset_q),
    .rx(pck_cp2af_sRx_q),
    .tx(pck_af2cp_sTx_q)
  );

endmodule
