
`include "platform_if.vh"

module ccip_std_afu (
    /*
     * Clocks
     */
    input logic pClk,
    input logic pClkDiv2,
    input logic pClkDiv4,
    input lgoic uClk_usr,
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
    pck_af2cp_sTx_q <= pck_af2cp_sTx;
  end

  /*
   * AFU CSR
   */
  

endmodule
