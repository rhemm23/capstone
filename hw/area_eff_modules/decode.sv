module decode
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input [3:0] opcode,
   
    output logic [1:0] reg_sel,
    output logic reg_wr_en,
    output logic halt,        
    output logic begin_rdn_load,
    output logic begin_dnn_load,
    output logic begin_proc           
  );


    always_comb begin
      reg_sel = 2'b00;
      reg_wr_en = 1'b0;
      halt = 1'b0;
      begin_rdn_load = 1'b0;
      begin_dnn_load = 1'b0; 
      begin_proc = 1'b0;
      case(opcode)
        4'b0000: begin // halt
          halt = 1'b1;
        end
        4'b0001: begin // Set result addr
          reg_sel = 2'b10;
          reg_wr_en = 1'b1;
        end
        4'b0010: begin // Load RNN Weights
          reg_sel = 2'b11;
          reg_wr_en = 1'b1;
          begin_rdn_load = 1'b1;
        end
        4'b0011: begin // Load DNN Weights
          reg_sel = 2'b11;
          reg_wr_en = 1'b1;
          begin_dnn_load = 1'b1;
        end
        4'b0100: begin // Begin Proc
          reg_sel = 2'b00;
          reg_wr_en = 1'b1;
          begin_proc = 1'b1;
        end
        4'b0101: begin // Set Img count
          reg_sel = 2'b01;
          reg_wr_en = 1'b1;
        end
        default: begin
          halt = 1'b1;
        end
    end
    
    
endmodule
