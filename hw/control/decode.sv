module decode
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    //instruction_fetch -> Decode
    input   [31:0]  instr,
    
    //ctrl_unit (registers) <- Decode
    output  [1:0]   reg_sel,
    output          wr_en,
    output  [27:0]  reg_databus,         
    //ctrl_unit (cmds) <- Decode
    output          begin_rdn_load,
    output          begin_dnn_load,
    output          begin_proc           
  );
    
    wire [2:0] opcode;
    
    assign opcode = instr[30:28];
    
    //opcode 000, 001, 010
    assign wr_en = opcode[2]==1'b0;
    assign reg_sel = opcode[1:0];
    
    //reg_data_bus
    assign reg_data_bus = instr[27:0];    

    assign begin_proc = opcode==3'b000;    

    //opcode 011 wasted    

    //opcode 100, 101
    assign begin_rdn_load = opcode == 3'b100;
    assign begin_dnn_load = opcode == 3'b101;
    
    
    
endmodule
