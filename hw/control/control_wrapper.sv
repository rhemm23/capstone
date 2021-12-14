module control_wrapper 
(
    input clk,
    input rst_n,

    //controlWrapper(ctrl_unit) <-> memory
    input buffer_addr_valid,
    input data_valid,
    input write_done,
    input [511:0] read_data,
    output [31:0] address,
    output [511:0] write_data,
    output read_request_valid,
    output write_request_valid,

    //fdp(IPGU) <-> controlWrapper(ctrlUnit)
    output            wrAll,
    output   [8-1:0]  wrAllData [300-1:0][300-1:0],
    output            initIpgu,
    input             rdyIpgu,

    //rdn <-> ctrl_unit
    input           rdnResVld,
    input   [1085:0]rdnResults,
    output          rdnResRdy,
    //weights
    input           rdnReqWeightMem,
    input           doneWeightRdn,
    output  [63:0]  rdn_weights [7:0],

    //dnn <-> ctrl_unit
    input           dnnResVld,
    input   [1085:0]dnnResults,
    output  reg     dnnResRdy,
    //weights
    input           dnnReqWeightMem,
    input           doneWeightDnn,
    output  [63:0]  dnn_weights [7:0]

);

    wire incPc, instrVld;
    wire [31:0] instructionsIn [4095:0];
    wire [31:0] instr;
    /*
    Interconnection needed
        input incPc,                                  
        input instrVld,                               
        input [31:0] instructionsIn [4095:0];         
        output [31:0] instr;                    
    */
    instruction_fetch instructionFetch (.*);


    wire [1:0] reg_sel;
    wire wr_en;
    wire [27:0] reg_databus;
    wire begin_rdn_load, begin_dnn_load, begin_proc;     
    /*
    Interconncetion needed
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
    */
    decode decode(.*);


    //ctrl_unit
        
    //!!instructions = instructionsIn!!
    /*
    Interconncetion needed
        //ctrl_unit <-> instructionFetch
        output  reg       incPc,                //DONE
        output  [31:0]    instructions [4095:0],
        output  reg       instrVld,             //DONE
    */

    /*
    Interconncetion needed
        //decode -> ctrl_unit (cmds)
        input           begin_rdn_load,
        input           begin_dnn_load,
        input           begin_proc,     
        //decode -> ctrl_unit (registers)
        input   [1:0]   reg_sel,
        input           reg_wr_en,
        input   [27:0]  reg_databus,         
    */

    //defined as is out of the wrapper 
    /*
    Interconncetion needed
        //ctrl_unit <-> memory
        input buffer_addr_valid,
        input data_valid,
        input write_done,
        input [511:0] read_data,
        output reg [31:0] address,
        output [511:0] write_data,
        output reg read_request_valid,

        //IPGU <-> ctrlUnit
        output    reg     wrAll,
        output   [8-1:0]  wrAllData [300-1:0][300-1:0],
        output   reg      initIpgu,
        input             rdyIpgu,

        //rdn <-> ctrl_unit
        input           rdnResVld,
        input   [1085:0]   rdnResults,
        output  reg     rdnResRdy,
        //weights
        input           rdnReqWeightMem,
        input           doneWeightRdn,
        output  [63:0]  rdn_weights [7:0],

        //dnn <-> ctrl_unit
        input           dnnResVld,
        input   [1085:0]dnnResults,
        output  reg     dnnResRdy,
        //weights
        input           dnnReqWeightMem,
        input           doneWeightDnn,
        output  [63:0]  dnn_weights [7:0]
    */
    
    ctrl_unit ctrlUnit(.*, .instructions(instructionsIn)); 

endmodule
