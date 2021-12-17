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
    output   [7:0]  wrAllData [299:0][299:0],
    output            initIpgu,
    input             rdyIpgu,

    //rdn <-> ctrl_unit
    //weights
    input           rdnReqWeightMem,
    input           doneWeightRdn,
    output  [63:0]  rdn_weights [7:0],
    output          begin_rdn_load,
    output          weights_ready,

    //dnn <-> ctrl_unit
    input           dnnResVld,
    input   [511:0]dnnResults,
    output          dnnResRdy,
    output          begin_dnn_load,

    //weights
    input           dnnReqWeightMem,
    input           doneWeightDnn,
    output  [63:0]  dnn_weights [7:0]

);
    parameter NUM_INSTR = 16;


    wire en_pc, instrVld, halt;
    wire [31:0] instructionsIn [15:0];
    wire [31:0] instr;

    wire [1:0] reg_sel;
    wire reg_wr_en;
    wire [27:0] reg_databus;
    wire begin_proc;    
    /*
    Interconnection needed
        input incPc,                                  
        input instrVld,                               
        input [31:0] instructionsIn [NUM_INSTR-1:0];         
        output [31:0] instr;                    
    */
    instruction_fetch  instructionFetch (
        .clk(clk),
        .rst_n(rst_n),
        .en_pc(en_pc),
        .halt(halt),
        .instrVld(instrVld),
        .instructionsIn(instructionsIn),

        .instr(instr)
    );


     
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
    decode decode(
        .clk(clk),
        .rst_n(rst_n),
        .opcode(instr[31:28])),
        
        .reg_sel(reg_sel),
        .reg_wr_en(reg_wr_en),
        .halt(halt),
        .begin_rdn_load(begin_rdn_load),
        .begin_dnn_load(begin_dnn_load),
        .begin_proc(begin_proc)
    );


    //ctrl_unit
        
    //!!instructions = instructionsIn!!
    /*
    Interconncetion needed
        //ctrl_unit <-> instructionFetch
        output  reg       incPc,                //DONE
        output  [31:0]    instructions [NUM_INSTR-1:0],
        output  reg       instrVld,             //DONE
    */

    //!!reg_wr_en=wr_en!!
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
    
    ctrl_unit  ctrlUnit(
        .clk(clk),
        .rst_n(rst_n),
        .buffer_addr_valid(buffer_addr_valid),
        .data_valid(data_valid),
        .write_done(write_done),
        .read_data(read_data),
        .begin_rdn_load(begin_rdn_load),
        .begin_dnn_load(begin_dnn_load),
        .begin_proc(begin_proc),
        .reg_sel(reg_sel),
        .reg_wr_en(reg_wr_en),
        .reg_databus(reg_databus),
        .rdyIpgu(rdyIpgu),
        .rdnReqWeightMem(rdnReqWeightMem),
        .doneWeightRdn(doneWeightRdn),
        .dnnResVld(dnnResVld),
        .dnnResults(dnnResults),
        .dnnReqWeightMem(dnnReqWeightMem),
        .doneWeightDnn(doneWeightDnn),


        .address(address),
        .write_data(write_data),
        .read_request_valid(read_request_valid),
        .write_request_valid(write_request_valid),
        .en_pc(en_pc),
        .instructions(instructionsIn),
        .instrVld(instrVld),
        .wrAll(wrAll),
        .wrAllData(wrAllData),
        .initIpgu(initIpgu),
        .rdn_weights(rdn_weights),
        .weights_ready(weights_ready),
        .dnnResRdy(dnnResRdy),
        .dnn_weights(dnn_weights)
        ); 

endmodule
