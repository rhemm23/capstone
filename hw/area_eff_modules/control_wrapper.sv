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
    output            image_data_ready,
    output   [7:0]  ipgu_data [63:0],
    output            initIpgu,
    input             rdyIpgu,
    input             ipgu_req_mem,

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
 
    
    instruction_fetch  instructionFetch (
        .clk(clk),
        .rst_n(rst_n),
        .en_pc(en_pc),
        .halt(halt),
        .instrVld(instrVld),
        .instructionsIn(instructionsIn),

        .instr(instr)
    );


     

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


    
    
    ctrl_unit2  ctrlUnit(
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
        .ipgu_req_mem(ipgu_req_mem),


        .address(address),
        .write_data(write_data),
        .read_request_valid(read_request_valid),
        .write_request_valid(write_request_valid),
        .en_pc(en_pc),
        .instructions(instructionsIn),
        .instrVld(instrVld),
        .image_data_ready(image_data_ready),
        .ipgu_data(ipgu_data),
        .initIpgu(initIpgu),
        .rdn_weights(rdn_weights),
        .weights_ready(weights_ready),
        .dnnResRdy(dnnResRdy),
        .dnn_weights(dnn_weights)
        ); 

endmodule
