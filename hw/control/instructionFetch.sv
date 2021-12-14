module instructionFetch
(
    input clk,
    input rst_n,
    input incPc, 

    input instrVld,
    input [31:0] instructionsIn [4095:0];
    output [31:0] instruction;
);

wire halt;

always_ff @(posedge clk, negedge rst_n)
    if(!rst_n)
        pc_addr <= '0;
    else if (halt&!incPc)
        pc_addr <= pc_addr;
    else
        pc_addr <= pc_addr+1;

assign halt = instr[30]&instr[29];

always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n)
        instructons <= {4096{{4'b0110,28'b0}}};
    else if(instrVld)
        instructions <= instructionsIn;
end

assign instruction = instructionsIn[pc_addr];


endmodule
