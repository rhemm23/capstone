module instruction_fetch 
(
    input clk,
    input rst_n,
    input en_pc, 
    input halt,
    input instrVld,
    input [31:0] instructionsIn [15:0],
    
    output [31:0] instr
);

reg [7:0] pc_addr;
reg [31:0] instructions [15:0];

always_ff @(posedge clk, negedge rst_n)begin
  if(!rst_n)
    pc_addr <= 8'h00;
  else if (!halt)
    pc_addr <= pc_addr + 1;
end

always_ff @(posedge clk, negedge rst_n) begin
  if(!rst_n)
    instructions <= '{16{32'h00000000}};
  else if(instrVld)
    instructions <= instructionsIn;
end

assign instr = (en_pc) ? instructions[pc_addr[3:0]]: 0;


endmodule
