module registers (
  input clk,
  input rst_n,
  input [1:0]  wr_reg_sel,
  input        reg_wr_en,
  input [27:0] reg_databus,
  input [1:0]  rd_reg_sel,
  input inc_img_addr,
  input inc_img_cnt,
  input inc_rslt_addr,
  input inc_weight_addr,

  output [31:0] out
);

  reg [27:0] img_addr, img_cnt, rslt_addr, weight_addr;

  case(rd_reg_sel)
    2'b00: out = {4'h0, img_addr};
    2'b01: out = {4'h0, img_cnt};
    2'b10: out = {4'h0, rslt_addr};
    2'b11: out = {4'h0, weight_addr};
  endcase
  
  //Image Addr Register
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n) 
      img_addr <= 28'h0000000;
    else if(reg_wr_en && wr_reg_sel==2'b00)
      img_addr <=  reg_databus;
    else if(!reg_wr_en && inc_img_addr)
      img_addr <= img_addr + 1'b1;
  end

  //Image Count Register
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n) 
      img_cnt <= 28'h0000000;
    else if(reg_wr_en && wr_reg_sel==2'b01)
      img_cnt <=  reg_databus;
    else if(!reg_wr_en && inc_img_cnt)
      img_cnt <= img_cnt + 1'b1;
  end

  //Result Addr Register
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n) 
      rslt_addr <= 28'h0000000;
    else if(reg_wr_en && wr_reg_sel==2'b10)
      rslt_addr <=  reg_databus;
    else if(!reg_wr_en && inc_rslt_addr)
      rslt_addr <= rslt_addr + 1'b1;
  end

  //Weight Addr Register
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n) 
      weight_addr <= 28'h0000000;
    else if(reg_wr_en && wr_reg_sel==2'b11)
      weight_addr <=  reg_databus;
    else if(!reg_wr_en && inc_weight_addr)
      weight_addr <= weight_addr + 1'b1;
  end

endmodule