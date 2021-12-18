module ctrl_unit
    (
      /*
       * Inputs
       */
      input clk,
      input rst_n,

      //ctrl_unit <-> memory
      input buffer_addr_valid,
      input data_valid,
      input write_done,
      input [511:0] read_data,
      output reg [31:0] address,
      output [511:0] write_data,
      output reg read_request_valid,
      output reg write_request_valid,

      //ctrl_unit <-> instructionFetch
      output logic      incPc, 
      output  [31:0]    instructions [15:0],
      output  reg       instrVld,

      //decode -> ctrl_unit (cmds)
      input           begin_rdn_load,
      input           begin_dnn_load,
      input           begin_proc,     
      //decode -> ctrl_unit (registers)
      input   [1:0]   reg_sel,
      input           reg_wr_en,
      input   [27:0]  reg_databus,         
       
      //IPGU <-> ctrlUnit
      output    reg     wrAll,
      output   [7:0]  wrAllData [299:0][299:0],
      output   reg      initIpgu,
      input             rdyIpgu,
 
      //rdn <-> ctrl_unit
        //weights
      input           rdnReqWeightMem,
      input           doneWeightRdn,
      output  [63:0]  rdn_weights [7:0],

      //dnn <-> ctrl_unit
      input           dnnResVld,
      input   [511:0] dnnResults,
      output  reg     dnnResRdy,
        //weights
      input           dnnReqWeightMem,
      input           doneWeightDnn,
      output  [63:0]  dnn_weights [7:0]
  );

  wire set_mem_addr, rst_imgPageCnt, inc_img_pg_cnt, img_cnt_neg, img_addr_inc;
  wire [31:0] mem_address;
  reg [11:0] pageCnt;
  reg [511:0] mem_data_reg;

  reg [27:0] img_addr, img_cnt, rslt_addr;
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
      img_addr <= 0;
      img_cnt <= 0;
      rslt_addr <= 0;
    end
    else if (reg_wr_en) begin
      img_addr <= img_addr;
      img_cnt <= img_cnt; 
      rslt_addr <= rslt_addr;
      if(reg_sel==2'b00) 
        img_addr <= reg_databus;
      else if(reg_sel==2'b01) 
        img_cnt <= reg_databus; 
      else if(reg_sel==2'b10) 
        rslt_addr <= reg_databus;
    end else begin
      if(img_cnt_neg) 
          img_cnt <= img_cnt-1;
      if(img_addr_inc)
          img_addr <= img_addr+1;
    end
  end

  //mem_data_reg
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n || rst_imgPageCnt) 
        mem_data_reg <= '0;
    else if(valid_data)
        mem_data_reg <=  read_data;
  end
  
  typedef enum logic [2:0] {
      WAIT_BUFFER,
      WAIT_PROGRAM_PAGE,
      EXECUTING,
      WAIT_RDN_LOAD,
      WAIT_DNN_LOAD,
      WAIT_IMAGE,
      DONE_IMG
    } ctrl_unit_state;

    ctrl_unit_state state, nxt_state;

  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
      address <= '0;
    end else if (set_mem_addr) begin
      address <= mem_address;
    end else if (inc_mem_addr) begin
      address <= address + 1'b1;
    end
  end

  //imgPageCnt
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n || rst_imgPageCnt) 
        pageCnt <= '0;
    else if(inc_img_pg_cnt)
        pageCnt <= pageCnt + 1'b1;
  end

  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n)
      state <= WAIT_BUFFER;
    else
      state <= nxt_state;
  end

  always_comb begin
    // defaults
    nxt_state = state;
    dnnResRdy = 1'b0;
    wrAll = 1'b0;
    img_addr_inc = 1'b0;
    img_cnt_neg = 1'b0;
    rst_imgPageCnt = 1'b0;
    set_mem_addr = 1'b0;
    mem_address = 0;
    read_request_valid = 1'b0;
    write_request_valid = 1'b0;
    incRdnAddr = 1'b0;
    incDnnAddr = 1'b0;
    inc_mem_addr = 1'b0;
    inc_img_pg_cnt = 1'b0;
    instrVld = 1'b0;
    incPc = 1'b0;
    case(state)
      WAIT_BUFFER:
        if (buffer_addr_valid) begin
          read_request_valid = 1'b1;
          nxt_state = WAIT_PROGRAM_PAGE;
        end
      WAIT_PROGRAM_PAGE:
        if (data_valid) begin
          instrVld = 1'b1;
          inc_mem_addr = 1'b1;
          read_request_valid = 1'b1;
          nxt_state = EXECUTING;
        end
      EXECUTING:  begin
        if(begin_proc) begin
          rst_imgPageCnt = 1'b1; 
          incPc = 1'b1;
          nxt_state = WAIT_IMAGE;
        end
        else if(begin_dnn_load) begin
          rst_imgPageCnt = 1'b1;
          mem_address = reg_databus;
          set_mem_addr = 1'b1;
          incPc = 1'b1;
          nxt_state = REQ_DNN_LOAD;
        end
        else if(begin_rdn_load) begin
          rst_imgPageCnt = 1'b1;
          mem_address = reg_databus;
          set_mem_addr = 1'b1;
          incPc = 1'b1;
          nxt_state = REQ_RDN_LOAD;
        end
      end
      REQ_RDN_MEM: begin
        if (data_valid) begin
          
        end else begin
          read_request_valid = 1'b1;
        end
      end
      WAIT_RDN_LOAD: begin
        if(doneWeightRdn) begin
          incPc <= 1'b1;
          nxt_state = EXECUTING;
        end else if (data_valid) begin
          
        end
      end


    endcase
