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
    output [31:0] address,
    output [511:0] write_data,
    output read_request_valid,
    output reg write_request_valid,

    //ctrl_unit <-> instructionFetch
    output logic      en_pc, 
    output [31:0]    instructions [15:0],
    output            instrVld,

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
    output          weights_ready,

    //dnn <-> ctrl_unit
    input           dnnResVld,
    input   [511:0] dnnResults,
    output  reg     dnnResRdy,
      //weights
    input           dnnReqWeightMem,
    input           doneWeightDnn,
    output  [63:0]  dnn_weights [7:0]
  );

  wire inc_rslt_addr, inc_img_cnt, inc_img_addr, inc_weight_addr;
  wire rst_pg_cnt, inc_pg_cnt;
  reg [11:0] pageCnt;
  reg [511:0] mem_data_reg;

  wire [27:0] reg_out;
  wire [1:0] rd_reg_sel;

  registers regs (
    .clk(clk),
    .rst_n(rst_n),
    .wr_reg_sel(reg_sel),
    .reg_wr_en(reg_wr_en),
    .reg_databus(reg_databus),
    .rd_reg_sel(rd_reg_sel),
    .inc_img_addr(inc_img_addr),
    .inc_img_cnt(inc_img_cnt),
    .inc_rslt_addr(inc_rslt_addr),
    .inc_weight_addr(inc_weight_addr),

    .out(reg_out)
  );

  assign address = reg_out;

  // Connects Memory data to instructions
  generate
    for (genvar i = 0; i < 16; i++)
      assign instructions[i] = read_data[(i*32)-1:0];
  endgenerate

  // Connects Memory data to instructions
  generate
    for (genvar i = 0; i < 8; i++) begin
      assign rdn_weights[i] = read_data[(i*64)+:64];
      assign dnn_weights[i] = read_data[(i*64)+:64];
    end
  endgenerate


  //mem_data_reg
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n) 
        mem_data_reg <= '0;
    else if(valid_data)
        mem_data_reg <=  read_data;
  end

  //imgPageCnt
  always_ff @(posedge clk, negedge rst_n) begin
    if(!rst_n || rst_pg_cnt) 
        pageCnt <= 12'h000;
    else if(inc_pg_cnt)
        pageCnt <= pageCnt + 1'b1;
  end

  typedef enum logic [2:0] {
      WAIT_BUFFER,
      FETCH_PROGRAM_PAGE,

      EXECUTING,
      WAIT_RDN_LOAD,
      WAIT_DNN_LOAD,
      WAIT_IMAGE,
      DONE_IMG
    } ctrl_unit_state;

  ctrl_unit_state state, nxt_state;

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
    img_cnt_neg = 1'b0;
    read_request_valid = 1'b0;
    write_request_valid = 1'b0;
    incRdnAddr = 1'b0;
    incDnnAddr = 1'b0;
    instrVld = 1'b0;
    en_pc = 1'b0;

    // page Counter
    rst_imgPageCnt = 1'b0;
    inc_imgPageCnt = 1'b0;

    // Register Access
    rd_reg_sel = 2'b00;
    inc_img_addr = 1'b0;
    inc_img_cnt = 1'b0;
    inc_rslt_addr = 1'b0;
    inc_weight_addr = 1'b0;
    
    case(state)
      WAIT_BUFFER:
        if (buffer_addr_valid) begin
          read_request_valid = 1'b1;
          nxt_state = FETCH_PROGRAM_PAGE;
        end
      FETCH_PROGRAM_PAGE:
        if (data_valid) begin
          instrVld = 1'b1;
          nxt_state = EXECUTING;
        end
      EXECUTING:  begin
        if(begin_proc) begin
          rst_pg_cnt = 1'b1;
          rd_reg_sel = 2'b00;
          en_pc = 1'b1;
          read_request_valid = 1'b1;
          inc_pg_cnt = 1'b1;
          nxt_state = FETCH_IMAGE_PAGE;
        end
        else if(begin_dnn_load) begin
          rst_pg_cnt = 1'b1;
          rd_reg_sel = 2'b11;
          en_pc = 1'b1;
          read_request_valid = 1'b1;
          inc_pg_cnt = 1'b1;
          nxt_state = FETCH_DNN_PAGE;
        end
        else if(begin_rdn_load) begin
          rst_pg_cnt = 1'b1;
          rd_reg_sel = 2'b11;
          en_pc = 1'b1;
          read_request_valid = 1'b1;
          inc_pg_cnt = 1'b1;
          nxt_state = FETCH_RNN_PAGE;
        end else 
          en_pc = 1'b1;
      end
      FETCH_RNN_PAGE: begin
        if (data_valid) begin
          weights_ready = 1'b1;
          inc_weight_addr = 1'b1;
          nxt_state = STORE_RNN_WEIGHTS;
        end
      end
      STORE_RNN_WEIGHTS: begin
        if (data_valid) begin
          weights_ready
          nxt_state = _RNN_WEIGHTS;
        end
      end


    endcase
