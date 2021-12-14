module ctrl_unit #(IMG_SIZE=90000, NUM_INSTR=4096)
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
      output  reg       incPc, 
      output  [31:0]    instructions [NUM_INSTR-1:0],
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
      input   [511:0]dnnResults,
      output  reg     dnnResRdy,
        //weights
      input           dnnReqWeightMem,
      input           doneWeightDnn,
      output  [63:0]  dnn_weights [7:0]
  );

    typedef enum logic [2:0] {
      WAIT_BUFFER,
      FETCH_PROGRAM_PAGE,
      WAIT_PROGRAM_PAGE,
      EXECUTING,
      WAIT_RDN_LOAD,
      WAIT_DNN_LOAD,
      WAIT_IMAGE,
      DONE_IMG
    } ctrl_unit_state;

    ctrl_unit_state state;

    reg [31:0] cnt;
    reg [27:0] img_addr, img_cnt, rslt_addr, rdn_addr, dnn_addr;

    reg [10:0] imgPageCnt;

    reg [29:0] rnnResNumWritten;

    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        state <= WAIT_BUFFER;
        cnt <= '0;
        instrVld <= '0;
        incPc <= '0;
      end else begin
        state <= state;
        instrVld <= '0;
        incPc <= '0;
        case (state)
          WAIT_BUFFER: if (buffer_addr_valid) begin
            state <= FETCH_PROGRAM_PAGE;
          end
          FETCH_PROGRAM_PAGE: begin
            state <= WAIT_PROGRAM_PAGE;
          end
          WAIT_PROGRAM_PAGE: if (data_valid) begin
            if (cnt == 255) begin
              state <= EXECUTING;
              instrVld <= 1'b1;
            end else begin
              state <= FETCH_PROGRAM_PAGE;
              cnt <= cnt + 1;
            end
          end
          EXECUTING: begin
            if(begin_proc) begin
                state <= WAIT_IMAGE;
            end
            else if(begin_dnn_load) begin
                state <= WAIT_DNN_LOAD;
            end
            else if(begin_rdn_load) begin
                state <= WAIT_RDN_LOAD;
            end
          end
          WAIT_RDN_LOAD: begin
            if(doneWeightRdn) begin
                state <= EXECUTING;
                incPc <= 1'b1;
            end
          end
          WAIT_DNN_LOAD:
            if(doneWeightDnn) begin
                state <= EXECUTING;
                incPc <= 1'b1;
            end
          WAIT_IMAGE: begin
            if(write_done)
                state <= DONE_IMG;
            else if(data_valid) begin
                //imagePageCnt is incremented
                if(imgPageCnt==1406) begin
                    state <= DONE_IMG;  
                end
            end
          end
          DONE_IMG: begin
            if(dnnResVld) begin
                state <= WAIT_IMAGE;
                if(rnnResNumWritten==1) begin
                     state <= EXECUTING;
                     incPc <= 1'b1;
                end
            end
            else if(img_cnt!=0)
                state <= rdyIpgu ? WAIT_IMAGE : state;
          end 
        endcase
      end
    end

    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n)
            initIpgu <= '0;
        else if(wrAll)
            initIpgu <= '1;
        else if(!rdyIpgu)
            initIpgu <= '0;

    end


    reg img_addr_inc;              
    reg img_cnt_neg;               
    reg rst_imgPageCnt;            
    reg incRdnAddr, incDnnAddr;
 
    assign write_data = dnnResults;

    always_comb begin
        dnnResRdy = '0;
        wrAll = '0;
        img_addr_inc = '0;
        img_cnt_neg = '0;
        rst_imgPageCnt = '0;
        address = cnt;
        read_request_valid = '0;
        write_request_valid = '0;
        incRdnAddr = '1;
        incDnnAddr = '1;
        case(state)
            FETCH_PROGRAM_PAGE:
                read_request_valid = '1;
            EXECUTING: begin
                address = {4'b0, reg_databus};
                if(begin_proc|begin_rdn_load|begin_dnn_load) begin
                    read_request_valid = '1;
                    rst_imgPageCnt = '1;
                end
            end
            WAIT_RDN_LOAD: begin
                address = {4'b0, rdn_addr};
                if(data_valid) begin
                    incRdnAddr = '1;
                end
                if(rdnReqWeightMem)
                    read_request_valid = '1;
            end    
            WAIT_DNN_LOAD: begin
                address = {4'b0, dnn_addr};
                if(data_valid) begin
                    incDnnAddr = '1;
                end
                if(rdnReqWeightMem)
                    read_request_valid = '1;
            end    
            WAIT_IMAGE: begin
                if(write_done)
                    dnnResRdy = '1;
                else if(data_valid) begin
                    img_addr_inc = '1;   
                    if(imgPageCnt==1406) begin
                        dnnResRdy = '1;
                        img_cnt_neg = '1;
                    end
                    else
                        read_request_valid = '1;
                end
                address = {4'h0, img_addr + 1};
            end
            DONE_IMG: begin
                wrAll = rdyIpgu;
                rst_imgPageCnt = '1;
                dnnResRdy = '1; //risky
                if(dnnResVld) begin
                    address = {4'h0, rslt_addr};
                    write_request_valid = '1;
                end
                else begin
                    address = {4'h0, img_addr};
                    if(img_cnt!=0)
                        read_request_valid = rdyIpgu; 
                end
            end
        endcase
    end

    wire [511:0] img_data [1406:0];

    generate 
        for(genvar i=0;i<8;i++) begin
            assign rdn_weights[i] = read_data[(i+1)*64-1-:64];
            assign dnn_weights[i] = read_data[(i+1)*64-1-:64];
        end
        for(genvar i=0;i<1407;i++) begin
            for(genvar j=0;j<64;j++) begin
                if((((i*64)+j)/300 <300) && (((i*64)+j)%300 <300))
                    assign wrAllData[((i*64)+j)/300][((i*64)+j)%300] = img_data[i][(j+1)*8-1-:8];
            end
        end
        for(genvar i=0;i<NUM_INSTR;i++) begin
              assign instructions[i] = img_data[i/16][(((i+1)*32-1)%512)-:32];
        end
    endgenerate

    out_fifo  #(.DATA_WIDTH(512), .Q_DEPTH(1407)) ctrlImgBuffer (.*, .en(data_valid), .d(read_data), .q(img_data)); 
    
    //Number of RNN results written
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n)
            rnnResNumWritten <= '0;
        else if(reg_wr_en && reg_sel=='b01)
            rnnResNumWritten <= (img_cnt<<1)+img_cnt; //every image has three pages of results
        else if(write_request_valid)
            rnnResNumWritten <= rnnResNumWritten-1;
    end    
    
    //imgPageCnt
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n) 
            imgPageCnt <= '0;
        else if(rst_imgPageCnt)
            imgPageCnt <= '0;
        else if(data_valid)
            imgPageCnt <= imgPageCnt+1;
    end

    //Registers  
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n) begin
            img_addr <= '0;
            img_cnt <= '0;
            rslt_addr <= '0;
        end
        else if (reg_wr_en) begin
            img_addr <= img_addr;
            img_cnt <= img_cnt; 
            rslt_addr <= rslt_addr;
            if(reg_sel=='b00) 
                img_addr <= reg_databus;
            else if(reg_sel=='b01) 
                img_cnt <= reg_databus; 
            else if(reg_sel=='b10) 
                rslt_addr <= reg_databus;
        end
        else begin
            img_addr <= img_addr;
            img_cnt <= img_cnt; 
            rslt_addr <= rslt_addr;
            if(img_cnt_neg) 
                img_cnt <= img_cnt-1;
            if(img_addr_inc)
                img_addr <= img_addr+1;
        end
    end
   
    //rdn and dnn address
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n) begin
            rdn_addr <= '0;
            dnn_addr <= '0;
        end
        else begin
            rdn_addr <= (begin_rdn_load)?reg_databus:(incRdnAddr?rdn_addr+1:rdn_addr);
            dnn_addr <= (begin_dnn_load)?reg_databus:(incDnnAddr?dnn_addr+1:dnn_addr);
        end
    end


endmodule
