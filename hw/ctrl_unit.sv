module ctrl_unit #(IMG_SIZE=90000)
    (
      /*
       * Inputs
       */
      input clk,
      input rst_n,
      input buffer_addr_valid,
      input data_valid,
      input write_done,
      input [511:0] read_data,

      //rdn
      output  reg     [63:0]  rdn_weights [7:0],
      output  reg     [63:0]  dnn_weights [7:0],

      //decode -> ctrl_unit (cmds)
      input           begin_rdn_load,
      input           begin_dnn_load,
      input           begin_proc,     
      //decode -> ctrl_unit (registers)
      input   [1:0]   reg_sel,
      input           reg_wr_en,
      input   [27:0]  reg_databus,         
         
 
      //dnn -> ctrl_unit
      input           dnnResVld,
      input   [?:0]   dnnResults,
      output          dnnResRdy,

      /*
       * Outputs
       */
      output [31:0] address,
      output [511:0] write_data,
      output read_request_valid,
      output write_request_valid
    );

    typedef enum logic [2:0] {
      WAIT_BUFFER = 3'b000,
      FETCH_PROGRAM_PAGE = 3'b001,
      WAIT_PROGRAM_PAGE = 3'b010,
      EXECUTING = 3'b011,
      DONE = 3'b100
    } ctrl_unit_state;

    ctrl_unit_state state;

    reg [31:0] cnt;
    reg [31:0] instructions [4095:0];

    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        for (integer i = 0; i < 4096; i++) begin
          instructions[i] <= 32'h00000000;
        end
        state <= WAIT_BUFFER;
        cnt <= '0;
      end else begin
        state <= state;
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
            end else begin
              state <= FETCH_PROGRAM_PAGE;
              cnt <= cnt + 1;
            end
            for (integer i = 0; i < 16; i++) begin
              instructions[(cnt * 16) + i] <= read_data[(i * 32) +: 32];
            end
          end
          EXECUTING: begin
            if(begin_proc) begin
                state <= WAIT_IMAGE;
            end
            else if(begin_dnn_load) begin
                state <= 
            end
            else if(begin_rdn_load) begin

            end
          end
          WAIT_IMAGE: begin
            if(data_valid) begin
                //imagePageCnt is incremented
                if(imgPageCnt==1406) begin
                    state <= DONE_IMG;  
                end
            end
          end
          DONE_IMG: begin
            if(dnnResVld)
                state <= rnnResNumWritten==1 ? EXECUTING: state;
            else if(img_cnt!=0)
                state <= rdyIpgu ? WAIT_IMAGE : state;
          end 
        endcase
      end
    end

    assign dnnResRdy = (WAIT_IMAGE==state)&&data_valid&&(imgPageCnt==1046);
    assign write_data = {'0,dnnResults};
    assign write_request_valid = state==DONE_IMG&&dnnResVld;

    assign wrAll = rdyIpgu&& (state == DONE_IMG);

    //Img addr calc
    assign img_addr_inc = (state==WAIT_IMAGE) && data_valid;
    assign img_cnt_neg = (state==WAIT_IMAGE)&&data_valid&&(imgPageCnt==1406);
    assign rst_imgPageCnt = state==DONE_IMG;

    //Address
    assign address = (state==EXECUTING) ? {'0, reg_databus} : 
                            ((state==DONE_IMG) ? (dnnResVld?rslt_addr:img_addr) :
                            ((state==WAIT_IMAGE)?(img_addr+1):
                            cnt));

    assign read_request_valid = (EXECUTING==state&&(begin_proc|begin_rdn_load|begin_dnn_load))||
                                    (WAIT_IMAGE==state&&data_valid&&imgPageCnt!=1406)||
                                    (state==DONE_IMG&&!dnnResVld&&img_cnt!=0 && rdyIpgu)||
                                    (state == FETCH_PROGRAM_PAGE);


    en_img_fifo = (state == WAIT_IMAGE) && data_valid;
    out_fifo  #(.DATA_WIDTH(512), .Q_DEPTH(1407)) (.*, .en(en_img_fifo), .d(read_data), .q(img_data))); 
    
    //Number of RNN results written
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n)
            rnnResNumWritten <= '0;
        else if(reg_wr_en && reg_sel=='b01)
            rnnResNumWritten <= img_cnt;
        else if(write_request_valid)
            rnnResNumWritten <= rnnResNumWritten-1;
    end    
    
    //imgPageCnt
    always_ff @(posedge clk, negedge rst_n) begin
        if(!rst_n) 
            imgPageCnt <= '0;
        else if(rst_imgPageCnt)
            imgPageCnt <= '0;
        else if(en_img_fifo)
            imgPageCnt <= numImgPages+1;
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
    //always


endmodule
