module ctrl_unit
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
    WRITE_PROGRAM_PAGE = 3'b011,
    WAIT_WRITE_PAGE = 3'b100,
    EXECUTING = 3'b101,
    DONE = 3'b110
  } ctrl_unit_state;

  ctrl_unit_state state;

  reg [31:0] cnt;
  reg [31:0] instructions [4095:0];

  wire [31:0] page_instructions [15:0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (integer i = 0; i < 4096; i++) begin
        instructions[i] <= 32'h00000000;
      end
      state <= WAIT_BUFFER;
      cnt <= '0;
    end else begin
      case (state)
        WAIT_BUFFER: if (buffer_addr_valid) begin
          state <= FETCH_PROGRAM_PAGE;
        end
        FETCH_PROGRAM_PAGE: begin
          state <= WAIT_PROGRAM_PAGE;
        end
        WAIT_PROGRAM_PAGE: if (data_valid) begin
          if (cnt == 255) begin
            state <= WRITE_PROGRAM_PAGE;
            cnt <= 0;
          end else begin
            state <= FETCH_PROGRAM_PAGE;
            cnt <= cnt + 1;
          end
          for (integer i = 0; i < 16; i++) begin
            instructions[(cnt * 16) + i] <= read_data[(i * 32) +: 32];
          end
        end
        WRITE_PROGRAM_PAGE: begin
          state <= WAIT_WRITE_PAGE;
        end
        WAIT_WRITE_PAGE: if (write_done) begin
          if (cnt == 255) begin
            state <= DONE;
          end else begin
            state <= WRITE_PROGRAM_PAGE;
            cnt <= cnt + 1;
          end
        end
      endcase
    end
  end

  assign write_data = { 512 { 1'b1 } };

  assign address = cnt;
  assign read_request_valid = (state == FETCH_PROGRAM_PAGE);
  assign write_request_valid = (state == WRITE_PROGRAM_PAGE);

endmodule
