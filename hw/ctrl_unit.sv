module ctrl_unit
  (
    /*
     * Inputs
     */
    input clk,
    input rst_n,
    input buffer_addr_valid,
    input data_valid,
    input [511:0] data,

    /*
     * Outputs
     */
    output [31:0] address,
    output request_valid
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

  wire [31:0] page_instructions [15:0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (integer i = 0; i < 4096; i++) begin
        instructons[i] <= 32'h00000000;
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
            state <= EXECUTING;
          end else begin
            state <= FETCH_PROGRAM_PAGE;
            cnt <= cnt + 1;
          end
          for (integer i = 0; i < 16; i++) begin
            instructions[(cnt * 16) + i] <= data[(i * 32) +: 32];
          end
        end
        EXECUTING: begin
          /* TODO */
        end
      endcase
    end
  end

  assign address = cnt;
  assign request_valid = (state == FETCH_PROGRAM_PAGE);

endmodule
