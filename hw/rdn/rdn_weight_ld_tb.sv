module rdn_weight_ld_tb();

  logic clk, rst_n, go, mem_ready;
  logic signed [15:0] mem_data [31:0];

  logic signed [15:0] a_weight_bus;
  logic signed [15:0] b_weight_bus;
  logic signed [15:0] c_weight_bus;
  logic [$clog2(15)-1:0] a_sel;
  logic [$clog2(15)-1:0] b_sel;
  logic [$clog2(36)-1:0] c_sel;
  logic write_a;
  logic write_b;
  logic write_c;
  logic [8:0] a_weight_sel;
  logic [$clog2(15)-1:0] b_weight_sel;
  logic [$clog2(15)-1:0] c_weight_sel;
  logic weight_valid;
  logic req_mem;

rdn_weight_ld dut (
  /*
    * Inputs
    */
  .clk(clk),
  .rst_n(rst_n),
  .go(go),
  .mem_ready(mem_ready),
  .mem_data(mem_data),

  /*
    * logics
    */
  .a_weight_bus(a_weight_bus),
  .b_weight_bus(b_weight_bus),
  .c_weight_bus(c_weight_bus),
  .a_sel(a_sel),
  .b_sel(b_sel),
  .c_sel(c_sel),
  .write_a(write_a),
  .write_b(write_b),
  .write_c(write_c),
  .a_weight_sel(a_weight_sel),
  .b_weight_sel(b_weight_sel),
  .c_weight_sel(c_weight_sel),
  .weight_valid(weight_valid),
  .req_mem(req_mem)
  );
  
  integer i, j, word, weight, neuron;
  initial begin
    clk = 0;
    rst_n = 0;

    go = 0;
    mem_ready = 0;
    
    for (i = 0; i < 32; i = i + 1) mem_data[i] = 0;

    // reset
    @(posedge clk);
    rst_n = 1;

    go = 1;
    @(posedge clk);
    go = 0;

    fork
      begin : timeout1
        repeat(20000) @(posedge clk);
        $display("ERROR: timed out");
        $stop();
      end

      // Given input data when requested
      begin
        for (i = 0; i < 32; i = i + 1) mem_data[i] = $random;
        mem_ready = 1;
        while (!weight_valid) begin
          @(posedge clk);
          mem_ready = 0;
          if (req_mem) begin
            @(posedge clk);
            for (i = 0; i < 32; i = i + 1) mem_data[i] = $random;
            mem_ready = 1;
          end
        end
      end

      // Checks outputs
      begin
        // Checks results for A neurons
        for (neuron = 0; neuron < 15; neuron = neuron + 1) begin
          weight = 0;
          while (weight !== 401) begin
            @(posedge write_a);
            for (word = 0; word < 32; word = word + 1) begin
              if (weight === 401) break;
              @(posedge clk);
              if (a_sel !== neuron) begin
                $display("Error: Invalid output value for a_sel %h, expected %h", a_sel, neuron);
                $stop();
              end
              if (a_weight_sel !== weight) begin
                $display("Error: Invalid output value for a_weight_sel %h, expected %h", a_weight_sel, weight);
                $stop();
              end
              if (a_weight_bus !== mem_data[weight % 32]) begin
                $display("Error: Invalid output value for a_weight_bus %h, expected %h", a_weight_bus, mem_data[weight % 32]);
                $stop();
              end
              weight = weight + 1;
            end
          end
        end

        // Checks results for B neurons
        for (neuron = 0; neuron < 15; neuron = neuron + 1) begin
          @(posedge write_b);
          for (weight = 0; weight < 16; weight = weight + 1) begin
            @(posedge clk);
            if (b_sel !== neuron) begin
              $display("Error: Invalid output value for b_sel %h, expected %h", b_sel, neuron);
              $stop();
            end
            if (b_weight_sel !== weight) begin
              $display("Error: Invalid output value for b_weight_sel %h, expected %h", b_weight_sel, weight);
              $stop();
            end
            if (b_weight_bus !== mem_data[weight % 32]) begin
              $display("Error: Invalid output value for b_weight_bus %h, expected %h", b_weight_bus, mem_data[weight % 32]);
              $stop();
            end
          end
        end

        // Checks results for C neurons
        for (neuron = 0; neuron < 36; neuron = neuron + 1) begin
          @(posedge write_c);
          for (weight = 0; weight < 16; weight = weight + 1) begin
            @(posedge clk);
            if (c_sel !== neuron) begin
              $display("Error: Invalid output value for c_sel %h, expected %h", c_sel, neuron);
              $stop();
            end
            if (c_weight_sel !== weight) begin
              $display("Error: Invalid output value for c_weight_sel %h, expected %h", c_weight_sel, weight);
              $stop();
            end
            if (c_weight_bus !== mem_data[weight % 32]) begin
              $display("Error: Invalid output value for c_weight_bus %h, expected %h", c_weight_bus, mem_data[weight % 32]);
              $stop();
            end
          end
        end
        @(posedge weight_valid);
        if (weight_valid !== 1) begin
          $display("Error: Does not assert weight_valid, once done");
          $stop();
        end
        disable timeout1;
      end
    join

    $display("YAHOO!!! ALL tests PASSED!");
    $stop();
  end


  always
    #5 clk = ~clk;
endmodule