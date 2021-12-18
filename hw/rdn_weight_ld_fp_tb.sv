module rdn_weight_ld_fp_tb();

  logic clk, rst_n, go, mem_ready;
  logic [63:0] mem_data [7:0];

  logic [63:0] weight_bus;
  logic [1:0] layer_sel;
  logic [5:0] neuron_sel;
  logic write_weight;
  logic [8:0] weight_sel;
  logic weight_valid;
  logic req_mem;

rdn_weight_ld_fp dut (
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
  .weight_bus(weight_bus),
  .layer_sel(layer_sel),
  .neuron_sel(neuron_sel),
  .write_weight(write_weight),
  .weight_sel(weight_sel),
  .weight_valid(weight_valid),
  .req_mem(req_mem)
  );
  
  integer i, j, word, weight, neuron;
  initial begin
    clk = 0;
    rst_n = 0;

    go = 0;
    mem_ready = 0;
    
    for (i = 0; i < 8; i = i + 1) mem_data[i] = 0;

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
        for (i = 0; i < 8; i = i + 1) mem_data[i] = $random;
        mem_ready = 1;
        while (!weight_valid) begin
          @(posedge clk);
          mem_ready = 0;
          if (req_mem) begin
            @(posedge clk);
            for (i = 0; i < 8; i = i + 1) mem_data[i] = $random;
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
            @(posedge write_weight);
            for (word = 0; word < 8; word = word + 1) begin
              if (weight === 401) break;
              @(posedge clk);
              if (layer_sel !== 2'b00) begin
                $display("Error: Invalid output value for layer_sel %h, expected 2'b00", layer_sel);
                $stop();
              end
              if (neuron_sel !== neuron) begin
                $display("Error: Invalid output value for neuron_sel %h, expected %h", neuron_sel, neuron);
                $stop();
              end
              if (weight_sel !== weight) begin
                $display("Error: Invalid output value for weight_sel %h, expected %h", weight_sel, weight);
                $stop();
              end
              if (weight_bus !== mem_data[weight % 8]) begin
                $display("Error: Invalid output value for A weight_bus %h, expected %h", weight_bus, mem_data[weight % 8]);
                $stop();
              end
              weight = weight + 1;
            end
          end
        end

        // Checks results for B neurons
        for (neuron = 0; neuron < 30; neuron = neuron + 1) begin
          weight = 0;
          while (weight !== 15) begin
            @(posedge write_weight);
            for (word = 0; word < 8; word = word + 1) begin
              if (weight === 15) break;
              @(posedge clk);
              if (layer_sel !== 2'b01) begin
                $display("Error: Invalid output value for layer_sel %h, expected 2'b01", layer_sel);
                $stop();
              end
              if (neuron_sel !== neuron) begin
                $display("Error: Invalid output value for neuron_sel %h, expected %h", neuron_sel, neuron);
                $stop();
              end
              if (weight_sel !== weight) begin
                $display("Error: Invalid output value for weight_sel %h, expected %h", weight_sel, weight);
                $stop();
              end
              if (weight_bus !== mem_data[weight % 8]) begin
                $display("Error: Invalid output value for B weight_bus %h, expected %h", weight_bus, mem_data[weight % 8]);
                $stop();
              end
              weight = weight + 1;
            end
          end
        end

        // Checks results for C neurons
        for (neuron = 0; neuron < 36; neuron = neuron + 1) begin
          weight = 0;
          while (weight !== 30) begin
            @(posedge write_weight);
            for (word = 0; word < 8; word = word + 1) begin
              if (weight === 30) break;
              @(posedge clk);
              if (layer_sel !== 2'b10) begin
                $display("Error: Invalid output value for layer_sel %h, expected 2'b10", layer_sel);
                $stop();
              end
              if (neuron_sel !== neuron) begin
                $display("Error: Invalid output value for neuron_sel %h, expected %h", neuron_sel, neuron);
                $stop();
              end
              if (weight_sel !== weight) begin
                $display("Error: Invalid output value for weight_sel %h, expected %h", weight_sel, weight);
                $stop();
              end
              if (weight_bus !== mem_data[weight % 8]) begin
                $display("Error: Invalid output value for C weight_bus %h, expected %h", weight_bus, mem_data[weight % 8]);
                $stop();
              end
              weight = weight + 1;
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

    $display("YAHOO!!! ALL RDN_WEIGHT_LD_FP Tests PASSED!");
    $stop();
  end


  always
    #5 clk = ~clk;
endmodule