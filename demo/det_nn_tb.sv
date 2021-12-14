module det_nn_tb();

  logic clk;
  logic rst_n;

  logic start;

  logic [63:0] weight_data [1256:0];

  logic [7:0] data [399:0];

  logic write_weight;

  logic [2:0] layer_sel;
  logic [3:0] neuron_sel;
  logic [6:0] weight_sel;
  logic [63:0] weight_bus;

  wire result;
  wire done;

  int image_fd;

  int weight_fd;
  int weight_index;

  int layer_sizes [];
  int input_sizes [];

  det_nn det (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data),
    .write_weight(write_weight),
    .layer_sel(layer_sel),
    .neuron_sel(neuron_sel),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .result(result),
    .done(done)
  );

  initial begin

    clk = 0;
    rst_n = 1;
    start = 0;
    write_weight = 0;

    #5;
    rst_n = 0;
    #5;
    rst_n = 1;
    #5;

    weight_fd = $fopen("dnw.bin", "rb");

    if (weight_fd == 0) begin
      $display("Could not open weight file");
      $stop();
    end
    if ($fread(weight_data, weight_fd) != 10056) begin
      $display("Could not read from weight file");
      $stop();
    end

    weight_index = 0;

    layer_sizes = new [7];
    input_sizes = new [7];

    layer_sizes = '{ 4, 16, 5, 1, 1, 1, 1 };
    input_sizes = '{ 100, 25, 80, 4, 16, 5, 3 };

    for (integer i = 0; i < 7; i++) begin
      for (integer j = 0; j < layer_sizes[i]; j++) begin

        layer_sel = i;
        neuron_sel = j;
        weight_sel = 0;
        write_weight = 1;
        weight_bus = weight_data[weight_index++];
        @(posedge clk);

        for (integer k = 0; k < input_sizes[i]; k++) begin

          layer_sel = i;
          neuron_sel = j;
          weight_sel = k + 1;
          write_weight = 1;
          weight_bus = weight_data[weight_index++];
          @(posedge clk);

        end
      end
    end

    write_weight = 0;

    @(posedge clk);
    @(posedge clk);

    image_fd = $fopen("image.bin", "rb");

    if (image_fd == 0) begin
      $display("Could not open image file");
      $stop();
    end
    if ($fread(data, image_fd) != 400) begin
      $display("Could not read from image file");
      $stop();
    end

    start = 1;
    @(posedge clk);
    start = 0;
    @(posedge done);

    $display("%b", result);
    $stop();

  end

  always
    #5 clk = ~clk;

endmodule
