module rot_nn_tb();

  logic clk;
  logic rst_n;

  logic start;

  logic [63:0] weight_data [7610:0];

  logic [7:0] data [399:0];

  logic write_weight;
  logic [1:0] layer_sel;
  logic [5:0] neuron_sel;
  logic [8:0] weight_sel;
  logic [63:0] weight_bus;

  wire [35:0] angle;
  wire done;

  int image_fd;

  int weight_fd;
  int weight_index;

  int layer_sizes [];
  int input_sizes [];

  rot_nn rot (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .data_in(data),
    .write_weight(write_weight),
    .layer_sel(layer_sel),
    .neuron_sel(neuron_sel),
    .weight_sel(weight_sel),
    .weight_bus(weight_bus),
    .angle(angle),
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

    weight_fd = $fopen("rnw.bin", "rb");

    if (weight_fd == 0) begin
      $display("Could not open weight file");
      $stop();
    end
    if ($fread(weight_data, weight_fd) != 60888) begin
      $display("Could not read from weight file");
      $stop();
    end

    weight_index = 0;

    layer_sizes = new [3];
    input_sizes = new [3];

    layer_sizes = '{ 15, 30, 36 };
    input_sizes = '{ 400, 15, 30 };

    for (integer i = 0; i < 3; i++) begin
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

    $display("%b", angle);
    $stop();

  end

  always
    #5 clk = ~clk;

endmodule
