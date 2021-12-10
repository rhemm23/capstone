module fp_mult_tb();

  logic clk;
  logic rst_n;

  logic [63:0] fp_in;

  logic start;

  wire [63:0] c;
  
  wire done;

  fp_exp exp (
    .clk(clk),
    .rst_n(rst_n),
    .fp_in(fp_in),
    .start(start),
    .fp_out(c),
    .done(done)
  );

  initial begin
    
    clk = 0;
    rst_n = 1;
    start = 0;

    rst_n = 0;
    #5;
    rst_n = 1;
    #5;

    fp_in = 64'h3ff0000000000000;
    start = 1;
    
    @(posedge clk);
    @(posedge done);

    $display("%e", $bitstoreal(c));
    $stop();

  end

  always
    #5 clk = ~clk;

endmodule
