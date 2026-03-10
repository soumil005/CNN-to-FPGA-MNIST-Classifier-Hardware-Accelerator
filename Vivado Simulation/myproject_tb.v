`timescale 1ns/1ps

module myproject_tb;

reg ap_clk;
reg ap_rst_n;
reg ap_start;
wire ap_done;
wire ap_idle;
wire ap_ready;

reg  [15:0] conv1_input_TDATA;
reg         conv1_input_TVALID;
wire        conv1_input_TREADY;

wire [159:0] layer12_out_TDATA;
wire         layer12_out_TVALID;
reg          layer12_out_TREADY;

myproject uut (
    .ap_clk            (ap_clk),
    .ap_rst_n          (ap_rst_n),
    .ap_start          (ap_start),
    .ap_done           (ap_done),
    .ap_idle           (ap_idle),
    .ap_ready          (ap_ready),
    .conv1_input_TDATA (conv1_input_TDATA),
    .conv1_input_TVALID(conv1_input_TVALID),
    .conv1_input_TREADY(conv1_input_TREADY),
    .layer12_out_TDATA (layer12_out_TDATA),
    .layer12_out_TVALID(layer12_out_TVALID),
    .layer12_out_TREADY(layer12_out_TREADY)
);

// 100MHz clock
initial ap_clk = 0;
always #5 ap_clk = ~ap_clk;

// Pixel data for digit 8
reg [15:0] pixels [0:63];
initial begin
    pixels[ 0]=16'h0000; pixels[ 1]=16'h0000; pixels[ 2]=16'h0000; pixels[ 3]=16'h0000;
    pixels[ 4]=16'h0000; pixels[ 5]=16'h0000; pixels[ 6]=16'h0000; pixels[ 7]=16'h0000;
    pixels[ 8]=16'h0000; pixels[ 9]=16'h0000; pixels[10]=16'h0000; pixels[11]=16'h0000;
    pixels[12]=16'h006A; pixels[13]=16'h0000; pixels[14]=16'h0000; pixels[15]=16'h0000;
    pixels[16]=16'h0000; pixels[17]=16'h0000; pixels[18]=16'h004C; pixels[19]=16'h03D2;
    pixels[20]=16'h014C; pixels[21]=16'h00D9; pixels[22]=16'h03AA; pixels[23]=16'h0000;
    pixels[24]=16'h0000; pixels[25]=16'h0000; pixels[26]=16'h001A; pixels[27]=16'h0351;
    pixels[28]=16'h0000; pixels[29]=16'h02CB; pixels[30]=16'h0290; pixels[31]=16'h0000;
    pixels[32]=16'h0000; pixels[33]=16'h0000; pixels[34]=16'h005F; pixels[35]=16'h0351;
    pixels[36]=16'h03C4; pixels[37]=16'h000A; pixels[38]=16'h0000; pixels[39]=16'h0000;
    pixels[40]=16'h0000; pixels[41]=16'h0374; pixels[42]=16'h0303; pixels[43]=16'h024D;
    pixels[44]=16'h03E5; pixels[45]=16'h0000; pixels[46]=16'h0000; pixels[47]=16'h0000;
    pixels[48]=16'h0000; pixels[49]=16'h0000; pixels[50]=16'h0000; pixels[51]=16'h0000;
    pixels[52]=16'h0000; pixels[53]=16'h0000; pixels[54]=16'h0000; pixels[55]=16'h0000;
    pixels[56]=16'h0000; pixels[57]=16'h0000; pixels[58]=16'h0000; pixels[59]=16'h0000;
    pixels[60]=16'h0000; pixels[61]=16'h0000; pixels[62]=16'h0000; pixels[63]=16'h0000;
end

integer i;
integer predicted_class;
reg [15:0] max_val;
reg [15:0] class_scores [0:9];

initial begin
    ap_rst_n           = 0;
    ap_start           = 0;
    conv1_input_TDATA  = 0;
    conv1_input_TVALID = 0;
    layer12_out_TREADY = 1;

    // Reset for 30 cycles
    repeat(30) @(posedge ap_clk);
    ap_rst_n = 1;
    repeat(10) @(posedge ap_clk);

    // Keep ap_start high for 10 cycles then release
    ap_start = 1;
    repeat(10) @(posedge ap_clk);
    ap_start = 0;

    $display("================================================");
    $display("CNN MNIST Testbench - True Label: 8");
    $display("================================================");
    $display("ap_idle=%b ap_ready=%b ap_done=%b", ap_idle, ap_ready, ap_done);

    // Stream all 64 pixels - try one per clock with TVALID always high
    for (i = 0; i < 64; i = i + 1) begin
        conv1_input_TDATA  = pixels[i];
        conv1_input_TVALID = 1;
        @(posedge ap_clk);
        // Wait if not ready
        while (conv1_input_TREADY == 0) begin
            @(posedge ap_clk);
        end
        $display("Pixel[%0d] sent: 0x%04h  TREADY=%b", i, pixels[i], conv1_input_TREADY);
    end
    conv1_input_TVALID = 0;
    conv1_input_TDATA  = 0;

    $display("All pixels sent. Waiting for layer12_out_TVALID...");

    // Wait up to 100000 cycles for output
    begin : wait_output
        integer timeout;
        timeout = 0;
        while (layer12_out_TVALID == 0 && timeout < 100000) begin
            @(posedge ap_clk);
            timeout = timeout + 1;
        end
        if (timeout >= 100000) begin
            $display("ERROR: Timed out waiting for output after 100000 cycles");
            $finish;
        end
    end

    @(posedge ap_clk);

    // Read output scores
    class_scores[0] = layer12_out_TDATA[15:0];
    class_scores[1] = layer12_out_TDATA[31:16];
    class_scores[2] = layer12_out_TDATA[47:32];
    class_scores[3] = layer12_out_TDATA[63:48];
    class_scores[4] = layer12_out_TDATA[79:64];
    class_scores[5] = layer12_out_TDATA[95:80];
    class_scores[6] = layer12_out_TDATA[111:96];
    class_scores[7] = layer12_out_TDATA[127:112];
    class_scores[8] = layer12_out_TDATA[143:128];
    class_scores[9] = layer12_out_TDATA[159:144];

    // Find max
    max_val = class_scores[0];
    predicted_class = 0;
    for (i = 1; i < 10; i = i + 1) begin
        if (class_scores[i] > max_val) begin
            max_val = class_scores[i];
            predicted_class = i;
        end
    end

    $display("================================================");
    $display("OUTPUT SCORES:");
    $display("  Class 0: 0x%04h", class_scores[0]);
    $display("  Class 1: 0x%04h", class_scores[1]);
    $display("  Class 2: 0x%04h", class_scores[2]);
    $display("  Class 3: 0x%04h", class_scores[3]);
    $display("  Class 4: 0x%04h", class_scores[4]);
    $display("  Class 5: 0x%04h", class_scores[5]);
    $display("  Class 6: 0x%04h", class_scores[6]);
    $display("  Class 7: 0x%04h", class_scores[7]);
    $display("  Class 8: 0x%04h", class_scores[8]);
    $display("  Class 9: 0x%04h", class_scores[9]);
    $display("================================================");
    $display("Predicted: %0d  |  True: 8", predicted_class);
    if (predicted_class == 8)
        $display("RESULT: CORRECT!");
    else
        $display("RESULT: WRONG");
    $display("================================================");

    #100;
    $finish;
end

// Global timeout
initial begin
    #100000000;
    $display("GLOBAL TIMEOUT");
    $finish;
end

endmodule
