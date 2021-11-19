
/*
 * Package for all floating point operations
 */
package fp

  typedef struct packed {
    bit s;
    bit i;
    bit [8:0] f;
  } fp_1_7;

  typedef struct packed {
    bit s
    bit [2:0] i;
    bit [4:0] f;
  } fp_3_5;
  
endpackage
