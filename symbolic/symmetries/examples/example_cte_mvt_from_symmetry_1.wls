#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Example of the directed star *)

f1[z1_] = I w1 z1; (* w1 neq w + 2 y *)
f2[z1_, z2_] = I w z2 + (x + I y) z1 - (x - I y) z2^2/z1;
f3[z1_, z3_] = I w z3 + (x + I y) z1 - (x - I y) z3^2/z1;
f4[z1_, z4_] = I w z4 + (x + I y) z1 - (x - I y) z4^2/z1;
f5[z1_, z5_] = I w z5 + (x + I y) z1 - (x - I y) z5^2/z1;
 
K[f_] := f1[z1] D[f, z1] + f2[z1, z2] D[f, z2] + f3[z1, z3] D[f, z3] + f4[z1, z4] D[f, z4] + f2[z1, z5] D[f, z5]

I1[z1_, z2_, z3_, z4_, z5_] =((z4 - z2)(z5 - z3))/((z4 - z3)(z5 - z2));
I2[z1_, z2_, z3_, z4_, z5_] = (f2[z1, z2] - I w1 z2) ((z5 - z3)(z4 - z5))/((z4 - z3)(z5 - z2)^2);
I3[z1_, z2_, z3_, z4_, z5_] = (f3[z1, z3] - I w1 z3) ((z4 - z2)(z5 - z4))/((z4 - z3)^2 (z5 - z2));
I4[z1_, z2_, z3_, z4_, z5_] = (f4[z1, z4] - I w1 z4) ((z2 - z3)(z5 - z3))/((z4 - z3)^2 (z5 - z2));
I5[z1_, z2_, z3_, z4_, z5_] = (f5[z1, z5] - I w1 z5) ((z4 - z2)(z3 - z2))/((z4 - z3)(z5 - z2)^2);

FullSimplify[K[I2[z1, z2, z3, z4, z5]]]
FullSimplify[K[I3[z1, z2, z3, z4, z5]]]
FullSimplify[K[I4[z1, z2, z3, z4, z5]]]
FullSimplify[K[I5[z1, z2, z3, z4, z5]]]

FullSimplify[I2[z1, z2, z3, z4, z5] + I3[z1, z2, z3, z4, z5] + I4[z1, z2, z3, z4, z5] + I5[z1, z2, z3, z4, z5]]

(*F[I1_, I2_, I3_] := a1*I1^b1 * I2^b2 * I3^b3+ a2*I1^b4 * I2^b5 * I3^b6 + a3*I1^b7 * I2^b8 * I3^b9
targetExpr = ((z3 - z1) (z4 - z2))/((z3 - z2) (z4 - z1));
sol = Solve[F[I1[z1, z2, z3, z4, z5], I2[z1, z2, z3, z4, z5], I3[z1, z2, z3, z4, z5]] == targetExpr, {a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8, b9}]
(*doesn't work*)*)

(*F[I1_, I2_, I3_, I5_] := a I1 + b I2 + c I3
sol = Solve[F[I1[z1, z2, z3, z4, z5], I2[z1, z2, z3, z4, z5], I3[z1, z2, z3, z4, z5], I5[z1, z2, z3, z4, z5]] == I4[z1, z2, z3, z4, z5], {a, b, c}]
(*doesn't work*)*)

J = D[
   {I1[z1, z2, z3, z4, z5], 
    I2[z1, z2, z3, z4, z5], 
    I3[z1, z2, z3, z4, z5], 
    I4[z1, z2, z3, z4, z5], 
    I5[z1, z2, z3, z4, z5]}, 
   {{z1, z2, z3, z4, z5}}
];

(* Compute the rank of the Jacobian matrix *)
MatrixRank[J]



(*Denominator[K[I2[z1, z2, z3, z4, z5]]]*)

(*FullSimplify[K[((z4 - z2)(z5 - z3))/((z4 - z3)(z5 - z2))]]  Test: must be 0*)

FullSimplify[K[((z3 - z1)(z4 - z2))/((z3 - z2)(z4 - z1))]]  (*Test: 0 iff w1 = w + 2 y*)

(*Collect[w1^2 z1^3-((-\[ImaginaryI] x+y) z1^2+w z1 z2+(\[ImaginaryI] x+y) z2^2) (w z1+(\[ImaginaryI] x+y) (z2+z5))+w1 z1 (y (-z1^2-z1 z2+z2^2+z1 z5)+\[ImaginaryI] x (z1^2-z1 z2+z2^2+z1 z5)), {z1}]*)



