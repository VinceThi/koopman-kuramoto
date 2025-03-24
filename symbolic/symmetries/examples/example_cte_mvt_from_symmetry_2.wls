#!/usr/bin/env wolframscript
(* ::Package:: *)

(*** Define the model ***)

$Assumptions = {Subscript[\[Omega], 1] \[Element] Reals, Subscript[\[Omega], 2] \[Element] Reals, Subscript[\[Omega], 6] \[Element] Reals};
A = {{I Subscript[\[Omega], 1]/2, 0, 0, 0, 0, 0},
     {A1, I Subscript[\[Omega], 2]/2, A3, A4, A5, 0},
     {A1, A2, I (Subscript[\[Omega], 2] + 2Im[A3 - A2])/2, A4, A5, 0},
     {A1, A2, A3, I (Subscript[\[Omega], 2] + 2Im[A4 - A2])/2, A5, 0},
     {A1, A2, A3, A4, I (Subscript[\[Omega], 2] + 2Im[A5 - A2])/2, 0},
     {A61, 0, 0, 0, 0, I Subscript[\[Omega], 6]/2}};
nbNodes = 6
zvar = Array[z, nbNodes]; 
K[f_] := Sum[Sum[(A[[j, k]]z[[k]] - Conjugate[A[[j, k]]]z[[j]]^2/z[[k]])D[f, z[[j]]], {k, 1, nbNodes}], {j, 1, nbNodes}]
(*FullSimplify[K[z[[2]]]]*)


(*** Define the cross-ratio and the symmetry generator ***)

crossRatio[z2_, z3_, z4_, z5_] = ((z4 - z2)(z5 - z3))/((z4 - z3)(z5 - z2));
(*FullSimplify[K[crossRatio[z[[2]],z[[3]],z[[4]],z[[5]]]]]  (* succeeded verification*) *)

S[f_] := Sum[Sum[(A[[j, k]]z[[k]] - Conjugate[A[[j, k]]]z[[j]]^2/z[[k]])D[f, z[[j]]], {k, 1, nbNodes}], {j, 2, 5}] - I Subscript[\[Omega], 1] Sum[z[[j]]D[f, z[[j]]], {j, 2, 5}]

FullSimplify[S[crossRatio[z[[2]],z[[3]],z[[4]],z[[5]]]]]

(* Conclusion: in this setup, the symmetry S does not generate a new integral of motion *)



