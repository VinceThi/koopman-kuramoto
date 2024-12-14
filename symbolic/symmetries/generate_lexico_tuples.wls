#!/usr/bin/env wolframscript
(* ::Package:: *)

generateLexicoTuples[d_, N_, bounds_] := Module[{vars, solutions},
  vars = Array[p, N]; (* Create variables p1, p2, ..., pN *)
  solutions = Solve[Total[vars] == d && And @@ Thread[bounds[[1]] <= vars <= bounds[[2]]], vars, Integers];
  Return[SortBy[vars /. solutions, Identity]]] (* Sort solutions lexicographically*)

generateCoefficients[N_, pSubset_] := Flatten[Table[c[ell, p], {p, pSubset}, {ell, 1, N}], 1] (* Flatten one level to get a single list *)
