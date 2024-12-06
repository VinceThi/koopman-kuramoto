#!/usr/bin/env wolframscript
(* ::Package:: *)

GenerateLexicoTuples[d_Integer, N_Integer, bounds_List] := Module[{vars, solutions},
  vars = Array[p, N]; (* Create variables p1, p2, ..., pN *)
  solutions = Solve[
    Total[vars] == d && 
    And @@ Thread[bounds[[1]] <= vars <= bounds[[2]]], 
    vars, 
    Integers
  ];
  SortBy[vars /. solutions, Identity] (* Sort solutions lexicographically *)
]
