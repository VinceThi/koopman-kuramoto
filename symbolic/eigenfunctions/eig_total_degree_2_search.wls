#!/usr/bin/env wolframscript
(* ::Package:: *)

ClearAll["Global`*"];

(* Import the function from the external file *)
Get["/Users/benja/Documents/GitHub/koopman-kuramoto/symbolic/eigenfunctions/get_determining_equations.wls"];

TotalDegree = 2;
NbNodes = 3;
AMat = Array[B,{NbNodes, NbNodes}];
pBounds = {-3, 3};
determiningresult = GetDeterminingEquations[AMat, TotalDegree, NbNodes, pBounds];
DetEqS = determiningresult[[1]];
DetEqR = determiningresult[[2]];
pSub= determiningresult[[3]];
nbNodes= determiningresult[[4]];

result = GetDeterminingMatrices[DetEqS, DetEqR, pSub,nbNodes];
Print[result[[1]] // TraditionalForm]
Print[result[[2]] // TraditionalForm]


