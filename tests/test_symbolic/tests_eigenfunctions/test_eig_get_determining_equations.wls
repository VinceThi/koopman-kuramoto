#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Import the function from the external file *)
Get["C:\\Users\\thivi\\Documents\\GitHub\\koopman-kuramoto\\symbolic\\symmetries\\get_determining_equations.wls"];

(* Test 1: L_0 must be a generator of symmetry, d = 1, N = 3, bounds = {0, 1} *)
TotalDegree = 1;
NbNodes = 3;
AMat = Array[B, {NbNodes, NbNodes}];
pBounds = {0, 1};
determiningresult = GetDeterminingEquations[AMat, TotalDegree, NbNodes, pBounds];
DetEqS = determiningresult[[1]];
DetEqR = determiningresult[[2]];
pSub = determiningresult[[3]];
nbNodes = determiningresult[[4]];

matrixresult = GetDeterminingMatrices[DetEqS, DetEqR, pSub, nbNodes];
DetMatrix = matrixresult[[3]];

L0eigenvector = Transpose[{{0, 0, 1, 0, 1, 0, 1, 0, 0}}];
test1 = VerificationTest[DetMatrixSquare . L0eigenvector, 
{{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}}];

(* Collect all tests into a test suite *)
Tests = {test1};

(* Run the tests and print the report *)
TestReport[Tests]
