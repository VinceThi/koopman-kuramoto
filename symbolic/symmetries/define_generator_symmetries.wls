#!/usr/bin/env wolframscript
(* ::Package:: *)

ClearAll["Global`*"];
Get["C:\\Users\\thivi\\Documents\\GitHub\\koopman-kuramoto\\symbolic\\symmetries\\generate_lexico_tuples.wls"];

kookuGenerator[z_, pd_, A_, N_]:= Module[{}, 
	Return[Sum[Sum[(A[[j, k]]z[[k]] - Conjugate[A[[j, k]]]Conjugate[z[[k]]]z[[j]]^2)pd[[j]], {k, 1, N}], {j, 1, N}]]];

kookuCoefficients[l_, p_, A_, N_] := Module[{},
	Return[Sum[A[[l, r]]KroneckerDelta[p, UnitVector[N, r]] - Conjugate[A[[l, r]]]KroneckerDelta[p, 2UnitVector[N, l]- UnitVector[N, r]], {r, 1, N}]]];
	
l0Generator[z_, pd_, N_]:= Module[{}, 
	Return[Sum[z[[j]]pd[[j]],{j, 1, N}]]];
	
l0Coefficients[l_, p_, N_] := Module[{}, 
	Return[KroneckerDelta[p, UnitVector[N, l]]]];

generatorArray[bounds_, OrderedCoeffs_, GeneratorCoeffs_, extraArgs___] := (*Only for degree one generator*)
 Module[{genArray},
  (* Ensure that the bounds on p_i include KookuGenerator and L0
   (considering there are no "holes" in pSubset"*)
  If[Not[bounds[[1]] <= -1 && bounds[[2]] >= 2],
   Return["Error: p_i values are out of the specified bounds and cannot include the generator
           of the Koopman operator for the Kuramoto model."]];
 
  (* Map over OrderedCoeffs to compute values using GeneratorCoeffs *)
  genArray = Map[
   With[{ell = # /. c[x_, _] :> x, parr = # /. c[_, y_] :> y}, (*Extract coefficient indices*)
    (* Evaluate the coefficient function for ell and p *)
    GeneratorCoeffs[ell, parr, extraArgs]] &, OrderedCoeffs];
 If[genArray === ConstantArray[0, Dimensions[genArray]],
   "Error: The array is equal to the zero array, probably meaning that pSubset
    do not contain p of total degree 1", genArray]
 ];
  
nbNodes = 4;
totalDegree = 1;
boundaries = {-1, 2};
(*Amat = Array[A, {nbNodes, nbNodes}];*)
(*A4 ={{I/2,0, 0, 0},
        {1,I/2,0, 0},
        {1,0, I/2,0},
        {1,0,0,I/2}};*)
A4={{I/4,0, 0, 0},
       {0.3Exp[0.1I],I/2,0, 0},
       {0.1Exp[0.2I],0, I,0},
       {0.5Exp[0.3I],0,0,2I}};        
       
pSub = generateLexicoTuples[totalDegree, nbNodes, boundaries];
ordCoeffs = generateCoefficients[nbNodes, pSub];
l0Array = generatorArray[boundaries, ordCoeffs, l0Coefficients, nbNodes]
kookuArray = generatorArray[boundaries, ordCoeffs, kookuCoefficients, A4, nbNodes]

zvar = Array[z, nbNodes];
partialdervar = Array[pd, nbNodes];
kookuGenerator[zvar, partialdervar, A4, nbNodes]
l0Generator[zvar, partialdervar, nbNodes]
(*pSubset = GenerateLexicoTuples[d, N, bounds];*)







