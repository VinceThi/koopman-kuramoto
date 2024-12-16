#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Import the function from the external file *)
Get["C:\\Users\\thivi\\Documents\\GitHub\\koopman-kuramoto\\symbolic\\symmetries\\get_determining_equations.wls"];

(* One must first compile get_determining_equations.wls to have the functions in the global context *)

totalDegree=2;
nbNodes=4;
AMat={{I, 0, 0, 0},
      {1, I, 0, 0},
      {1, 0, I, 0},
      {1, 0, 0, I}};
pBounds={-4, 6};
determiningResult=getDeterminingEquations[AMat,totalDegree,nbNodes,pBounds];
detEqS=determiningResult[[1]];
detEqR=determiningResult[[2]];
pSub=determiningResult[[3]];
nbNodes=determiningResult[[4]];

result=getDeterminingMatrices[detEqS,detEqR,pSub,nbNodes];
DSquare = result[[1]];
NRectangle = result[[2]];
detMatrix = result[[3]];
nonZeroCoeffs =  result[[4]];
reimDetMatrix=Map[ReIm,detMatrix,{2}];
(*Export["C:\\Users\\thivi\\Documents\\GitHub\\koopman-kuramoto\\symbolic\\symmetries\\determining_matrices\\ReImDetMatrix_N4_d1_pm1_2.json",reimDetMatrix,"JSON"];*)

(* Print["Nonzero coefficients in ", nonZeroCoeffs]*)
Print["# of nonzero coefficients = ", Length[nonZeroCoeffs]]
Print["square determining matrix M dimension = ", Dimensions[DSquare]]
Print["rectangular determining matrix N dimension = ", Dimensions[NRectangle]]
(* Print[DSquare//TraditionalForm] *)
(* Print[NRectangle//TraditionalForm] *)
(* Print[DetMatrix//TraditionalForm] *)

singVals := SingularValueList[N[detMatrix]];  (* DetMatrix for symbolic and N[DetMatrix] for numerical *)
Print[singVals]
Print["There is (are) ", Length[nonZeroCoeffs] - Length[singVals], " zero singular value(s)."]



