#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Import the function from the external file *)
Get["C:\\Users\\thivi\\Documents\\GitHub\\koopman-kuramoto\\symbolic\\symmetries\\get_determining_equations.wls"];

totalDegree=1;
nbNodes=4;
(*AMat={{I,0, 0, 0},
        {0.3Exp[0.1I],2I,0, 0},
        {0.1Exp[0.2I],0, I/2,0},
        {0.5Exp[0.3I],0,0,3I/2}};*)
AMat={{I/2,0, 0, 0},
        {1, I/2,0, 0},
        {1,0, I/2,0},
        {1,0,0,I/2}};
pBounds={-1, 2};
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

(* Singular vectors for AMat={{I/2,0, 0, 0}, {1, I/2,0, 0}, {1,0, I/2,0},{1,0,0,I/2}};*)
v1 = {0,0,0,0,0,0,0,0,0,0,0,-0.65073,0,0,0,0,0,0,-0.21271,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.17584,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.01388,0,0,0.01388,0,0,0,0,0,0,0,0,0,0,0.01388,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.01388,0.17584,0.21271,0.65073,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
v2 ={0,0,0,0,0,0,0,0,0,0,0,0.25429,0,0,0,0,0,0,-0.61748,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.20608,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.07611,0,0,-0.07611,0,0,0,0,0,0,0,0,0,0,-0.07611,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.07611,0.20608,0.61748,-0.25429,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
v3 ={0,0,0,0,0,0,0,0,0,0,0,-0.09396,0,0,0,0,0,0,-0.25387,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.65305,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.01123,0,0,-0.01123,0,0,0,0,0,0,0,0,0,0,-0.01123,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.01123,-0.65305,0.25387,0.09396,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
v4 ={0,0,0,0,0,0,0,0,0,0,0,-0.05534,0,0,0,0,0,0,0.09495,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.,0,0,0,0,0,0.01196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.49385,0,0,-0.49385,0,0,0,0,0,0,0,0,0,0,-0.49385,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.49385,-0.01196,-0.09495,0.05534,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

Print[nonZeroCoeffs*v1]



