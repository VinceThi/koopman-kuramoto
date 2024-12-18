#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Import the function from the external file *)
Get["C:\\Users\\thivi\\Documents\\GitHub\\koopman-kuramoto\\symbolic\\symmetries\\get_determining_equations.wls"];

(* One must first compile get_determining_equations.wls to have the functions in the global context *)

totalDegree=1;
nbNodes=5;
(*AMat={{I/4, 0, 0, 0, 0},
      {0.3Exp[0.1I], I/2, 0, 0, 0},
      {0.1Exp[0.2I], 0, I, 0, 0},
      {0.5Exp[0.3I], 0, 0, 2I, 1},
      {0, 0, 0, 1, 3I}};*)
  AMat={{I, 0, 0, 0, 0},
        {0, I/2, 0, Exp[I], 2.0 Exp[2I]},
        {Exp[I/2]/2, 0, I/4, 0, 0},
        {2.0 Exp[I], 0, 0, I, 0},
        {0, 0, 0, 0, I}};
          
      
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
(*Export["C:\\Users\\thivi\\Documents\\GitHub\\koopman-kuramoto\\symbolic\\symmetries\\determining_matrices\\ReImDetMatrix_N5_d1_pm1_2_phaselags.json",reimDetMatrix,"JSON"];*)

(* Print["Nonzero coefficients in ", nonZeroCoeffs]*)
Print["# of nonzero coefficients = ", Length[nonZeroCoeffs]]
Print["square determining matrix M dimension = ", Dimensions[DSquare]]
Print["rectangular determining matrix N dimension = ", Dimensions[NRectangle]]
(* Print[DSquare//TraditionalForm] *)
(* Print[NRectangle//TraditionalForm] *)
(* Print[DetMatrix//TraditionalForm] *)

singVals := SingularValueList[N[detMatrix]];  (* DetMatrix for symbolic and N[DetMatrix] for numerical *)
Print[singVals]
Print["There are ", Length[nonZeroCoeffs] - Length[singVals], " zero singular value(s)."]

(*Singular vectors for   AMat={{I, 0, 0, 0, 0}, 
                               {0, I/2, 0, Exp[I], 2.0 Exp[2I]},
                               {Exp[I/2]/2, 0, I/4, 0, 0},
                               {2.0 Exp[I], 0, 0, I, 0},
                               {0, 0, 0, 0, I}};    *)
                         




