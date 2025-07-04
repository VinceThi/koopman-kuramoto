#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*ClearAll["Global`*"];*)
(*Get["/Users/benja/Documents/GitHub/koopman-kuramoto/symbolic/eigenfunctions/eig_generate_lexico_tuples.wls"];*)
(**)
(*(* Get the determining equations for a given weight matrix, total degree d, number of oscillators N and bounds on the elements of each tuple p *)*)
(*GetDeterminingEquations[A_, d_, N_, bounds_] := Module[{pSubset, pRange, detEqSquare, detEqRectangle,  dc},*)
(*pSubset =GenerateLexicoTuples[d, N, bounds];  (* Indices to non zero coefficients, subset of Z^N in lexicographic order *)*)
(*pRange =GenerateLexicoTuples[d, N, bounds+ {-1,1}];  (* Max range for potentially nontrivial determining equations subset of Z^N in lexicographic order *)*)
(*c[p_]:=0/;Not[MemberQ[pSubset,p]];    (*Truncation of the coefficients*)*)
(*dc[p_] := (* Recurrence equation for a tuple p in a subset of Z^N and weight matrix A *)*)
(*Sum[Sum[c[p + UnitVector[N, j] - UnitVector[N, k]](A[[j,k]](p[[j]]+1) - Conjugate[A[[k,j]]](p[[k]] - 1)), {j,1, N}], {k,1,N}] ;*)
(*(*dc[p_] :=      (* Recurrence equation for a tuple p in a subset of Z^N and weight matrix A *)Sum[A[[ell, k]]c[k, p] + Conjugate[A[[ell, k]]]c[p - 2UnitVector[N,ell]+ 2UnitVector[N,k]] - 2Conjugate[A[[ell, k]]]c[ell, p - UnitVector[N,ell]+ UnitVector[N,k]] + Sum[(p[[j]]-1+KroneckerDelta[j,k])Conjugate[A[[j, k]]] c[p - UnitVector[N,j]+ UnitVector[N,k]] - (p[[j]]+1-KroneckerDelta[j,k])A[[j, k]] c[ell,p + UnitVector[N,j]- UnitVector[N,k]],{j, 1, N}], {k, 1, N}];*)*)
(**)
(*(* Get the first set of determining equations regarding the indices for nonzero coefficients in pSubset *)*)
(*detEqSquare = {};*)
(*Do[*)
(*AppendTo[detEqSquare, dc[q]];,*)
(*{q, pSubset}];*)
(**)
(*(* Get the second set of determining equations regarding the indices for zero coefficients in Complement[pRange,pSubset]*)*)
(*detEqRectangle = {};*)
(*Do[*)
(*currentEquation=dc[q];*)
(* If[currentEquation=!=0 &&Not[MemberQ[detEqRectangle,currentEquation]],*)
(*AppendTo[ detEqRectangle,currentEquation];];,*)
(*{q, Complement[pRange,pSubset]}];*)
(*Return[{detEqSquare, detEqRectangle, pSubset, N}]*)
(*];*)
(**)
(*GetDeterminingMatrices[DetEqS_, DetEqR_, pSubset_,N_]:=Module[{DetMatSquare, DetMatRectangle, DetMat, nonZeroCoefficients},*)
(*	nonZeroCoefficients = EigGenerateCoefficients[N, pSubset];*)
(*	DetMatSquare=Normal@CoefficientArrays[DetEqS, nonZeroCoefficients][[2]];*)
(*	DetMatRectangle=Normal@CoefficientArrays[DetEqR, nonZeroCoefficients][[2]];*)
(*  DetMat = Join[DetMatSquare,DetMatRectangle,1];*)
(*Return[{DetMatSquare, DetMatRectangle, DetMat, nonZeroCoefficients, pSubset, N}]*)
(*];*)
(**)
(*TotalDegree=3;*)
(*NbNodes=3;*)
(*(* AMat=Array[B,{NbNodes,NbNodes}] *)*)
(*(*AMat={{I,0, 0, 0},*)
(*        {0.3Exp[0.1I],2I,0, 0},*)
(*        {0.1Exp[0.2I],0, I/2,0},*)
(*        {0.5Exp[0.3I],0,0,3I/2}};*)*)
(*(*AMat={{0*I/2,1, 1, 1},*)
(*        {1,0*I/2,1, 1},*)
(*        {1,1, 0*I/2,1},*)
(*        {1,1,1,0*I/2}};*)*)
(*AMat={{1*I/2,1, 1},*)
(*        {1,1*I/2,0},*)
(*        {1,0, 1*I/2}};*)
(*pBounds={-20, 20};*)
(*determiningresult=GetDeterminingEquations[AMat,TotalDegree,NbNodes,pBounds];*)
(*DetEqS=determiningresult[[1]];*)
(*DetEqR=determiningresult[[2]];*)
(*pSub=determiningresult[[3]];*)
(*nbNodes=determiningresult[[4]];*)
(**)
(*result=GetDeterminingMatrices[DetEqS,DetEqR,pSub,nbNodes];*)
(*DSquare = result[[1]];*)
(*NRectangle = result[[2]];*)
(*DetMatrix = result[[3]];*)
(*NonZeroCoeffs :=  result[[4]];*)
(*ReImDetMatrix=Map[ReIm,DetMatrix,{2}];*)
(*(*Export["/Users/benja/Documents/GitHub/koopman-kuramoto/symbolic/eigenfunctions/determining_matrices/ReImDetMatrix_blabla.json",ReImDetMatrix,"JSON"];*)*)
(**)
(*(* Print["Nonzero coefficients in ", NonZeroCoeffs]*)*)
(*Print["# of nonzero coefficients = ", Length[NonZeroCoeffs]]*)
(*Print["square determining matrix M dimension = ", Dimensions[DSquare]]*)
(*Print["rectangular determining matrix N dimension = ", Dimensions[NRectangle]]*)
(* (*Print[DSquare//TraditionalForm] *)*)
(*(* Print[NRectangle//TraditionalForm] *)*)
(*(* Print[DetMatrix//TraditionalForm]*) *)
(**)
(*(*SingVals := SingularValueList[N[DetMatrix]]  (* DetMatrix for symbolic and N[DetMatrix] for numerical *)*)
(*Print[SingVals]*)
(*Print["There is (are) ", Length[NonZeroCoeffs] - Length[SingVals], " zero singular value(s)."]*)*)
(**)
(*(* Print[SingularValueDecomposition[N[DetMatrix]]] *)*)
(**)
(*(*SVD of the rectangular system*)*)
(*singValsRectangleSys = SingularValueList[N[NRectangle]];(**)
(*Print["Singular values of the rectangle matrix:", singValsRectangleSys]*)*)
(*Print["There is (are) ", Min[Dimensions[NRectangle]] - Length[singValsRectangleSys], " zero singular value(s)."]*)
(**)
(*{u, \[Sigma], v} = SingularValueDecomposition[N[NRectangle]];*)
(**)
(*nullSpaceRectangle = NullSpace[NRectangle];*)
(*Print["Dimension of the nullspace of the rectangle matrix: ", Length[nullSpaceRectangle]]*)
(**)


(* Change of basis to the right singular vectors of the rectangle matrix *)
DSquarePrime = Transpose[v] . DSquare . v;
Print[Dimensions[DSquarePrime]]

(* Compute the eigenvectors of the square matrix in this new basis *)
{eigvalsDSquarePrime, eigvectsDSquarePrime} = Eigensystem[DSquarePrime];

(* check for degenerate eigenvalues *)
Print["degenerate eigenvalues: ", Length[eigvalsDSquarePrime] != Length[Union[eigvalsDSquarePrime]]]

bound = 10^(-12);
eigvectsDSquarePrime = Map[If[Abs[#] < bound, 0, #] &, eigvectsDSquarePrime, {2}];

sortedEigenvectors = SortBy[eigvectsDSquarePrime, Abs[#] &];

(* Get the ordering based on the absolute values of eigvectsDSquarePrime *)
ordering = Ordering[Abs[eigvectsDSquarePrime]];

(* Apply the same ordering to eigvectsDSquarePrime and anotherList *)
sortedEigenvectors = eigvectsDSquarePrime[[ordering]];
sortedEigenvalues = eigvalsDSquarePrime[[ordering]];


index = 1
sortedEigenvalues[[index]]
sortedEigenvectors[[index, ;;-Length[nullSpaceRectangle]-1]]
sortedEigenvectors[[1, -Length[nullSpaceRectangle];;]]


(* inverse change of basis on the good eigenvectors *)
index = 1;
goodEigenvectorWrongBasis = sortedEigenvectors[[index]];
(* Consider the truncation exact and the eigenvector approximate: zero-pad the eigenvector *)
(*goodEigenvectorWrongBasis[[1;;-Length[nullSpaceRectangle]-1]] = 0;*)
goodEigenvalue = sortedEigenvalues[[index]];

goodEigenvector = v . goodEigenvectorWrongBasis;


(* Verify that it satisfies approximately the rectangle and square matrix equations*)
rectangleVerif = Norm[NRectangle . goodEigenvector]
squareVerif = Norm[DSquare . goodEigenvector - goodEigenvalue goodEigenvector]



