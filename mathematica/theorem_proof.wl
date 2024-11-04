(* ::Package:: *)

(*define the quantities*)
$Assumptions = z \[Element] Vectors[N, Complexes]
$Assumptions = A \[Element] Matrices[{N, N}, Complexes]

crossRatio[a_,b_,c_,d_] = ((Indexed[z, c] - Indexed[z, a])(Indexed[z, d] - Indexed[z, b])) / 
				((Indexed[z, c] - Indexed[z, b])(Indexed[z, d] - Indexed[z, a]))


(*define the action of the KOG on the cross-ratio*)
zCrossRatio = {Indexed[z, a],
			   Indexed[z, b],
			   Indexed[z, c],
			   Indexed[z, d]}
crossRatioDerivatives = {};
Do[deriv = D[crossRatio[a,b,c,d], i]; AppendTo[crossRatioDerivatives, {deriv}],
   {i, zCrossRatio}]
crossRatioDerivatives
kogFirstTerm = Transpose[A . z] . crossRatioDerivatives


(*let's skip ahead*)
A[i_,j_] = Indexed[A, {i,j}]
z[i_] = Indexed[z, i]
kogTerm1 = z[k]Sum[(A[c,k] - A[a, k])(z[d] - z[b])(z[c] - z[b])(z[d] - z[a]) +
			   (A[d,k] - A[b, k])(z[c] - z[a])(z[c] - z[b])(z[d] - z[a]) -
			    (A[c,k] - A[b, k])(z[c] - z[a])(z[d] - z[b])(z[d] - z[a]) -
			    (A[d,k] - A[a, k])(z[c] - z[a])(z[d] - z[b])(z[c] - z[b]), {k, 1, N}]
kogTerm2 = z[k]\[Conjugate]Sum[] (*to continue*)



