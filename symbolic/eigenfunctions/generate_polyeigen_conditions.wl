(* ::Package:: *)

(*STEP 1: FIND ALL RELEVANT COMBINATIONS*)


q1 = 1;
q2 = 2;
q3 = 3;
qList = {{q1, q2, q3}, {q1+1, q2-1, q3}};
n = 3;
(*find all (+,-) index tuples for N-dim*)
allModifs[n_] := (tuples = Tuples[Range[n], 2];
				  tuples = DeleteElements[tuples, {{1,1},{2,2},{3,3}}])
modifs = allModifs[n];


(*compute all relevant combinations*)
computeCombinations[qList_, modifs_] := 
    (pList = qList;
	Do[Do[newP = q;
	      newP[[modif[[1]]]] = newP[[modif[[1]]]]+1;
	      newP[[modif[[2]]]] = newP[[modif[[2]]]]-1;
	      AppendTo[pList, newP], {q, qList}], {modif, modifs}];
	      pList = DeleteDuplicates[pList];
	      Return[pList])
pList = computeCombinations[qList, modifs];


(*order all obtained combinations*)
qList = Sort[qList];
qToIndex = <||>;
Do[AppendTo[qToIndex, qList[[i]]->i], {i, Length[qList]}]
qToIndex


(*build the equation for a given combination of type p or q*)
AMat = Array[A, {n, n}];
cVec = Array[c, Length[qList]];
\[Omega]Vec = Array[\[Omega], n];

buildPEquation[p_, qList_] := 
	(lhs = 0;
	goodDiff = Join[{-1}, ConstantArray[0, Length[p]-2], {1}];
	Do[diff = q - p;
		If[Sort[diff]==goodDiff,
			j = Position[diff, 1][[1]][[1]];
			k = Position[diff, -1][[1]][[1]];
			lhs = lhs + cVec[[qToIndex[q]]](AMat[[j,k]]q[[j]] - Conjugate[AMat[[k,j]]]q[[k]]),
			Continue[]],
	 {q, qList}];
	 Return[lhs == 0])

buildQEquation[qi_, qList_] := 
	(lhs = -I cVec[[qToIndex[qi]]] \[Omega]Vec . qi;
	goodDiff = Join[{-1}, ConstantArray[0, Length[qi]-2], {1}];
	Do[diff = q - qi;
		If[Sort[diff]==goodDiff,
			j = Position[diff, 1][[1]][[1]];
			k = Position[diff, -1][[1]][[1]];
			lhs = lhs + cVec[[qToIndex[q]]](AMat[[j,k]]q[[j]] - Conjugate[AMat[[k,j]]]q[[k]]),
			Continue[]],
	 {q, qList}];
	 Return[lhs == \[Lambda] cVec[[qToIndex[qi]]]])

buildQEquation[qList[[1]], qList] // TraditionalForm
buildQEquation[qList[[2]], qList] // TraditionalForm


(*build the system of equations*)
sysEqs = {};
Do[AppendTo[sysEqs, buildQEquation[qi, qList]], {qi, qList}]
Do[AppendTo[sysEqs, buildPEquation[pi, qList]], {pi, pList}]

Do[Print[TraditionalForm[eq]], {eq, sysEqs}]


(*Reduce[sysEqs, Flatten[AMat]]*)
