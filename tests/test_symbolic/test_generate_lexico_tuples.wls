#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Import the function from the external file *)
Get["C:\\Users\\thivi\\Documents\\GitHub\\koopman-kuramoto\\symbolic\\symmetries\\generate_lexico_tuples.wls"];

(* Define individual tests *)

(* Test 1: Simple case d = 0, N = 2 *)
test1 = VerificationTest[generateLexicoTuples[0, 2, {-1, 1}], {{-1, 1}, {0, 0}, {1, -1}}];

(* Test 2: d = 1, N = 3 *)
test2 = VerificationTest[generateLexicoTuples[1, 3, {0, 2}], {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}}];

(* Test 3: Edge case d = 0, N = 1 *)
test3 = VerificationTest[generateLexicoTuples[0, 1, {0, 0}], {{0}}];

(* Test 4: Larger N and d *)
test4 = VerificationTest[generateLexicoTuples[3, 4, {-1, 1}], {{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, 0}}];

(* Test 5: With negative values *)
test5 = VerificationTest[generateLexicoTuples[-2, 2, {-2, 1}], {{-2, 0}, {-1, -1}, {0, -2}}];

(* Test 6: GenerateCoefficients, N = 2, total degree 2 and bounds -3 to 5 *)
test6 = VerificationTest[generateCoefficients[2, generateLexicoTuples[2, 2, {-3, 5}]], 
{c[1, {-3, 5}], c[2, {-3, 5}], c[1, {-2, 4}],  c[2, {-2, 4}], c[1, {-1, 3}], c[2, {-1, 3}],
 c[1, {0, 2}], c[2, {0, 2}], c[1, {1, 1}], c[2, {1, 1}], c[1, {2, 0}], c[2, {2, 0}],
  c[1, {3, -1}], c[2, {3, -1}], c[1, {4, -2}], c[2, {4, -2}], c[1, {5, -3}], c[2, {5, -3}]}];

(* Collect all tests into a test suite *)
tests = {test1, test2, test3, test4, test5};

(* Run the tests and print the report *)
TestReport[tests]
