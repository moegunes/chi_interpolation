Without kappa, we had regular Pi(q). Its moments are produced by the following script:

============================================================================
ClearAll[q, kF, chi0, vc, fxcFun, fxcSeries, chi, chiSeries, chiHat, \
Cn]

(*----\[Chi]0(q) small-q (use enough terms;q^6 needed for C3)----*)
chi0[q_] := (-kF/(Pi^2) + (1/(6 Pi^2 kF)) q^2/
      2! + (1/(10 Pi^2 kF^3)) q^4/4! + (5/(42 Pi^2 kF^5)) q^6/6!);

(*Coulomb kernel*)
vc[q_] := 4 Pi/q^2;

(*Put your actual fxc(q) here (function of q).*)
fxcFun[q_] := f0 + f2 q^2 + f4 q^4 + f6 q^6;

(*Series-expand fxc around q=0 to the order needed*)
fxcSeries = Normal@Series[fxcFun[q], {q, 0, 6}] // FullSimplify;

(*Dyson*)
chi[q_] := chi0[q]/(1 - (fxcSeries) chi0[q]);

(*Laurent/Taylor series of \[Chi](q) around q=0 (prevents 1/0)*)
chiSeries = Normal@Series[chi[q], {q, 0, 6}] // FullSimplify;

(*\[Chi]^(n)(0) in YOUR notation=ordinary (2n)-th derivative at 0*)
chiHat[n_Integer?NonNegative] := 
  FullSimplify[(2 n)!*SeriesCoefficient[chiSeries, {q, 0, 2 n}]];

(*Paper moment mapping*)
Cn[n_Integer?NonNegative] := 
  FullSimplify[(-1)^n*(2 n + 1)/(4 Pi)*chiHat[n]];

C0 = Cn[0];
C1 = Cn[1];
C2 = Cn[2];
C3 = Cn[3];

{C0, C1, C2, C3}

output:

{-(kF/(4 (f0 kF \[Pi] + \[Pi]^3))), -((12 f2 kF^3 + \[Pi]^2)/(
  8 kF \[Pi] (f0 kF + \[Pi]^2)^2)), (-720 (f2^2 - f0 f4) kF^6 + 
  8 kF (f0 - 15 f2 kF^2 + 90 f4 kF^4) \[Pi]^2 + 3 \[Pi]^4)/(
 24 kF^3 \[Pi] (f0 kF + \[Pi]^2)^3), -1/(
   48 kF^5 \[Pi] (f0 kF + \[Pi]^2)^4) (60480 (f2^3 - 2 f0 f2 f4 + 
        f0^2 f6) kF^9 + 
     3 kF^2 (29 f0^2 + 5040 f2 kF^4 (f2 - 8 f4 kF^2) - 
        224 f0 (2 f2 kF^2 + 15 kF^4 (f4 - 12 f6 kF^2))) \[Pi]^2 + 
     2 kF (31 f0 - 
        42 kF^2 (f2 + 120 kF^2 (f4 - 6 f6 kF^2))) \[Pi]^4 + 
     10 \[Pi]^6)}


========================================================================================================================================================

with introduction of kappa, the script is modified now:

=============================================================

ClearAll[q, kF, chi0, vc, fxcFun, fxcSeries, chi, chiSeries, chiHat, \
Cn, \[Kappa]]

(*----\[Chi]0(q) small-q (use enough terms;q^6 needed for C3)----*)
chi0[q_] := (-kF/(Pi^2) + (1/(6 Pi^2 kF)) q^2/
      2! + (1/(10 Pi^2 kF^3)) q^4/4! + (5/(42 Pi^2 kF^5)) q^6/6!);

(*Coulomb kernel*)
vc[q_] := 4 Pi/(q^2 + \[Kappa]^2 );

(*Put your actual fxc(q) here (function of q).*)
fxcFun[q_] := f0 + f2 q^2 + f4 q^4 + f6 q^6;

(*Series-expand fxc around q=0 to the order needed*)
fxcSeries = Normal@Series[fxcFun[q], {q, 0, 6}] // FullSimplify;

(*Dyson*)
chi[q_] := chi0[q]/(1 - (fxcSeries + vc[q]) chi0[q]);

(*Laurent/Taylor series of \[Chi](q) around q=0 (prevents 1/0)*)
chiSeries = Normal@Series[chi[q], {q, 0, 6}] // FullSimplify;

(*\[Chi]^(n)(0) in YOUR notation=ordinary (2n)-th derivative at 0*)
chiHat[n_Integer?NonNegative] := 
  FullSimplify[(2 n)!*SeriesCoefficient[chiSeries, {q, 0, 2 n}]];

(*Paper moment mapping*)
Cn[n_Integer?NonNegative] := 
  FullSimplify[(-1)^n*(2 n + 1)/(4 Pi)*chiHat[n]];

C0 = Cn[0];
C1 = Cn[1];
C2 = Cn[2];
C3 = Cn[3];

{C0, C1, C2, C3}

output:

{-((kF \[Kappa]^2)/(
  4 \[Pi] (4 kF \[Pi] + (f0 kF + \[Pi]^2) \[Kappa]^2))), -((\[Pi]^2\
 \[Kappa]^4 + 12 kF^3 (-4 \[Pi] + f2 \[Kappa]^4))/(
  8 kF \[Pi] (4 kF \[Pi] + (f0 kF + \[Pi]^2) \[Kappa]^2)^2)), \
(2880 kF^5 \[Pi] (f0 kF + \[Pi]^2) + 
    480 kF^3 \[Pi] (12 f2 kF^3 + \[Pi]^2) \[Kappa]^2 + 
    32 kF \[Pi] (90 f4 kF^5 + \[Pi]^2) \[Kappa]^4 + (-720 (f2^2 - 
          f0 f4) kF^6 + 8 kF (f0 - 15 f2 kF^2 + 90 f4 kF^4) \[Pi]^2 + 
       3 \[Pi]^4) \[Kappa]^6)/(24 kF^3 \[Pi] (4 kF \[Pi] + (f0 kF + \
\[Pi]^2) \[Kappa]^2)^3), -((80640 kF^6 \[Pi] (-3 f0^2 kF^3 + 
         12 f2 kF^3 \[Pi] - 6 f0 kF^2 \[Pi]^2 + \[Pi]^3 - 
         3 kF \[Pi]^4) - 
      2688 kF^4 \[Pi] (180 f0 f2 kF^5 - 720 f4 kF^5 \[Pi] + 
         15 kF^2 (f0 + 12 f2 kF^2) \[Pi]^2 - 8 \[Pi]^3 + 
         15 kF \[Pi]^4) \[Kappa]^2 + 
      48 kF^2 \[Pi] (5040 (-3 f2^2 + 2 f0 f4) kF^7 + 
         20160 f6 kF^7 \[Pi] + 
         56 kF^2 (2 f0 - 45 kF^2 (f2 - 4 f4 kF^2)) \[Pi]^2 + 
         29 \[Pi]^3 + 7 kF \[Pi]^4) \[Kappa]^4 + 
      8 kF \[Pi] (60480 (-f2 f4 + f0 f6) kF^8 + 
         3 kF (29 f0 - 
            112 (2 f2 kF^2 + 15 kF^4 (f4 - 12 f6 kF^2))) \[Pi]^2 + 
         31 \[Pi]^4) \[Kappa]^6 + (60480 (f2^3 - 2 f0 f2 f4 + 
            f0^2 f6) kF^9 + 
         3 kF^2 (29 f0^2 + 5040 f2 kF^4 (f2 - 8 f4 kF^2) - 
            224 f0 (2 f2 kF^2 + 15 kF^4 (f4 - 12 f6 kF^2))) \[Pi]^2 + 
         2 kF (31 f0 - 
            42 kF^2 (f2 + 120 kF^2 (f4 - 6 f6 kF^2))) \[Pi]^4 + 
         10 \[Pi]^6) \[Kappa]^8)/(48 kF^5 \[Pi] (4 kF \[Pi] + (f0 kF \
+ \[Pi]^2) \[Kappa]^2)^4))}

=============================================================

