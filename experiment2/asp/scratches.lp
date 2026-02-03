longitudinal(S) :- scratch(S), orientation(S,theta), theta < 15.
longitudinal(S) :- scratch(S), orientation(S,theta), theta > 165.

continuous(S) :- scratch(S), gap_length(S,G), G < 5.

separated(S1,S2) :-
    scratch(S1), scratch(S2),
    distance(S1,S2,D), D > 20.

% High-level interpretations 
single_scratch(S) :-
    longitudinal(S),
    continuous(S).

separate_scratches(S1,S2) :-
    separated(S1,S2).

%  Consistency rules
:- single_scratch(S), separate_scratches(S,_).
