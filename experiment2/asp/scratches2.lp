% --- Scratch reasoning rules ---

single_scratch(S) :-
    longitudinal_patterns_in_bottom(S, yes).

separate_scratches(S1,S2) :-
    untouched_surface_visible_between(S1,S2, yes),
    S1 != S2.

% --- Constraints ---
:- single_scratch(S), separate_scratches(S,_).
