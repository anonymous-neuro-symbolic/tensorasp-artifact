% perception-derived facts
person(P).
holding_ball(P).
standing(P).
running(P).

% activity inference
playing(P) :-
    person(P),
    holding_ball(P),
    running(P).

idle(P) :-
    person(P),
    standing(P),
    not holding_ball(P).

% constraints
:- playing(P), idle(P).
