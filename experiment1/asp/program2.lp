% ---------- Facts provided by Python ----------
% in_place(c).
% rel_object(c).
% eng_rel(c).
% full_dedic(c).
% shift_dedic(c).
% low_time(c).
% high_time(c).

%%%%%%Involvement rules
non_involv(c) :-
    in_place(c),
    low_time(c),
    not full_dedic(c),
    not rel_object(c),
    not eng_rel(c).

part_involv(c) :-
    in_place(c),
    low_time(c),
    shift_dedic(c),
    not rel_object(c),
    not eng_rel(c).

full_involv(c) :-
    in_place(c),
    not low_time(c),
    full_dedic(c),
    rel_object(c),
    eng_rel(c).

hook_involv(c) :-
    in_place(c),
    high_time(c),
    full_dedic(c),
    rel_object(c),
    eng_rel(c).

%%%%%%% Mutual exclusivity
:- full_involv(c), non_involv(c).
:- part_involv(c), non_involv(c).
:- hook_involv(c), non_involv(c).
