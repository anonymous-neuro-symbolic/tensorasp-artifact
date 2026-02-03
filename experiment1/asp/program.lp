% Involvement rules (with NAF)

non_involv :- in_place, low_time, not rel_object.
part_involv :- in_place, low_time, rel_object, not full_dedic.
full_involv :- in_place, rgt_time, full_dedic, rel_object.
hook_involv :- in_place, high_time, full_dedic, rel_object.
overl_involv :- in_place, high_time, over_dedic, rel_object.

% exclusivity
:- part_involv, non_involv.
:- full_involv, part_involv.
:- hook_involv, full_involv.
:- overl_involv, hook_involv.
