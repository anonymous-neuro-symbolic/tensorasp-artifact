% involvement rules with NAF and strong negation
non_involv :- in_place, low_time, -full_dedic, not rel_object, not eng_rel.
part_involv :- in_place, low_time, shift_dedic, not rel_object, not eng_rel.
full_involv :- in_place, rgt_time, full_dedic, rel_object, eng_rel.
hook_involv :- in_place, high_time, hook_dedic, rel_object, eng_rel.
overl_involv :- in_place, high_time, toom_dedic, rel_object, eng_rel.
