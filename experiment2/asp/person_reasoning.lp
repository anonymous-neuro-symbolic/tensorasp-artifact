% ================================
% Input facts (from projections)
% ================================
% person_detected.
% multiple_people.

% ================================
% Scene interpretation
% ================================

single_person_scene :-
    person_detected,
    not multiple_people.

group_scene :-
    person_detected,
    multiple_people.

empty_scene :-
    not person_detected.

% ================================
% Higher-level semantic labels
% ================================

crowded :-
    group_scene.

normal_activity :-
    single_person_scene.

% ================================
% Some constraints
% ================================

:- crowded, normal_activity.
