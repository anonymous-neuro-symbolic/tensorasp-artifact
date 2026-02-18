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

% crowded :-
%    group_scene.

crowded :-
    group_scene,
    strong_collective_presence.


normal_activity :-
    single_person_scene.

% ================================
% Some constraints
% ================================

:- crowded, normal_activity.


% ========================
% Border line rules----> for the ablation test!!
% ========================

uncertain_group :-
    person_detected,
    multiple_people,
    not strong_evidence.

strong_evidence :-
    aggregated_confidence_high.

group_scene :-
    multiple_people,
    strong_evidence.

