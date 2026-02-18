#program base.
#external multiple_people_detected.
#external single_person_detected.

% Inconsistent scene description:
% A scene cannot simultaneously be single-person and multi-person.
inconsistent_scene :-
    multiple_people_detected,
    single_person_detected.

:- inconsistent_scene.

% Deterministic scene interpretation
group_scene :-
    multiple_people_detected,
    not single_person_detected.

single_person_scene :-
    single_person_detected,
    not multiple_people_detected.

#show group_scene/0.
#show single_person_scene/0.
