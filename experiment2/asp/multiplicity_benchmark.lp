#program base.
#external signal1.
#external signal2.

conflict :- signal1, signal2.
:- conflict.

choice1 :- signal1, not signal2.
choice2 :- signal2, not signal1.

#show choice1/0.
#show choice2/0.
