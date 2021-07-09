nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

initiatedAtNoise(X, Y) :- initiatedAt(X, Y).

% Number of timestamps to look at (should be min the length of the sequence)
givenRemaining(10).

initiatedAt(sequence0 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([1, not9, 0], Remaining, T).
initiatedAt(sequence0 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([2, not9, 1], Remaining, T).

initiatedAt(sequence1 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([3, not9, 2], Remaining, T).
initiatedAt(sequence1 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([4, not9, 3], Remaining, T).

initiatedAt(sequence2 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([5, not9, 4], Remaining, T).
initiatedAt(sequence2 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([6, not9, 5], Remaining, T).

initiatedAt(sequence3 = true, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([7, not9, 6], Remaining, T).
initiatedAt(sequence3 = false, T) :-
    givenRemaining(Remaining),
    sequenceEndingAt([8, not9, 7], Remaining, T).

isNegation(not9, 9).

wrapper(X, Y) :- digit(X, Y).

itemIn(X, [X | _]).
itemIn(X, [_ | L]) :- itemIn(X, L).

% An empty sequence will always be within if there are no simple events excluded
sequenceWithin([], [], _, _).

% An empty sequence will also be within if we have used all of the Remaining
sequenceWithin([], _, 0, _).

sequenceWithin([], E, Remaining, T) :-
    Remaining > 0,
    happensAt(H, T),
    wrapper(H, Y),
    \+ itemIn(Y, E),
    NextRemaining is Remaining - 1,
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    sequenceWithin([], E, NextRemaining, Tprev).

% A sequence can be within Remaining of T if it starts at T
sequenceWithin(L, E, Remaining, T) :-
    sequenceEndingAt(L, Remaining, T).

sequenceWithin([X | L], E, Remaining, T) :-
    isNegation(X, Y),
    sequenceWithin(L, [Y | E], Remaining, T).

% A sequence can be within Remaining of T if it is within NextRemaining of Tprev
sequenceWithin([X | L], E, Remaining, T) :-
    Remaining > 0,
    T >= 0,
    \+ isNegation(X, _),
    happensAt(H, T),
    wrapper(H, Y),
    \+ itemIn(Y, E),
    NextRemaining is Remaining - 1,
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    sequenceWithin([X | L], E, NextRemaining, Tprev).

% A sequence will start at T and be within Remaining if X happens at T and the rest of the sequence is within NextRemaining of Tprev
sequenceEndingAt([X | L], Remaining, T) :-
    Remaining > 0,
    T >= 0,
    happensAt(Y, T),
    wrapper(Y, X),
    NextRemaining is Remaining - 1,
    allTimeStamps(Timestamps),
    previousTimeStamp(T, Timestamps, Tprev),
    sequenceWithin(L, [], NextRemaining, Tprev).

sdFluent( aux ).
