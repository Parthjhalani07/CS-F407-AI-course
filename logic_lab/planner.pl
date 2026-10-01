% Task 6: Prolog as a Plan Verifier
% Task 7: Using Prolog to Check a Proposed Plan
% Task 8: Connect Prolog to Logical Reasoning

% --- Task 6: warehouse connectivity facts ---
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :-
    connected(X,Y).

% --- Task 7: checking moves proposed by the Python planner ---
valid_move(X,Y) :-
    connected(X,Y).

% --- Task 8: wet road toy example ---
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
