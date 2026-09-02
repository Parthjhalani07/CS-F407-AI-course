% planner.pl  -  Prolog logical verifier for the warehouse planning domain
% Task 6: Prolog as a Plan Verifier
% Task 7: Using Prolog to check a proposed plan
% Task 8: Connect Prolog to logical reasoning

% -----------------------------------------------------------------------
% Task 6 - Warehouse connectivity facts
% -----------------------------------------------------------------------

connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Rule: the robot can move between connected locations
can_move(X, Y) :-
    connected(X, Y).

% Task 6 queries:
%   ?- can_move(a,b).   % true  - a and b are connected
%   ?- can_move(a,c).   % false - a and c are not directly connected

% -----------------------------------------------------------------------
% Task 7 - Validate individual moves from a proposed plan
% -----------------------------------------------------------------------

% A move is valid iff the locations are connected
valid_move(X, Y) :-
    connected(X, Y).

% Task 7 queries:
%   ?- valid_move(a,b).   % true
%   ?- valid_move(b,c).   % true
%   ?- valid_move(a,c).   % false  (no direct link a-c)

% Challenge: Move(a,c) is NOT supported by the warehouse knowledge base.
% Prolog returns false for valid_move(a,c) because connected(a,c) has
% no matching fact.

% -----------------------------------------------------------------------
% Task 8 - Wet-road logical reasoning example
% -----------------------------------------------------------------------

wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% Query:  ?- reduce_speed.
%
% Reasoning chain (Fact => Rule => Rule => Conclusion):
%   wet_road  =>  slippery  =>  reduce_speed
%
% Prolog proves reduce_speed by backward chaining:
%   1. reduce_speed requires slippery.
%   2. slippery requires wet_road.
%   3. wet_road is a base fact  =>  true.
%   Therefore reduce_speed is true.

% -----------------------------------------------------------------------
% Task 7 - Verify the full Python-generated plan step by step
%
% The Python planner produced:
%   PickUp(Package, a)
%   Move(a, b)
%   Move(b, c)
%   Drop(Package, c)
%
% We verify only the movement steps here (PickUp/Drop are not
% movement-connectivity checks).
% -----------------------------------------------------------------------

valid_plan_moves :-
    valid_move(a, b),   % Move(a,b) is supported
    valid_move(b, c).   % Move(b,c) is supported

% Query:  ?- valid_plan_moves.   % true
