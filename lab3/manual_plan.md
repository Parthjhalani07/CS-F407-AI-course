# Task 1: Construct a Plan by Hand

Plan: Move(A, B), PickUp(Package, B), Move(B, C), Drop(Package, C)

| State | Facts |
|---|---|
| S0 | At(Robot, A), At(Package, A) |
| S1 = Apply(S0, Move(A,B)) | At(Robot, B), At(Package, A) |
| S2 = Apply(S1, PickUp(Package,B)) | — wait, invalid, see note below |

**Note:** PickUp(Package, B) is *not* applicable in S1, because At(Package, B) does
not hold in S1 (the package is still at A). The robot must bring the package with
it, not itself, to move it. The correct plan first picks the package up at A:

## Corrected plan

Plan: PickUp(Package, A), Move(A, B), Move(B, C), Drop(Package, C)

| State | Facts |
|---|---|
| S0 | At(Robot, A), At(Package, A) |
| S1 = Apply(S0, PickUp(Package,A)) | At(Robot, A), Holding(Package) |
| S2 = Apply(S1, Move(A,B)) | At(Robot, B), Holding(Package) |
| S3 = Apply(S2, Move(B,C)) | At(Robot, C), Holding(Package) |
| S4 = Apply(S3, Drop(Package,C)) | At(Robot, C), At(Package, C) |

S4 ⊨ G, since At(Package, C) ∈ S4. This plan has length 4 and is minimal
(shorter plans cannot reach C from A, since A and C are not directly connected —
the robot must pass through B).
