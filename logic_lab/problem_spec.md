# Task 0: Understand the Planning Problem

## (a) Initial state I

I = { At(Robot, A), At(Package, A) }

## (b) Goal G

G = { At(Package, C) }

## (c) Available actions

- Move(A, B), Move(B, A)
- Move(B, C), Move(C, B)
- PickUp(Package, A), PickUp(Package, B), PickUp(Package, C)
- Drop(Package, A), Drop(Package, B), Drop(Package, C)

(PickUp/Drop are only ever used at the location the robot/package actually occupy
during the plan, but the action *schema* is defined generically over any location.)

## (d) Preconditions and effects

| Action | Preconditions | Effects |
|---|---|---|
| Move(X, Y) | At(Robot, X) | ¬At(Robot, X), At(Robot, Y) |
| PickUp(Package, X) | At(Robot, X), At(Package, X) | ¬At(Package, X), Holding(Package) |
| Drop(Package, X) | At(Robot, X), Holding(Package) | ¬Holding(Package), At(Package, X) |

Move is only defined between connected locations: A–B and B–C (not A–C directly).

## Which actions are initially applicable?

Starting from I = { At(Robot, A), At(Package, A) }:

- **PickUp(Package, A)** — preconditions are At(Robot, A) and At(Package, A).
  Both are members of I, so I ⊨ Preconditions(PickUp(Package, A)). **Applicable.**

- **Drop(Package, C)** — preconditions are At(Robot, C) and Holding(Package).
  Neither holds in I (the robot is at A, not C, and the package is not being held).
  I ⊭ Preconditions(Drop(Package, C)). **Not applicable.**

- Also applicable initially: **Move(A, B)**, since At(Robot, A) ∈ I.
  Move(B, A), Move(B, C), Move(C, B) are *not* applicable (their precondition
  At(Robot, B) or At(Robot, C) does not hold in I).
