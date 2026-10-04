"""Coupled networks for a 1 -> 2 channel bifurcation, built on developing_flow.

Network 1  developing_flow.pinn: the developing, inertial flow in the parent
           channel. It supplies the inlet profile (and, to come, a warm start).
Network 2  `flow`: the pressure-drop PINN on one bifurcation, scored against
           the finite-element solution of the same problem. Stage 3b (to come)
           makes the design parameters inputs.
Network 3  the geometric network: from the daughters' angles it proposes the
           width split with the lowest pressure drop (to come).

`geometry` builds the bifurcation the networks share (two angles, any
orientation, bifurcation angles up to and including 180 degrees);
`reference` is the finite-element judge, with either equal outlet pressures
or a prescribed share of the flow through each daughter.
"""
