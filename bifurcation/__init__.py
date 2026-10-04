"""Coupled networks for a 1 -> 2 channel bifurcation, built on developing_flow.

Network 1  developing_flow.pinn: the developing, inertial flow in the parent
           channel. It supplies the inlet profile and a warm start.
Network 2  the pressure-drop PINN on the bifurcation (to come).
Network 3  the geometric network: from the daughters' angles it proposes the
           width split with the lowest pressure drop (to come).

`geometry` builds the bifurcation the networks share.
"""
