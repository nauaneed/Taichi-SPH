
#set text(
  font: "New Computer Modern", size: 12pt,
)

#set par(
  justify: true, spacing: 1.4em
)


#set page(margin: (x: 15mm, y: 15mm))

#align(center)[
  #text(size: 18pt, weight: "bold")[Simulation of Thermoviscous Flow of \ Heated Glass Preform]
]

#align(center)[
  #text(size: 12pt)[Navaneet Villodi]
]
#v(-0.7em)
#align(center)[
  #text(size: 12pt, style: "italic")[Department of Aerospace Engineering, Indian Institute of Technology Bombay]
]



#set heading(numbering: "1.1.")


= Introduction
This short report presents a simulation of a glass preform being fed into a furnace, where it undergoes a temperature-dependent viscosity change and the subsequent creeping flow of the resulting highly viscous molten glass under gravity. The simulation is performed using the Smoothed Particle Hydrodynamics (SPH) method. The objective of this simulation is to demonstrate the ability of the SPH to handle such complex industrial phenomena, including heat transfer, temperature-dependent material properties, and highly viscous flow. As this is intended to be a serve as a demonstrator, various assumptions have been made. So, the results are not intended to be fully accurate of the real-world process, but a proof of concept of the capabilities of the method and how this type of a problem can be handled in SPH parlance. The simulation setup, results, and analysis are presented in the following sections.

= Setup

In this section, the setup of the simulation is described, including the geometry, initial conditions, boundary conditions, and material properties, discretization details. Care has been taken to ensure that the setup captures the essential physics of the problem while remaining computationally tractable. Care has been taken to state all assumptions and simplifications made in the setup, as well as the rationale behind them. 

The setup consists of a glass preform being fed into a furnace vertically from the top. The preform is initially at room temperature. As it enters the furnace, it is heated up to a high temperature, which causes its viscosity to decrease significantly. The molten glass then flows under the influence of gravity, exhibiting a creeping flow behavior. The glass preform is modeled as a rectangular block of dimensions 6 cm x 6 cm cross section. The rectangular geometry is chosen for simplicity. The rectangular geometry still captures the essential physics in terms of how a high-viscosity fluid column behaves under gravity. The shape can be made cylinderical to match with the common geometry of preform when glass drawing simulations are to be performed. 

== Material Properties and Other Parameters

Specific heat capacity, $c_p$, of the glass is assumed constant at 840 J/kg-K, and the thermal conductivity, $k$, is assumed constant at 1.1 W/m-K. The density, $rho$, of the glass is assumed constant at 2500 kg/m^3. The thermal diffusivity, $alpha$, comes out to be $5.24 times 10^(-7) "m"^2"s"^(-1)$. 

Typical preform feed rate is of the order of $1 times 10^(-4)$ to $1 times 10^(-3)$ m/s and the fiber draw speed is of the order of 10 to 30 m/s according to literature @computation12050086 @Mawardi. Since this simulation is about the fiber drawing process at present, a feed rate of 0.5 m/s, lying in between the typical feed rate and the fiber draw speed, is chosen. The preform is assumed to be at 300K and the furnace is assumed to heat the preform to 1100K. A sigmoidal temperature profile is applied in the furnace to model the heating process. This can later by replaced by a Dirichlet (temperature) or Neumann (heat-flux) boundary condition, to include the radiation effects, later. 

With the feed rate of 0.5 m/s, the furnace would be operating at 3.02 MW, without considering losses. This would be physically unrealistic in terms of energy consumption. Moreover sudden heating will also cause thermal shock and other undesirable effects. We sidestep these issues for the sake of simplicity. Specifically, the former is just numbers in the simulation, and the latter is unaccounted by not including the themerature dependent density and thermal expansion.

While glass exhibits non-newtonian behavior at high stress levels @liFlowGlassHigh1970, for simplicity, the viscosity is assumed to be a function of temperature only, and the shear-thinning behavior is neglected. The viscosity is modeled using the Vogel-Fulcher-Tammann (VFT) equation, which is commonly used to describe the temperature dependence of viscosity in glass-forming liquids @parsonsViscosityProfilesPhosphate2015. The temperature-dependent viscosity is modelled using the VFT equation as,

$ log_10(mu) = A + frac(B, T - T_0), $

where $mu$ is the viscosity, $T$ is the temperature, and $A$, $B$, and $T_0$ are material-specific constants. For the glass being simulated, the following values are used: $A = -0.5$, $B = 3000 K$, and $T_0 = 200 K$. 

== Governing Equations
The governing equations for the simulation is the Navier-Stokes equations for incompressible flow, coupled with the heat equation for temperature evolution. The continuity equation reads,
$ frac(dif rho, dif t) = - rho nabla dot bold(v), $
where $dif$ denotes the material derivative and $bold(v)$ is the velocity vector. With the incompressibility assumption, the continuity equation simplifies to,
$ nabla dot bold(v) = 0. $
The momentum equation reads,
$ rho frac(dif bold(v), dif t) = - nabla p + mu(T) nabla bold(v)^2 + g, $
where $p$ is the pressure field, $mu$ is the dynamic viscosity, and $g$ is the gravitational acceleration vector.
The heat conduction equation reads,
$ frac(dif T, dif t) = alpha nabla^2 T, $
where $alpha$ is the thermal diffusivity introduced earlier. Note hat we are working with material derivatives here; we can avoid explicitly accounting for the convective term since we are working in a Lagrangian framework. The viscous heating term, which accounts for the conversion of mechanical energy into thermal energy due to viscous dissipation is neglected as the heat generated by viscous dissipation is expected to be negligible compared to the heat input from the furnace. 


== Discretisation
The Smoothed Particle Hydrodynamics (SPH) method is used to discretize and solve these equations. The SPH method is a mesh-free Lagrangian method that represents the fluid as a collection of particles, each carrying properties such as mass, velocity, temperature, and viscosity. The interactions between particles are computed using a smoothing kernel function, which allows for the approximation of spatial derivatives and the enforcement of boundary conditions. The Divergence-Free SPH (DFSPH) @benderDivergenceFreeSmoothed2015 @huangJourneySPHSimulation2024. 

The timestep in viscosity dominated flows is majorly restricted by the viscous timestep criterion, given by $Delta t_("visc") < (rho h^2 slash mu)$, where $h$ is the smoothing length. As the viscosity increases, the viscous timestep decreases significantly, leading to very small time steps and long simulation times. An implicit viscosity formulation @weilerPhysicallyConsistentImplicit2018 is employed for the viscous term in the momentum equation, allowing for realisable time steps while maintaining stability. A symplectic semi-implicit euler integrator is employed for postition update and explicit euler for temperature update. 

A detailed exposition on the spatial and temporal discretisation is not included in this report for ths sake of brevity.

= Results
The following results are obtained for a simulation of 15 seconds of physical time, with timestep of 0.002 seconds and a particle spacing of 1 cm.



#bibliography("ref.bib", style: "elsevier-vancouver")

