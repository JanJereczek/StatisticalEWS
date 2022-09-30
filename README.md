# StatisticalEWS

## Getting started

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> StatisticalEWS

It is authored by Swierczek-Jereczek.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

## Aim

This repository is a prototype for a future package. It aims to provide the basic tools for the computation of statistical early warning signals (EWSs). These should allow the recognition of a state approaching a critical transition by analysis of the associated time series.

Examples are provided in the *scripts* folder.