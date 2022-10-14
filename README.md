# StatisticalEWS

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

This repository is a prototype for the future package [TransitionIndicators.jl](https://github.com/JuliaDynamics/TransitionIdentifiers.jl). It aims to provide the basic tools for the computation of statistical transition indicators (TIs), also largely called early warning signals (EWSs). These should allow the recognition of a state approaching a critical transition by analysis of the associated time series.
## Getting started

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to make a reproducible scientific project named
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
## Algorithm diversity and agnosticity

The computation of significant TIs relies on several steps:
1. Detrend the time series to obtain the residuals
2. Compute surrogates of the residuals.
3. Compute TIs for both original data, as well as surrogates.
4. Compute the trend of the TIs for both original data, as well as surrogates.
5. Test the significance of the trend of the original data against the surrogates

For each of these steps, several metrics/algorithms are available. The present prototype provides functions that are agnostic to eachother. This means that using algorithms 1.A. and 2.B. or 1.C. and 2.A. will result in the same syntax (apart from the algorithm specification of course).

## Algorithm diversity and agnosticity

Some of these computations can be very expensive in the context of big data. Therefore, the most expensive functions are implemented as compatible with a GPU parallelisation that tends to give a speedup of factor [10, 100].

An example of computation is provided in the *notebooks* folder.