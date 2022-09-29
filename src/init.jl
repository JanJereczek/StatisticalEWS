using LinearAlgebra, BenchmarkTools, RollingFunctions
using Statistics, StatsBase, Random, Distributions
using CairoMakie, DifferentialEquations, FFTW

include(srcdir("ews.jl"))
include(srcdir("utils.jl"))
include(srcdir("models.jl"))
include(srcdir("stattest.jl"))