using DrWatson
@quickactivate "StatisticalEWS"

using LinearAlgebra, BenchmarkTools, RollingFunctions
using Statistics, StatsBase, Random, Distributions
using CairoMakie, DifferentialEquations, FFTW
using SparseArrays, CUDA

include(srcdir("utils.jl"))
include(srcdir("transition_indicators.jl"))
include(srcdir("models.jl"))
include(srcdir("significance_test.jl"))