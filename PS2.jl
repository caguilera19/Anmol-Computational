# Trying

using Optim
using LinearAlgebra
using Plots
using ForwardDiff
using NLsolve

# 1) Write a code that derives the log-linearized solution to the prototype
#growth model with wedges in state-space form with measurement error
# First, we need to find the steady state of the model

θ, ψ, τxss, τlss, β̄, δ, zss, gn, gz, gss, σ = .33, 2.5, .2, .3, .975, .05, 1, 1.01, 1.019, .5, 1

β = β̄*gn

function v!(F,x)
    # x[1]=k , x[2]= c , x[3]=l
    F[1] = x[1] - (((1+τxss)*(1-β*(1-δ))/(β*θ*(zss^(1-θ))))^(1/(θ-1))) *x[3]
    F[2] = x[2] - ((x[1]/x[3])^(θ-1) * (zss^(1-θ)) - (gz)*(gn) + (1-δ))*x[1]+gss
    F[3] = x[2] - ((1-τlss)*(1-θ)*(x[1]/x[3])^θ * zss^(1-θ)/ψ)*(1-x[3])
end

intial_x = [4, .56 , .3]
SS=nlsolve(v!,intial_x,autodiff = :forward)
SS.zero

kss = SS.zero[1]
css = SS.zero[2]
lss = SS.zero[3]


# Now I proceed to log-linearize
# First, following the appendix I first loglinearize the resource constraint

function R(x)
    k1, k2, l, g, z = x
    m = log(((kss^θ * exp(θ*k1))*((lss)^(1-θ) * exp((1-θ)*l))*((zss)^(1-θ) * exp((1-θ)*z)) - (gz*gn)*kss*exp(k2)
    + (1-δ)*(kss*exp(k1)) - gss*exp(g))/css)
end

# Checking that at SS the value of the function is zero:
R([0,0,0,0,0])
# This are the coefficients for C as a function of the other variables:
grad_R= ForwardDiff.gradient(R,[0,0,0,0,0])

# Checking this respect analytical results at page 8
-gz*gn*kss/css   # This is the coefficient for k'
-gss/css # This is the coefficient for g


# For the labor input I am going to work with the eqution  A.2.2. Here, I will
# solve it as an implicit function.


function Labor(x)
    k1, k2, z, l, g, τ = x
    m = (ψ*(((kss*exp(k1))^θ)*(((zss*exp(z))*(lss*exp(l)))^(1-θ))-(gss*exp(g))-(gz*gn*(kss*exp(k2)))+(1-δ)*(kss*exp(k1)))/(1-lss*exp(l))) - (1-(τlss*exp(τ)))*(1-θ)* (kss*exp(k1))^θ *(zss*exp(z))^(1-θ)*(lss*exp(l))^(-θ)
end


Labor([0,0,0,0,0,0])

grad_L= ForwardDiff.gradient(Labor,[0,0,0,0,0,0])
# Since we have an implicit function of L, we need to divide all the coefficients
# by the coefficient for L and mulltiply by -1 to get L as a function of the states:
coeff_L = -grad_L/ grad_L[4]
# Eliminating the L coefficient (= -1)
Φl =filter(e->e≠coeff_L[4],coeff_L)



# Just to check if I did not do something wrong, the coefficient (in negative) for L (page 8):
a=-((ψ*(((kss^θ)*(((zss)*(lss))^(1-θ)))*(1-θ))+((((kss*exp(0))^θ)*(((zss*exp(0))*(lss*exp(0)))^(1-θ))))*(1-θ)*(1-τlss))+a2*(1-τlss)*θ)
# Checking the coefficient for g :
ϕ_g = -ψ*gss/a
# Which is very close

# Now I will proceed to estimate Output

# Output

yss = kss^(θ)*(zss*lss)^(1-θ)
function Y(x)
    k,z,l = x
    m = log((kss^(θ)*exp(θ*k)*(zss*lss)^(1-θ)*exp((1-θ)*z)*exp((1-θ)*l))/yss)
end

Y([0,0,0])
coeff_Y = ForwardDiff.gradient(Y, [0,0,0])

# Getting the coefficients for L as a function of  k1, k2, z, g, τ
Φ_yl = coeff_Y[3]*Φl
Φ_yk1 = coeff_Y[1]+ Φ_yl[1]
Φ_yk2 = Φ_yl[2]
Φ_yz = Φ_yl[3] + coeff_Y[2]
Φ_yg = Φ_yl[4]
Φ_yτ = Φ_yl[5]

# For Investment, I use law motion of capital:
xss = kss*(gn*gz) - ((1-δ)*kss)
function Inv(x)
    k1, k2 = x
    m = log(((kss*exp(k2))*(gn*gz) - ((1-δ)*kss*exp(k1)))/xss)
end

Inv([0,0])

coeff_I = ForwardDiff.gradient(Inv,[0,0])

# Now I proceed with the Euler Equation
