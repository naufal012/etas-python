import numpy as np
from scipy.integrate import quad
from scipy.special import beta

def h_z_zprime(z, z_prime, Z, eta):
    u = z / Z
    v = z_prime / Z
    
    # beta function denominator
    b_val = beta(eta * v + 1.0, eta * (1.0 - v) + 1.0)
    
    val = (u ** (eta * v)) * ((1.0 - u) ** (eta * (1.0 - v)))
    return val / (Z * b_val)

Z_max = 30.0
eta_val = 5.0

# Let's test for different parent depths (z_prime)
z_primes = [5.0, 15.0, 25.0]

print("Numerical Integration of h(z; z') from 0 to Z_max:")
for zp in z_primes:
    res, err = quad(h_z_zprime, 0, Z_max, args=(zp, Z_max, eta_val))
    print(f"Parent Depth z' = {zp} km -> Integral = {res:.6f} (Error: {err:.2e})")
