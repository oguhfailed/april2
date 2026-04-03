"""
Reverse (Inverse) of a Polynomial Function
============================================
Given a polynomial f(x), the "reverse" finds x such that f(x) = y.
Since polynomials of degree > 1 are generally not invertible over their
full domain, we find the numerical inverse over a specified range.
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt


def make_polynomial(coeffs):
    """
    Build a polynomial function from coefficients.
    coeffs: list from highest to lowest degree, e.g. [1, -3, 2] → x² - 3x + 2
    """
    return np.poly1d(coeffs)


def reverse_polynomial(poly, y_value, x_min, x_max, num_segments=1000):
    """
    Find all x in [x_min, x_max] such that poly(x) == y_value.

    Strategy:
      1. Sample the polynomial on a fine grid.
      2. Look for sign changes in (poly(x) - y_value) between adjacent samples.
      3. Use Brent's method to nail each root precisely.

    Returns a list of x values (roots).
    """
    f = lambda x: poly(x) - y_value
    xs = np.linspace(x_min, x_max, num_segments)
    ys = f(xs)

    roots = []
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:          # sign change → a root is bracketed
            root = brentq(f, xs[i], xs[i + 1], xtol=1e-10)
            roots.append(round(root, 10))
        elif abs(ys[i]) < 1e-12:            # landed exactly on the root
            roots.append(round(xs[i], 10))

    # Remove near-duplicates
    roots = sorted(set(roots))
    unique = [roots[0]] if roots else []
    for r in roots[1:]:
        if abs(r - unique[-1]) > 1e-8:
            unique.append(r)
    return unique


def plot_polynomial_and_inverse(poly, x_min, x_max, y_query=None):
    """Plot f(x) and, optionally, mark the inverse for a queried y value."""
    xs = np.linspace(x_min, x_max, 500)
    ys = poly(xs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "b-", linewidth=2, label=f"f(x) = {poly}")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axvline(0, color="k", linewidth=0.8)

    if y_query is not None:
        roots = reverse_polynomial(poly, y_query, x_min, x_max)
        ax.axhline(y_query, color="r", linestyle="--", label=f"y = {y_query}")
        for r in roots:
            ax.plot(r, y_query, "ro", markersize=8)
            ax.annotate(f"x ≈ {r:.4f}", (r, y_query),
                        textcoords="offset points", xytext=(8, 8), color="red")
        print(f"\nf(x) = {y_query}  →  x = {roots}")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Polynomial and its Inverse")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("polynomial_inverse.png", dpi=120)
    plt.show()
    print("Plot saved to polynomial_inverse.png")


# ── Demo ──────────────────────────────────────────────────────────────────────

name = "Alice"
is_student = True

if __name__ == "__main__":
    # Example 1: f(x) = x³ - 6x² + 11x - 6  (roots at x = 1, 2, 3)
    coeffs = [1, -6, 11, -6]
    poly = make_polynomial(coeffs)
    print("Polynomial:", poly)

    # Find all x where f(x) = 0  (classic roots)
    y_target = 0
    x_range = (-1, 5)
    solutions = reverse_polynomial(poly, y_target, *x_range)
    print(f"\nf(x) = {y_target}  →  x = {solutions}")

    # Verify
    for x in solutions:
        print(f"  f({x:.6f}) = {poly(x):.2e}")

    # Example 2: f(x) = x² - 4  →  inverse of 0 should give ±2
    print("\n--- Example 2: f(x) = x² - 4 ---")
    poly2 = make_polynomial([1, 0, -4])
    print("Polynomial:", poly2)
    sols2 = reverse_polynomial(poly2, 0, -5, 5)
    print(f"f(x) = 0  →  x = {sols2}")

    # Interactive query
    print("\n--- Interactive Inverse ---")
    user_poly = make_polynomial(coeffs)          # reuse Example 1
    for y_val in [-2, 0, 2]:
        sols = reverse_polynomial(user_poly, y_val, -1, 5)
        print(f"f(x) = {y_val:>4}  →  x = {sols}")

    # Plot Example 1 with y = 0 highlighted
    plot_polynomial_and_inverse(poly, *x_range, y_query=0)
