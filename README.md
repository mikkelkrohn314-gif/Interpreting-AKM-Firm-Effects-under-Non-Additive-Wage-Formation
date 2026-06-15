# Interpreting AKM Firm Effects under Non-Additive Wage Formation

This project studies how the AKM firm wage premium should be interpreted when
wages are not additively separable in worker and firm types. The code builds a
search-and-matching model of the labor market with on-the-job search, Nash
bargaining, and complementary (non-additive) production, and shows how and when
a single estimated firm effect misrepresents the underlying type-specific firm
premia.

## Contents

A single self-contained script, `afleveringskode.py`, that:

1. Solves the structural model in closed form (a 2×2 worker × firm type economy).
2. Simulates a worker-firm panel from the model.
3. Estimates a two-way (worker + firm) fixed-effects AKM regression on the panel.
4. Prints three result tables and produces five figures, including a 10×10
   cell-residual extension of the model.

The script is organized into `# %%` cells and runs top-to-bottom.

## Running

```bash
pip install numpy pandas matplotlib
python afleveringskode.py
