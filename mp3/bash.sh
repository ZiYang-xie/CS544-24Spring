#!/bin/bash

# Configuration 1: Fix nums (variables), vary num_eqs (equations)
echo "Configuration 1: Fixing number of variables, varying number of equations"
fixed_vars=100  # Adjust as needed
for num_eqs in $(seq 20 10 50); do  # Example: 10, 15, 20, ..., 50
    echo "Running experiment with ${fixed_vars} variables and ${num_eqs} equations..."
    python run.py --num_vars=${fixed_vars} --num_eqs=${num_eqs} --vis
done

echo "Configuration 1 completed."

# Configuration 2: Fix num_eqs (equations), vary nums (variables)
echo "Configuration 2: Fixing number of equations, varying number of variables"
fixed_eqs=10  # Adjust as needed
for num_vars in $(seq 30 30 60); do  # Example: 30, 40, 50, ..., 100
    echo "Running experiment with ${num_vars} variables and ${fixed_eqs} equations..."
    python run.py --num_vars=${num_vars} --num_eqs=${fixed_eqs} --vis
done

echo "Configuration 2 completed."


# Large scale experiment
# num_vars=200
# num_eqs=100
# python run.py --num_vars=${num_vars} --num_eqs=${num_eqs} --vis

