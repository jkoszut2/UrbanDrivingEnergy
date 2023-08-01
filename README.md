# Urban Driving Energy Consumption

### Description
This work was used as part of the final project for CE 295 - Data Science for Energy (formerly known as Energy Systems and Control). At the core, the goal was to study how passenger vehicle energy consumption in an urban setting is affected by the environment, e.g., configurations of traffic lights, and driving style, e.g., how knowledge of traffic light timing can allow a vehicle to travel slower and use less energy without taking any longer to complete a journey. The report for the final project is included in this repository.

### Required Packages
The following packages are required: `numpy`, `pandas`, `matplotlib`, and `cvxpy`.

### Solvers
The open-source `ECOS_BB` solver which comes with CVXPY can be used to solve the mixed-integer second order cone program. The `MOSEK` solver has the potential to converge faster but requires a license.

### Running
After ensuring that the required packages and desired solver are installed, open the command line, navigate to the location of this repository, and run `python main.py`.

### Example Output
![test](https://raw.githubusercontent.com/jkoszut2/UrbanDrivingEnergy/main/Comparison.png)
