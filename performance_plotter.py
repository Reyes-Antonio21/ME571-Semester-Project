# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import LogFormatter, LogLocator

# -------------------------------------------------------------- Fitting functions --------------------------------------------------------------
# Cubit Polynomial Fit
def fitFunc1(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

# Quadratic Polynomial Fit
def fitFunc2(x, a, b, c):
    return a * x ** 2 + b * x + c

# Linear Fit
def fitFunc3(x, a, b):
    return a * x + b

# Cubit Polynomial Fit for FLOP performance
# Note: The coefficients are determined from the total number of FLOPs within each respective section for a general problem size, N. 
# The expression was simplied to the form shown below.
def fitFunc4(x, d):
    return d *(67.0 * x ** 3 + 104.0 * x ** 2 + 77.0 * x)

# Power Law Fit
def fitFunc5(x, a, b):
    return a * x ** b

# -------------------------------------------------------------- CSV Data -------------------------------------------------------------- 
# Path to the CSV files
# Note: The path should be updated to the location of your CSV files
path ='C:/Users/Antonio Reyes/OneDrive/Documents/ME 571 Parallel Scientific Computing/Shallow_Water_Equations_Performance_csv_Files/'
cudaKernelPerformance = 'Shallow_Water_Equations_Cuda_Kernel_Runtime_Performance.csv'
cudaTotalPerformance = 'Shallow_Water_Equations_Cuda_Total_Runtime_Performance.csv'
mpiTotalPerformance = 'Shallow_Water_Equations_Mpi_Total_Runtime_Performance.csv'
mpiSectionPerformance = 'Shallow_Water_Equations_Mpi_Section_Runtime_Performance.csv'
serialTotalPerformance = 'Shallow_Water_Equations_Serial_Total_Runtime_Performance.csv'
serialSectionPerformance = 'Shallow_Water_Equations_Serial_Section_Runtime_Performance.csv'

# -------------------------------------------------------------- parallel total performance --------------------------------------------------------------
# Read the CSV file
pTPData = pd.read_csv(path + cudaTotalPerformance, header = 0, names = ['Problem size', 'Time steps','Iteration', 'Elapsed time (s)', 'Host-device transfer time (s)', 'Device-Host transfer time (s)'])

# Define the columns to average
columns_to_avg = [
    "Elapsed time (s)",
    "Host-device transfer time (s)",
    "Device-Host transfer time (s)"
]

# Group by 'Problem size' and calculate the mean and std deviation
data_to_avg = pTPData.groupby('Problem size')[columns_to_avg]
averaged_data = data_to_avg.mean()
std_data = data_to_avg.std()

# Rename std columns to distinguish them
std_data.columns = [col + " (std dev)" for col in std_data.columns]

pTPData = averaged_data.join(std_data, how='left').reset_index()

pTPData.columns = ['Problem size', 'Avg elapsed time (s)', 'Avg host-device transfer time (s)', 'Avg device-host transfer time (s)', 'Std elapsed time (s)', 'Std host-device transfer time (s)', 'Std device-host transfer time (s)']
problemSizept = pTPData['Problem size']
avgParallelElapsedTime = pTPData['Avg elapsed time (s)']
avgHostDeviceTransfer = pTPData['Avg host-device transfer time (s)']
avgDeviceHostTransfer = pTPData['Avg device-host transfer time (s)']

# Fit the model to the data
poptpt, pcovpt = curve_fit(fitFunc1, problemSizept, avgParallelElapsedTime)
apt, bpt, cpt, dpt = poptpt[0], poptpt[1], poptpt[2], poptpt[3]

# Predict y values
y_predpt = fitFunc1(problemSizept, apt, bpt, cpt, dpt)

# Compute R²
ss_respt = np.sum((avgParallelElapsedTime - y_predpt)**2)
ss_totpt = np.sum((avgParallelElapsedTime - np.mean(avgParallelElapsedTime))**2)
r_squaredpt = 1 - (ss_respt / ss_totpt)

print(f"R\u00B2 Parallel Elapsed Time: {r_squaredpt:.4f}")

plt.figure(figsize=(10,6))
plt.plot(problemSizept, y_predpt, color = 'pink', label = f'Elapsed Time Curve Fit, a = {apt:.4e}, b = {bpt:.4e}, c = {cpt:.4e}, d = {dpt:.4e}')
plt.scatter(problemSizept, avgParallelElapsedTime, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

# Fit the model to the data
poptdthd, pcovdthd = curve_fit(fitFunc2, problemSizept, avgHostDeviceTransfer)
adthd, bdthd, cdthd = poptdthd[0], poptdthd[1], poptdthd[2]

# Predict y values
y_preddthd = fitFunc2(problemSizept, adthd, bdthd, cdthd)

# Compute R\u00B2
ss_resdthd = np.sum((avgHostDeviceTransfer - y_preddthd)**2)
ss_totdthd = np.sum((avgHostDeviceTransfer - np.mean(avgHostDeviceTransfer))**2)
r_squareddthd = 1 - (ss_resdthd / ss_totdthd)

print(f"R\u00B2 Host-Device Transfer Time: {r_squareddthd:.4f}")

plt.figure(figsize=(10,6))
plt.plot(problemSizept, y_preddthd * 1000, color = 'green', label= f'Host-Device Data Transfer Curve Fit, a = {adthd:.4e}, b = {bdthd:.4e}, c = {cdthd:.4e}')
plt.scatter(problemSizept, avgHostDeviceTransfer * 1000, color = 'blue', marker = '.', label = 'Average Host-Device Data Transfer Time')
plt.xlabel("Problem Size")
plt.ylabel("Host-Device Data Transfer Time (ms)")
plt.legend()
plt.show()

# Fit the model to the data
poptdtdh, pcovdtdh = curve_fit(fitFunc2, problemSizept, avgDeviceHostTransfer)
adtdh, bdtdh, cdtdh = poptdtdh[0], poptdtdh[1], poptdtdh[2]

# Predict y values
y_preddtdh = fitFunc2(problemSizept, adtdh, bdtdh, cdtdh)

# Compute R\u00B2
ss_resdtdh = np.sum((avgDeviceHostTransfer - y_preddtdh)**2)
ss_totdtdh = np.sum((avgDeviceHostTransfer - np.mean(avgDeviceHostTransfer))**2)
r_squareddtdh = 1 - (ss_resdtdh / ss_totdtdh)

print(f"R\u00B2 Device-Host Transfer Time: {r_squareddtdh:.4f}")

plt.figure(figsize=(10,6))
plt.plot(problemSizept, y_preddtdh * 1000, color = 'green', label= f'Device Data Transfer Curve Fit, a = {adtdh:.4e}, b = {bdtdh:.4e}, c = {cdtdh:.4e}')
plt.scatter(problemSizept, avgDeviceHostTransfer * 1000, color = 'blue', marker = '.', label = 'Average Host-Device Data Transfer Time')
plt.xlabel("Problem Size")
plt.ylabel("Device-Host Data Transfer Time (ms)")
plt.legend()
plt.show()

# -------------------------------------------------------------- parallel kernel performance --------------------------------------------------------------
# Read the CSV file
pKPData = pd.read_csv(path + cudaKernelPerformance, header = 0, names = ['Problem size', 'Time steps', 'Iteration', 'Elapsed time (s)', 'Avg compute fluxes time (s)', 'Avg compute variables time (s)', 'Avg update variables time (s)', 'Avg apply boundary conditions time (s)'])

# Define the columns to average
columns_to_avg = [
    "Elapsed time (s)",
    "Avg compute fluxes time (s)",
    "Avg compute variables time (s)",
    "Avg update variables time (s)",
    "Avg apply boundary conditions time (s)"
]

# Group by 'Problem size' and calculate the mean and std deviation
data_to_avg = pKPData.groupby('Problem size')[columns_to_avg]
averaged_data = data_to_avg.mean()
std_data = data_to_avg.std()

# Rename std columns to distinguish them
std_data.columns = [col + " (std dev)" for col in std_data.columns]

pKPData = averaged_data.join(std_data, how='left').reset_index()

# Rename columns for clarity
pKPData.columns = ['Problem size', 'Avg elapsed time (s)', 'Avg compute fluxes time (s)', 'Avg compute variables time (s)', 'Avg update variables time (s)', 'Avg apply boundary conditions time (s)', 'Std elapsed time (s)', 'Std compute fluxes (s)', 'Std compute variables (s)', 'Std update variables (s)', 'Std apply boundary conditions (s)']
problemSizepk = pKPData['Problem size']
avgComputeFluxpk = pKPData['Avg compute fluxes time (s)']
avgComputeVariablespk = pKPData['Avg compute variables time (s)']
avgUpdateVariablespk = pKPData['Avg update variables time (s)']
avgApplyBoundaryConditionspk = pKPData['Avg apply boundary conditions time (s)']

# Fit the model to the data
poptpk1, pcovpk1 = curve_fit(fitFunc2, problemSizepk, avgComputeFluxpk)
apk1, bpk1, cpk1 = poptpk1[0], poptpk1[1], poptpk1[2]

# Predict y values
y_predpk1 = fitFunc2(problemSizepk, apk1, bpk1, cpk1)

# Compute R²
ss_respk1 = np.sum((avgComputeFluxpk - y_predpk1)**2)
ss_totpk1 = np.sum((avgComputeFluxpk - np.mean(avgComputeFluxpk))**2)
r_squaredpk1 = 1 - (ss_respk1 / ss_totpk1)

print(f"R\u00B2: {r_squaredpk1:.4f}")

# Fit the model to the data
poptpk2, pcovpk2 = curve_fit(fitFunc2, problemSizepk, avgComputeVariablespk)
apk2, bpk2, cpk2 = poptpk2[0], poptpk2[1], poptpk2[2]

# Predict y values
y_predpk2 = fitFunc2(problemSizepk, apk2, bpk2, cpk2)

# Compute R²
ss_respk2 = np.sum((avgComputeVariablespk - y_predpk2)**2)
ss_totpk2 = np.sum((avgComputeVariablespk - np.mean(avgComputeVariablespk))**2)
r_squaredpk2 = 1 - (ss_respk2 / ss_totpk2)

print(f"R\u00B2: {r_squaredpk2:.4f}")

# Fit the model to the data
poptpk3, pcovpk3 = curve_fit(fitFunc2, problemSizepk, avgUpdateVariablespk)
apk3, bpk3, cpk3 = poptpk3[0], poptpk3[1], poptpk3[2]

# Predict y values
y_predpk3 = fitFunc2(problemSizepk, apk3, bpk3, cpk3)

# Compute R²
ss_respk3 = np.sum((avgUpdateVariablespk - y_predpk3)**2)
ss_totpk3 = np.sum((avgUpdateVariablespk - np.mean(avgUpdateVariablespk))**2)
r_squaredpk3 = 1 - (ss_respk3 / ss_totpk3)

print(f"R\u00B2: {r_squaredpk3:.4f}")

# Fit the model to the data
poptpk4, pcovpk4 = curve_fit(fitFunc2, problemSizepk, avgApplyBoundaryConditionspk)
apk4, bpk4, cpk4 = poptpk4[0], poptpk4[1], poptpk4[2]

# Predict y values
y_predpk4 = fitFunc2(problemSizepk, apk4, bpk4, cpk4)

# Compute R²
ss_respk4 = np.sum((avgApplyBoundaryConditionspk - y_predpk4)**2)
ss_totpk4 = np.sum((avgApplyBoundaryConditionspk - np.mean(avgApplyBoundaryConditionspk))**2)
r_squaredpk4 = 1 - (ss_respk4 / ss_totpk4)

print(f"R\u00B2: {r_squaredpk4:.4f}")

plt.figure(figsize=(12,7))
plt.plot(problemSizepk, y_predpk1 * 1000, color = 'purple', label= f'Compute Fluxes Curve Fit, a = {apk1:.4e}, b = {bpk1:.4e}, c = {cpk1:.4e}')
plt.plot(problemSizepk, y_predpk2 * 1000, color = 'green', label= f'Compute Variables Curve Fit, a = {apk2:.4e}, b = {bpk2:.4e}, c = {cpk2:.4e}')
plt.plot(problemSizepk, y_predpk3 * 1000, color = 'blue', label= f'Update Variables Curve Fit, a = {apk3:.4e}, b = {bpk3:.4e}, c = {cpk3:.4e}')
plt.plot(problemSizepk, y_predpk4 * 1000, color = 'pink', label= f'Apply Boundary Conditions Curve Fit, a = {apk4:.4e}, b = {bpk4:.4e}, c = {cpk4:.4e}')
plt.scatter(problemSizepk, avgComputeFluxpk * 1000, color = 'orange', marker = '.', label = 'Compute Fluxes')
plt.scatter(problemSizepk, avgComputeVariablespk * 1000, color = 'grey', marker = '.', label = 'Compute Variables')
plt.scatter(problemSizepk, avgUpdateVariablespk * 1000, color = 'brown', marker = '.', label = 'Update Variables')
plt.scatter(problemSizepk, avgApplyBoundaryConditionspk * 1000, color = 'red', marker = '.', label = 'Apply Boundary Conditions')
plt.xlabel("Problem Size")
plt.ylabel("Average Kernel Execution Time (ms)")
plt.legend()
plt.show()

# -------------------------------------------------------------- serial total performance -------------------------------------------------------------- 
# Read the CSV file
sTPData = pd.read_csv(path + serialTotalPerformance, header = 0, names = ['Problem size', 'Time steps', 'Iteration', 'Elapsed time (s)'])

# Define the columns to average
columns_to_avg = [
    "Elapsed time (s)"
]

# Group by 'Problem size' and calculate the mean and std deviation
data_to_avg = sTPData.groupby('Problem size')[columns_to_avg]
averaged_data = data_to_avg.mean()
std_data = data_to_avg.std()

# Rename std columns to distinguish them
std_data.columns = [col + " (std dev)" for col in std_data.columns]

sTPData = averaged_data.join(std_data, how='left').reset_index()

sTPData.columns = ['Problem size', 'Avg elapsed time (s)', 'Std elapsed time (s)']
problemSizest = sTPData['Problem size']
avgSerialElapsedTime = sTPData['Avg elapsed time (s)']

# Fit the model to the data
poptst1, pcovst1 = curve_fit(fitFunc1, problemSizest, avgSerialElapsedTime)
ast1, bst1, cst1, dst1 = poptst1[0], poptst1[1], poptst1[2], poptst1[3]

# Predict y values
y_predst1 = fitFunc1(problemSizest, ast1, bst1, cst1, dst1)

# Compute R²
ss_resst1 = np.sum((avgSerialElapsedTime - y_predst1)**2)
ss_totst1 = np.sum((avgSerialElapsedTime - np.mean(avgSerialElapsedTime))**2)
r_squaredst1 = 1 - (ss_resst1 / ss_totst1)

print(f"R\u00B2: {r_squaredst1:.4f}")

# Fit the model to the data
poptst2, pcovst2 = curve_fit(fitFunc4, problemSizest, avgSerialElapsedTime)
ast2 = poptst2[0]

# Predict y values
y_predst2 = fitFunc4(problemSizest, ast2)

# Compute R²
ss_resst2 = np.sum((avgSerialElapsedTime - y_predst2)**2)
ss_totst2 = np.sum((avgSerialElapsedTime - np.mean(avgSerialElapsedTime))**2)
r_squaredst2 = 1 - (ss_resst2 / ss_totst2)

print(f"R\u00B2: {r_squaredst2:.4f}")

plt.figure(figsize=(10,6))
plt.plot(problemSizest, y_predst1, color = 'pink', label= f'Elapsed Time Curve Fit, a = {ast1:.4e}, b = {bst1:.4e}, c = {cst1:.4e}, d = {dst1:.4e}')
plt.scatter(problemSizest, avgSerialElapsedTime, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(problemSizest, y_predst2, color='pink', label=f'FLOP Performance Curve Fit, \u03b4dt = {ast2:.4e}')
plt.scatter(problemSizest, avgSerialElapsedTime, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

# -------------------------------------------------------------- serial section performance --------------------------------------------------------------
# Read the CSV file
sSPData = pd.read_csv(path + serialSectionPerformance, header = 0, names = ['Problem size', 'Time steps', 'Iteration', 'Elapsed time (s)', 'Avg compute fluxes time (s)', 'Avg compute variables time (s)', 'Avg update variables time (s)', 'Avg apply boundary conditions time (s)'])

# Define the columns to average
columns_to_avg = [
    "Elapsed time (s)",
    "Avg compute fluxes time (s)",
    "Avg compute variables time (s)",
    "Avg update variables time (s)",
    "Avg apply boundary conditions time (s)"
]

# Group by 'Problem size' and calculate the mean and std deviation
data_to_avg = sSPData.groupby('Problem size')[columns_to_avg]
averaged_data = data_to_avg.mean()
std_data = data_to_avg.std()

# Rename std columns to distinguish them
std_data.columns = [col + " (std dev)" for col in std_data.columns]

sSPData = averaged_data.join(std_data, how='left').reset_index()

# Rename columns for clarity
sSPData.columns = ['Problem size', 'Avg elapsed time (s)', 'Avg compute fluxes time (s)', 'Avg compute variables time (s)', 'Avg update variables time (s)', 'Avg apply boundary conditions time (s)', 'Std elapsed time (s)', 'Std compute fluxes (s)', 'Std compute variables (s)', 'Std update variables (s)', 'Std apply boundary conditions (s)']
problemSizess = sSPData['Problem size']
avgComputeFluxss = sSPData['Avg compute fluxes time (s)']
avgComputeVariablesss = sSPData['Avg compute variables time (s)']
avgUpdateVariablesss = sSPData['Avg update variables time (s)']
avgApplyBoundaryConditionsss = sSPData['Avg apply boundary conditions time (s)']

# Fit the model to the data
poptss1, pcovss1 = curve_fit(fitFunc2, problemSizess, avgComputeFluxss)
ass1, bss1, css1 = poptss1[0], poptss1[1], poptss1[2]

# Predict y values
y_predss1 = fitFunc2(problemSizess, ass1, bss1, css1)

# Compute R²
ss_resss1 = np.sum((avgComputeFluxss - y_predss1)**2)
ss_totss1 = np.sum((avgComputeFluxss - np.mean(avgComputeFluxss))**2)
r_squaredss1 = 1 - (ss_resss1 / ss_totss1)

print(f"R\u00B2: {r_squaredss1:.4f}")

# Fit the model to the data
poptss2, pcovss2 = curve_fit(fitFunc2, problemSizess, avgComputeVariablesss)
ass2, bss2, css2 = poptss2[0], poptss2[1], poptss2[2]

# Predict y values
y_predss2 = fitFunc2(problemSizess, ass2, bss2, css2)

# Compute R²
ss_resss2 = np.sum((avgComputeVariablesss - y_predss2)**2)
ss_totss2 = np.sum((avgComputeVariablesss - np.mean(avgComputeVariablesss))**2)
r_squaredss2 = 1 - (ss_resss2 / ss_totss2)

print(f"R\u00B2: {r_squaredss2:.4f}")

# Fit the model to the data
poptss3, pcovss3 = curve_fit(fitFunc2, problemSizess, avgUpdateVariablesss)
ass3, bss3, css3 = poptss3[0], poptss3[1], poptss3[2]

# Predict y values
y_predss3 = fitFunc2(problemSizess, ass3, bss3, css3)

# Compute R²
ss_resss3 = np.sum((avgUpdateVariablesss - y_predss3)**2)
ss_totss3 = np.sum((avgUpdateVariablesss - np.mean(avgUpdateVariablesss))**2)
r_squaredss3 = 1 - (ss_resss3 / ss_totss3)

print(f"R\u00B2: {r_squaredss3:.4f}")

# Fit the model to the data
poptss4, pcovss4 = curve_fit(fitFunc2, problemSizess, avgApplyBoundaryConditionsss)
ass4, bss4, css4 = poptss4[0], poptss4[1], poptss4[2]

# Predict y values
y_predss4 = fitFunc2(problemSizess, ass4, bss4, css4)

# Compute R²
ss_resss4 = np.sum((avgApplyBoundaryConditionsss - y_predss4)**2)
ss_totss4 = np.sum((avgApplyBoundaryConditionsss - np.mean(avgApplyBoundaryConditionsss))**2)
r_squaredss4 = 1 - (ss_resss4 / ss_totss4)

print(f"R\u00B2: {r_squaredss4:.4f}")

plt.figure(figsize=(12,7))
plt.plot(problemSizess, y_predss1 * 1000, color = 'purple', label= f'Compute Fluxes Curve Fit, a = {ass1:.4e}, b = {bss1:.4e}, c = {css1:.4e}')
plt.plot(problemSizess, y_predss2 * 1000, color = 'green', label= f'Compute Variables Curve Fit, a = {ass2:.4e}, b = {bss2:.4e}, c = {css2:.4e}')
plt.plot(problemSizess, y_predss3 * 1000, color = 'blue', label= f'Update Variables Curve Fit, a = {ass3:.4e}, b = {bss3:.4e}, c = {css3:.4e}')
plt.plot(problemSizess, y_predss4 * 1000, color = 'pink', label= f'Apply Boundary Conditions Curve Fit, a = {ass4:.4e}, b = {bss4:.4e}, c = {css4:.4e}')
plt.scatter(problemSizess, avgComputeFluxss * 1000, color = 'orange', marker = '.', label = 'Compute Fluxes')
plt.scatter(problemSizess, avgComputeVariablesss * 1000, color = 'grey', marker = '.', label = 'Compute Variables')
plt.scatter(problemSizess, avgUpdateVariablesss * 1000, color = 'brown', marker = '.', label = 'Update Variables')
plt.scatter(problemSizess, avgApplyBoundaryConditionsss * 1000, color = 'red', marker = '.', label = 'Apply Boundary Conditions')
plt.xlabel("Problem Size")
plt.ylabel("Average Section Execution Time (ms)")
plt.legend()
plt.show()

# -------------------------------------------------------------- MPI total performance --------------------------------------------------------------

mTPData = pd.read_csv(path + mpiTotalPerformance, header = 0, names = ['Problem size', 'Number of processors', 'Time steps', 'Iteration', 'Elapsed time (s)'])

x = np.linspace(0, 50, 50)
y = np.ones_like(x)

# Define the columns to average
columns_to_avg = [
    "Elapsed time (s)"
]

# Group by 'Problem size' and calculate the mean and std deviation
data_to_avg = mTPData.groupby(['Problem size', 'Number of processors'])[columns_to_avg]
averaged_data = data_to_avg.mean()
std_data = data_to_avg.std()

# Rename std columns to distinguish them
std_data.columns = [col + " (std dev)" for col in std_data.columns]

mTPData = averaged_data.join(std_data, how='left').reset_index()

mTPData.columns = ['Problem size', 'Number of processors', 'Avg elapsed time (s)', 'Std elapsed time (s)']
problemSizemT = mTPData['Problem size']
numberOfProcessors = mTPData['Number of processors']
avgParallelElapsedTimemT = mTPData['Avg elapsed time (s)']

plt.figure(figsize=(16,9))

for problem_size in problemSizemT.unique():
    subset = mTPData[problemSizemT == problem_size]
    plt.plot(subset['Number of processors'], subset['Avg elapsed time (s)'], marker='.', label=f'Problem Size: {problem_size}')

plt.xlabel("Number of Processors")
plt.ylabel("Average Time to Solution (s)")  
plt.title("Average Elapsed Time vs Number of Processors")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()

# Choose a fixed problem size for strong scaling
problem_size_to_analyze = 5400  # Change this to whichever size you want
subset = mTPData[problemSizemT == problem_size_to_analyze]

# Sort by number of processors
subset = subset.sort_values(by='Number of processors')

# Compute speedup and efficiency
T1 = subset[subset['Number of processors'] == 1]['Avg elapsed time (s)'].values[0]
subset['Speedup'] = T1 / subset['Avg elapsed time (s)']
subset['Efficiency'] = subset['Speedup'] / subset['Number of processors']

# === 3-in-1 Plot ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Elapsed Time
axes[0].plot(subset['Number of processors'], subset['Avg elapsed time (s)'], marker='.')
axes[0].set_xlabel('Number of Processors')
axes[0].set_ylabel('Avg Elapsed Time (s)')
axes[0].set_title(f'Elapsed Time\n(Problem size {problem_size_to_analyze})')
axes[0].grid(True)
axes[0].set_xscale('log', base=2)
axes[0].set_yscale('log')

# Panel 2: Speedup
axes[1].plot(subset['Number of processors'], subset['Speedup'], marker='.', label='Measured Speedup')
axes[1].plot(subset['Number of processors'], subset['Number of processors'], linestyle='--', color='gray', label='Ideal Speedup')
axes[1].set_xlabel('Number of Processors')
axes[1].set_ylabel('Speedup')
axes[1].set_title('Speedup')
axes[1].legend()
axes[1].grid(True)
axes[1].set_xscale('log', base=2)
axes[1].set_yscale('log')

# Panel 3: Parallel Efficiency
axes[2].plot(subset['Number of processors'], subset['Efficiency'], marker='.')
plt.plot(x, y, c = 'k', linestyle = '--', label = 'Ideal Efficiency')
axes[2].set_xlabel('Number of Processors')
axes[2].set_ylabel('Efficiency')
axes[2].set_title('Parallel Efficiency')
axes[2].grid(True)
axes[2].set_xscale('log', base=2)
axes[2].set_ylim(0, 1.1)  # Efficiency should be between 0 and 1

# Final Layout
plt.suptitle(f'Strong Scaling Analysis for Problem Size {problem_size_to_analyze}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# -------------------------------------------------------------- MPI section performance --------------------------------------------------------------

mSPData = pd.read_csv(path + mpiSectionPerformance, header = 0, names = ['Problem size', 'Number of processors', 'Time steps', 'Iteration','Elapsed time (s)','Avg compute fluxes time (s)','Avg compute variables time (s)','Avg update variables time (s)', 'Avg apply boundary conditions time (s)', 'Avg data transfer time (s)'])

# Define the columns to average
columns_to_avg = [
    "Elapsed time (s)",
    "Avg compute fluxes time (s)",
    "Avg compute variables time (s)",
    "Avg update variables time (s)",
    "Avg apply boundary conditions time (s)",
    "Avg data transfer time (s)"
]

# Group by 'Problem size' and calculate the mean and std deviation
data_to_avg = mSPData.groupby(['Problem size', 'Number of processors'])[columns_to_avg]
averaged_data = data_to_avg.mean()
std_data = data_to_avg.std()

# Rename std columns to distinguish them
std_data.columns = [col + " (std dev)" for col in std_data.columns]

mSPData = averaged_data.join(std_data, how='left').reset_index()

mSPData.columns = ['Problem size', 'Number of processors', 'Avg elapsed time (s)', 'Avg compute fluxes time (s)', 'Avg compute variables time (s)', 'Avg update variables time (s)', 'Avg apply boundary conditions time (s)', 'Avg data transfer time (s)', 'Std elapsed time (s)', 'Std compute fluxes (s)', 'Std compute variables (s)', 'Std update variables (s)', 'Std apply boundary conditions (s)', 'Std data transfer (s)']
problemSizemS = mSPData['Problem size']
numberOfProcessorsmS = mSPData['Number of processors']
avgComputeFluxesmS = mSPData['Avg compute fluxes time (s)']
avgComputeVariablesmS = mSPData['Avg compute variables time (s)']
avgUpdateVariablesmS = mSPData['Avg update variables time (s)']
avgApplyBoundaryConditionsmS = mSPData['Avg apply boundary conditions time (s)']
avgDataTransfermS = mSPData['Avg data transfer time (s)']

# -------------------------------------------------------------- Parallel to serial SECTION performance comparison --------------------------------------------------------------
# Find the minimum length
minLen1 = min(len(avgComputeFluxss), len(avgComputeFluxpk))
minLen2 = min(len(avgComputeVariablesss), len(avgComputeVariablespk))
minLen3 = min(len(avgUpdateVariablesss), len(avgUpdateVariablespk))
minLen4 = min(len(avgApplyBoundaryConditionsss), len(avgApplyBoundaryConditionspk))
minLen5 = min(len(problemSizess), len(problemSizepk))

# Truncate DataFrames
avgComputeFluxss = avgComputeFluxss.iloc[:minLen1] * 1000 
avgComputeVariablesss = avgComputeVariablesss.iloc[:minLen2] * 1000 
avgUpdateVariablesss = avgUpdateVariablesss.iloc[:minLen3] * 1000 
avgApplyBoundaryConditionsss = avgApplyBoundaryConditionsss.iloc[:minLen4] * 1000 
problemSizess = problemSizess.iloc[:minLen5] * 1000 

avgComputeFluxpk = avgComputeFluxpk.iloc[:minLen1] * 1000 
avgComputeVariablespk = avgComputeVariablespk.iloc[:minLen2] * 1000 
avgUpdateVariablespk = avgUpdateVariablespk.iloc[:minLen3] * 1000 
avgApplyBoundaryConditionspk = avgApplyBoundaryConditionspk.iloc[:minLen4] * 1000 
problemSizepk = problemSizepk.iloc[:minLen5] * 1000   

# Kernel to serial section comparison
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# Set common log tick locators (optional)
log_major = LogLocator(base=10.0, numticks=10)
log_minor = LogLocator(base=2.0, subs=np.arange(2, 10)*0.1, numticks=10)

# ---- Subplot 1 ----
ax = axs[0, 0]
ax.scatter(problemSizess, avgComputeFluxss, color='red', marker='*', label='Serial Compute Fluxes')
ax.scatter(problemSizepk, avgComputeFluxpk, color='black', marker='.', label='Kernel Compute Fluxes')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

# ---- Subplot 2 ----
ax = axs[0, 1]
ax.scatter(problemSizess, avgComputeVariablesss, color='violet', marker='*', label='Serial Compute Variables')
ax.scatter(problemSizepk, avgComputeVariablespk, color='orange', marker='.', label='Kernel Compute Variables')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

# ---- Subplot 3 ----
ax = axs[1, 0]
ax.scatter(problemSizess, avgUpdateVariablesss, color='pink', marker='*', label='Serial Update Variables')
ax.scatter(problemSizepk, avgUpdateVariablespk, color='blue', marker='.', label='Kernel Update Variables')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

# ---- Subplot 4 ----
ax = axs[1, 1]
ax.scatter(problemSizess, avgApplyBoundaryConditionsss, color='grey', marker='*', label='Serial Apply Boundary Conditions')
ax.scatter(problemSizepk, avgApplyBoundaryConditionspk, color='brown', marker='.', label='Kernel Apply Boundary Conditions')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size: (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------- Parallel to serial TOTAL performance comparison --------------------------------------------------------------
#Find Minimum Length
minLen6 = min(len(avgParallelElapsedTime), len(avgSerialElapsedTime))
minLen7 = min(len(problemSizest), len(problemSizept))

# Truncate dataframes
avgSerialElapsedTime = avgSerialElapsedTime.iloc[:minLen6]
problemSizest = problemSizept.iloc[:minLen7]

avgParallelElapsedTime = avgParallelElapsedTime.iloc[:minLen7]
problemSizept = problemSizept.iloc[:minLen7]

# Speedup Comparison
speedup = avgSerialElapsedTime / avgParallelElapsedTime
plt.figure(figsize=(10,6))
plt.scatter(problemSizest, speedup, color = 'blue', marker = '.')    
plt.xlabel("Problem Size (N)")
plt.ylabel("Average Speedup") 
plt.show() 

# Serial & Parallel Total Runtime Comparison
fig, axs = plt.subplots(figsize=(10, 6))

# Set common log tick locators (optional)
log_major = LogLocator(base=10.0, numticks=10)
log_minor = LogLocator(base=2.0, subs=np.arange(2, 10)*0.1, numticks=10)

# Parallel to serial comparison
ax = axs
ax.scatter(problemSizest, avgSerialElapsedTime, color = 'brown', marker = '*', label = 'Average Serial Elapsed Time')
ax.scatter(problemSizept, avgParallelElapsedTime, color = 'blue', marker = '.', label = 'Average Parallel Elapsed Time')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (s)")
ax.legend()
plt.show()