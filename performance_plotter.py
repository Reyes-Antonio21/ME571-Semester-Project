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
path ='C:/Users/anton/OneDrive/Documents/ME 571 Parallel Scientific Computing/Shallow_Water_Equations_Performance_csv_Files/'
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
problemSizect = pTPData['Problem size']
avgParallelElapsedTimect = pTPData['Avg elapsed time (s)']
avgHostDeviceTransfer = pTPData['Avg host-device transfer time (s)']
avgDeviceHostTransfer = pTPData['Avg device-host transfer time (s)']

# Fit the model to the data
poptpt, pcovpt = curve_fit(fitFunc1, problemSizect, avgParallelElapsedTimect)
apt, bpt, cpt, dpt = poptpt[0], poptpt[1], poptpt[2], poptpt[3]

# Predict y values
y_predpt = fitFunc1(problemSizect, apt, bpt, cpt, dpt)

# Compute R²
ss_respt = np.sum((avgParallelElapsedTimect - y_predpt)**2)
ss_totpt = np.sum((avgParallelElapsedTimect - np.mean(avgParallelElapsedTimect))**2)
r_squaredpt = 1 - (ss_respt / ss_totpt)

print(f"R\u00B2 Parallel Elapsed Time: {r_squaredpt:.4f}")

plt.figure(figsize=(12,7))
plt.plot(problemSizect, y_predpt, color = 'pink', label = f'Elapsed Time Curve Fit, a = {apt:.4e}, b = {bpt:.4e}, c = {cpt:.4e}, d = {dpt:.4e}')
plt.scatter(problemSizect, avgParallelElapsedTimect, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

# Fit the model to the data
poptdthd, pcovdthd = curve_fit(fitFunc1, problemSizect, avgHostDeviceTransfer)
adthd, bdthd, cdthd, ddthd = poptdthd[0], poptdthd[1], poptdthd[2], poptdthd[3]

# Predict y values
y_preddthd = fitFunc1(problemSizect, adthd, bdthd, cdthd, ddthd)

# Compute R\u00B2
ss_resdthd = np.sum((avgHostDeviceTransfer - y_preddthd)**2)
ss_totdthd = np.sum((avgHostDeviceTransfer - np.mean(avgHostDeviceTransfer))**2)
r_squareddthd = 1 - (ss_resdthd / ss_totdthd)

print(f"R\u00B2 Host-Device Transfer Time: {r_squareddthd:.4f}")

plt.figure(figsize=(12,7))
plt.plot(problemSizect, y_preddthd * 1000, color = 'green', label= f'Host-Device Data Transfer Curve Fit, a = {adthd:.4e}, b = {bdthd:.4e}, c = {cdthd:.4e}, d = {ddthd:.4e}')
plt.scatter(problemSizect, avgHostDeviceTransfer * 1000, color = 'blue', marker = '.', label = 'Average Host-Device Data Transfer Time')
plt.xlabel("Problem Size")
plt.ylabel("Host-Device Data Transfer Time (ms)")
plt.legend()
plt.show()

# Fit the model to the data
poptdtdh, pcovdtdh = curve_fit(fitFunc1, problemSizect, avgDeviceHostTransfer)
adtdh, bdtdh, cdtdh, ddtdh = poptdtdh[0], poptdtdh[1], poptdtdh[2], poptdtdh[3]

# Predict y values
y_preddtdh = fitFunc1(problemSizect, adtdh, bdtdh, cdtdh, ddtdh)

# Compute R\u00B2
ss_resdtdh = np.sum((avgDeviceHostTransfer - y_preddtdh)**2)
ss_totdtdh = np.sum((avgDeviceHostTransfer - np.mean(avgDeviceHostTransfer))**2)
r_squareddtdh = 1 - (ss_resdtdh / ss_totdtdh)

print(f"R\u00B2 Device-Host Transfer Time: {r_squareddtdh:.4f}")

plt.figure(figsize=(12,7))
plt.plot(problemSizect, y_preddtdh * 1000, color = 'green', label= f'Device Data Transfer Curve Fit, a = {adtdh:.4e}, b = {bdtdh:.4e}, c = {cdtdh:.4e}, d = {ddtdh:.4e}')
plt.scatter(problemSizect, avgDeviceHostTransfer * 1000, color = 'blue', marker = '.', label = 'Average Host-Device Data Transfer Time')
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
problemSizeck = pKPData['Problem size']
avgComputeFluxesck = pKPData['Avg compute fluxes time (s)']
avgComputeVariablesck = pKPData['Avg compute variables time (s)']
avgUpdateVariablesck = pKPData['Avg update variables time (s)']
avgApplyBoundaryConditionsck = pKPData['Avg apply boundary conditions time (s)']

# Fit the model to the data
poptpk1, pcovpk1 = curve_fit(fitFunc2, problemSizeck, avgComputeFluxesck)
apk1, bpk1, cpk1 = poptpk1[0], poptpk1[1], poptpk1[2]

# Predict y values
y_predpk1 = fitFunc2(problemSizeck, apk1, bpk1, cpk1)

# Compute R²
ss_respk1 = np.sum((avgComputeFluxesck - y_predpk1)**2)
ss_totpk1 = np.sum((avgComputeFluxesck - np.mean(avgComputeFluxesck))**2)
r_squaredpk1 = 1 - (ss_respk1 / ss_totpk1)

print(f"R\u00B2: {r_squaredpk1:.4f}")

# Fit the model to the data
poptpk2, pcovpk2 = curve_fit(fitFunc2, problemSizeck, avgComputeVariablesck)
apk2, bpk2, cpk2 = poptpk2[0], poptpk2[1], poptpk2[2]

# Predict y values
y_predpk2 = fitFunc2(problemSizeck, apk2, bpk2, cpk2)

# Compute R²
ss_respk2 = np.sum((avgComputeVariablesck - y_predpk2)**2)
ss_totpk2 = np.sum((avgComputeVariablesck - np.mean(avgComputeVariablesck))**2)
r_squaredpk2 = 1 - (ss_respk2 / ss_totpk2)

print(f"R\u00B2: {r_squaredpk2:.4f}")

# Fit the model to the data
poptpk3, pcovpk3 = curve_fit(fitFunc2, problemSizeck, avgUpdateVariablesck)
apk3, bpk3, cpk3 = poptpk3[0], poptpk3[1], poptpk3[2]

# Predict y values
y_predpk3 = fitFunc2(problemSizeck, apk3, bpk3, cpk3)

# Compute R²
ss_respk3 = np.sum((avgUpdateVariablesck - y_predpk3)**2)
ss_totpk3 = np.sum((avgUpdateVariablesck - np.mean(avgUpdateVariablesck))**2)
r_squaredpk3 = 1 - (ss_respk3 / ss_totpk3)

print(f"R\u00B2: {r_squaredpk3:.4f}")

# Fit the model to the data
poptpk4, pcovpk4 = curve_fit(fitFunc2, problemSizeck, avgApplyBoundaryConditionsck)
apk4, bpk4, cpk4 = poptpk4[0], poptpk4[1], poptpk4[2]

# Predict y values
y_predpk4 = fitFunc2(problemSizeck, apk4, bpk4, cpk4)

# Compute R²
ss_respk4 = np.sum((avgApplyBoundaryConditionsck - y_predpk4)**2)
ss_totpk4 = np.sum((avgApplyBoundaryConditionsck - np.mean(avgApplyBoundaryConditionsck))**2)
r_squaredpk4 = 1 - (ss_respk4 / ss_totpk4)

print(f"R\u00B2: {r_squaredpk4:.4f}")

plt.figure(figsize=(12,7))
plt.plot(problemSizeck, y_predpk1 * 1000, color = 'purple', label= f'Compute Fluxes Curve Fit, a = {apk1:.4e}, b = {bpk1:.4e}, c = {cpk1:.4e}')
plt.plot(problemSizeck, y_predpk2 * 1000, color = 'green', label= f'Compute Variables Curve Fit, a = {apk2:.4e}, b = {bpk2:.4e}, c = {cpk2:.4e}')
plt.plot(problemSizeck, y_predpk3 * 1000, color = 'blue', label= f'Update Variables Curve Fit, a = {apk3:.4e}, b = {bpk3:.4e}, c = {cpk3:.4e}')
plt.plot(problemSizeck, y_predpk4 * 1000, color = 'pink', label= f'Apply Boundary Conditions Curve Fit, a = {apk4:.4e}, b = {bpk4:.4e}, c = {cpk4:.4e}')
plt.scatter(problemSizeck, avgComputeFluxesck * 1000, color = 'orange', marker = '.', label = 'Compute Fluxes')
plt.scatter(problemSizeck, avgComputeVariablesck * 1000, color = 'grey', marker = '.', label = 'Compute Variables')
plt.scatter(problemSizeck, avgUpdateVariablesck * 1000, color = 'brown', marker = '.', label = 'Update Variables')
plt.scatter(problemSizeck, avgApplyBoundaryConditionsck * 1000, color = 'red', marker = '.', label = 'Apply Boundary Conditions')
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

plt.figure(figsize=(12,7))
plt.plot(problemSizest, y_predst1, color = 'pink', label= f'Elapsed Time Curve Fit, a = {ast1:.4e}, b = {bst1:.4e}, c = {cst1:.4e}, d = {dst1:.4e}')
plt.scatter(problemSizest, avgSerialElapsedTime, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

plt.figure(figsize=(12,7))
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
avgComputeFluxesss = sSPData['Avg compute fluxes time (s)']
avgComputeVariablesss = sSPData['Avg compute variables time (s)']
avgUpdateVariablesss = sSPData['Avg update variables time (s)']
avgApplyBoundaryConditionsss = sSPData['Avg apply boundary conditions time (s)']

# Fit the model to the data
poptss1, pcovss1 = curve_fit(fitFunc2, problemSizess, avgComputeFluxesss)
ass1, bss1, css1 = poptss1[0], poptss1[1], poptss1[2]

# Predict y values
y_predss1 = fitFunc2(problemSizess, ass1, bss1, css1)

# Compute R²
ss_resss1 = np.sum((avgComputeFluxesss - y_predss1)**2)
ss_totss1 = np.sum((avgComputeFluxesss - np.mean(avgComputeFluxesss))**2)
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
plt.scatter(problemSizess, avgComputeFluxesss * 1000, color = 'orange', marker = '.', label = 'Compute Fluxes')
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
y1 = np.ones_like(x)
y = x
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
problemSizemt = mTPData['Problem size']
numberOfProcessorsmt = mTPData['Number of processors']
avgParallelElapsedTimemt = mTPData['Avg elapsed time (s)']

plt.figure(figsize=(12,7))

for problem_size in problemSizemt.unique():
    subset = mTPData[problemSizemt == problem_size]
    plt.plot(subset['Number of processors'], subset['Avg elapsed time (s)'], marker='.')

plt.xlabel("Number of Processors")
plt.ylabel("Average Time to Solution (s)")  
plt.title("Average Elapsed Time vs Number of Processors")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()

# Choose a fixed problem size for strong scaling
problem_size_to_analyze1 = 200  # Change this to whichever size you want
problem_size_to_analyze2 = 3000  # Change this to whichever size you want
problem_size_to_analyze3 = 7800  # Change this to whichever size you want

subset1 = mTPData[problemSizemt == problem_size_to_analyze1]
subset2 = mTPData[problemSizemt == problem_size_to_analyze2]
subset3 = mTPData[problemSizemt == problem_size_to_analyze3]

# Sort by number of processors
subset1 = subset1.sort_values(by='Number of processors')
subset2 = subset2.sort_values(by='Number of processors')
subset3 = subset3.sort_values(by='Number of processors')

# Compute speedup and efficiency
T1 = subset1[subset1['Number of processors'] == 1]['Avg elapsed time (s)'].values[0]
subset1['Speedup'] = T1 / subset1['Avg elapsed time (s)']
subset1['Efficiency'] = subset1['Speedup'] / subset1['Number of processors']

T2 = subset2[subset2['Number of processors'] == 1]['Avg elapsed time (s)'].values[0]
subset2['Speedup'] = T2 / subset2['Avg elapsed time (s)']
subset2['Efficiency'] = subset2['Speedup'] / subset2['Number of processors']

T3 = subset3[subset3['Number of processors'] == 1]['Avg elapsed time (s)'].values[0]
subset3['Speedup'] = T3 / subset3['Avg elapsed time (s)']
subset3['Efficiency'] = subset3['Speedup'] / subset3['Number of processors']

# === 3-in-1 Plot ===
fig, axes = plt.subplots(3, 3, figsize=(12, 7))

# Panel 1: Elapsed Time
axes[0][0].plot(subset1['Number of processors'], subset1['Avg elapsed time (s)'], marker='.')
axes[0][0].set_ylabel('Avg Elapsed Time (s)')
axes[0][0].grid(True)

# Panel 2: Speedup
axes[0][1].plot(subset1['Number of processors'], subset1['Speedup'], marker='.', label='Measured Speedup')
axes[0][1].plot(x, y, linestyle='--', color='gray', label='Ideal Speedup')
axes[0][1].set_ylabel('Speedup')
axes[0][1].legend()
axes[0][1].grid(True)

# Panel 3: Parallel Efficiency
axes[0][2].plot(subset1['Number of processors'], subset1['Efficiency'], marker='.')
axes[0][2].plot(x, y1, c = 'k', linestyle = '--', label = 'Ideal Efficiency')
axes[0][2].set_ylabel('Efficiency')
axes[0][2].grid(True)
axes[0][2].set_ylim(0, 1.1)  # Efficiency should be between 0 and 1

# Panel 1: Elapsed Time
axes[1][0].plot(subset1['Number of processors'], subset1['Avg elapsed time (s)'], marker='.')
axes[1][0].set_ylabel('Avg Elapsed Time (s)')
axes[1][0].grid(True)

# Panel 2: Speedup
axes[1][1].plot(subset1['Number of processors'], subset1['Speedup'], marker='.', label='Measured Speedup')
axes[1][1].plot(x, y, linestyle='--', color='gray', label='Ideal Speedup')
axes[1][1].set_ylabel('Speedup')
axes[1][1].legend()
axes[1][1].grid(True)

# Panel 3: Parallel Efficiency
axes[1][2].plot(subset1['Number of processors'], subset1['Efficiency'], marker='.')
axes[1][2].plot(x, y1, c = 'k', linestyle = '--', label = 'Ideal Efficiency')
axes[1][2].set_ylabel('Efficiency')
axes[1][2].grid(True)
axes[1][2].set_ylim(0, 1.1)  # Efficiency should be between 0 and 1

# Panel 1: Elapsed Time
axes[2][0].plot(subset1['Number of processors'], subset1['Avg elapsed time (s)'], marker='.')
axes[2][0].set_xlabel('Number of Processors')
axes[2][0].set_ylabel('Avg Elapsed Time (s)')
axes[2][0].grid(True)

# Panel 2: Speedup
axes[2][1].plot(subset1['Number of processors'], subset1['Speedup'], marker='.', label='Measured Speedup')
axes[2][1].plot(x, y, linestyle='--', color='gray', label='Ideal Speedup')
axes[2][1].set_xlabel('Number of Processors')
axes[2][1].set_ylabel('Speedup')
axes[2][1].legend()
axes[2][1].grid(True)

# Panel 3: Parallel Efficiency
axes[2][2].plot(subset1['Number of processors'], subset1['Efficiency'], marker='.')
axes[2][2].plot(x, y1, c = 'k', linestyle = '--', label = 'Ideal Efficiency')
axes[2][2].set_xlabel('Number of Processors')
axes[2][2].set_ylabel('Efficiency')
axes[2][2].grid(True)
axes[2][2].set_ylim(0, 1.1)  # Efficiency should be between 0 and 1

# Final Layout
plt.tight_layout()
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
problemSizems = mSPData['Problem size']
numberOfProcessorsms = mSPData['Number of processors']
avgComputeFluxesms = mSPData['Avg compute fluxes time (s)']
avgComputeVariablesms = mSPData['Avg compute variables time (s)']
avgUpdateVariablesms = mSPData['Avg update variables time (s)']
avgApplyBoundaryConditionsms = mSPData['Avg apply boundary conditions time (s)']
avgDataTransferms = mSPData['Avg data transfer time (s)']

number_of_processors_to_analyzems = 48  # Change this to whichever size you want

subset4 = mSPData[numberOfProcessorsms == number_of_processors_to_analyzems]

# Sort by problem size
subset4 = subset4.sort_values(by='Problem size') 

problemSizems4 = subset4['Problem size']
numberOfProcessorsms4 = subset4['Number of processors']
avgComputeFluxesms4 = subset4['Avg compute fluxes time (s)']
avgComputeVariablesms4 = subset4['Avg compute variables time (s)']
avgUpdateVariablesms4 = subset4['Avg update variables time (s)']
avgApplyBoundaryConditionsms4 = subset4['Avg apply boundary conditions time (s)']
avgDataTransferms4 = subset4['Avg data transfer time (s)']

# Fit the model to the data
poptms41, pcovms41 = curve_fit(fitFunc2, problemSizems4, avgComputeFluxesms4)
ams41, bms41, cms41 = poptms41[0], poptms41[1], poptms41[2]

# Predict y values
y_predms41 = fitFunc2(problemSizems4, ams41, bms41, cms41)

# Compute R²
ss_resms41 = np.sum((avgComputeFluxesms4 - y_predms41)**2)
ss_totms41 = np.sum((avgComputeFluxesms4 - np.mean(avgComputeFluxesms4))**2)
r_squaredms41 = 1 - (ss_resms41 / ss_totms41)

print(f"R\u00B2: {r_squaredms41:.4f}")

# Fit the model to the data
poptms42, pcovms42 = curve_fit(fitFunc2, problemSizems4, avgComputeVariablesms4)
ams42, bms42, cms42 = poptms42[0], poptms42[1], poptms42[2]

# Predict y values
y_predms42 = fitFunc2(problemSizems4, ams42, bms42, cms42)

# Compute R²
ss_resms42 = np.sum((avgComputeVariablesms4 - y_predms42)**2)
ss_totms42 = np.sum((avgComputeVariablesms4 - np.mean(avgComputeVariablesms4))**2)
r_squaredms42 = 1 - (ss_resms42 / ss_totms42)

print(f"R\u00B2: {r_squaredms42:.4f}")

# Fit the model to the data
poptms43, pcovms43 = curve_fit(fitFunc2, problemSizems4, avgUpdateVariablesms4)
ams43, bms43, cms43 = poptms43[0], poptms43[1], poptms43[2]

# Predict y values
y_predms43 = fitFunc2(problemSizems4, ams43, bms43, cms43)

# Compute R²
ss_resms43 = np.sum((avgUpdateVariablesms4 - y_predms43)**2)
ss_totms43 = np.sum((avgUpdateVariablesms4 - np.mean(avgUpdateVariablesms4))**2)
r_squaredms43 = 1 - (ss_resms43 / ss_totms43)

print(f"R\u00B2: {r_squaredms43:.4f}")

# Fit the model to the data
poptms44, pcovms44 = curve_fit(fitFunc2, problemSizems4, avgApplyBoundaryConditionsms4)
ams44, bms44, cms44 = poptms44[0], poptms44[1], poptms44[2]

# Predict y values
y_predms44 = fitFunc2(problemSizems4, ams44, bms44, cms44)

# Compute R²
ss_resms44 = np.sum((avgApplyBoundaryConditionsms4 - y_predms44)**2)
ss_totms44 = np.sum((avgApplyBoundaryConditionsms4 - np.mean(avgApplyBoundaryConditionsms4))**2)
r_squaredms44 = 1 - (ss_resms44 / ss_totms44)

print(f"R\u00B2: {r_squaredms44:.4f}")

plt.figure(figsize=(12,7))
plt.plot(problemSizems4, y_predms41 * 1000, color = 'purple', label= f'Compute Fluxes Curve Fit, a = {ass1:.4e}, b = {bss1:.4e}, c = {css1:.4e}')
plt.plot(problemSizems4, y_predms42 * 1000, color = 'green', label= f'Compute Variables Curve Fit, a = {ass2:.4e}, b = {bss2:.4e}, c = {css2:.4e}')
plt.plot(problemSizems4, y_predms43 * 1000, color = 'blue', label= f'Update Variables Curve Fit, a = {ass3:.4e}, b = {bss3:.4e}, c = {css3:.4e}')
plt.plot(problemSizems4, y_predms44 * 1000, color = 'pink', label= f'Apply Boundary Conditions Curve Fit, a = {ass4:.4e}, b = {bss4:.4e}, c = {css4:.4e}')
plt.scatter(problemSizems4, avgComputeFluxesms4 * 1000, color = 'orange', marker = '.', label = 'Compute Fluxes')
plt.scatter(problemSizems4, avgComputeVariablesms4 * 1000, color = 'grey', marker = '.', label = 'Compute Variables')
plt.scatter(problemSizems4, avgUpdateVariablesms4 * 1000, color = 'brown', marker = '.', label = 'Update Variables')
plt.scatter(problemSizems4, avgApplyBoundaryConditionsms4 * 1000, color = 'red', marker = '.', label = 'Apply Boundary Conditions')
plt.xlabel("Problem Size")
plt.ylabel("Average Section Execution Time (ms)")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------- CUDA Kernel to serial SECTION performance comparison --------------------------------------------------------------
# Find the minimum length
minLen1 = min(len(avgComputeFluxesss), len(avgComputeFluxesck))
minLen2 = min(len(avgComputeVariablesss), len(avgComputeVariablesck))
minLen3 = min(len(avgUpdateVariablesss), len(avgUpdateVariablesck))
minLen4 = min(len(avgApplyBoundaryConditionsss), len(avgApplyBoundaryConditionsck))
minLen5 = min(len(problemSizess), len(problemSizeck))

# Truncate DataFrames
avgComputeFluxesss = avgComputeFluxesss.iloc[:minLen1] * 1000 
avgComputeVariablesss = avgComputeVariablesss.iloc[:minLen2] * 1000 
avgUpdateVariablesss = avgUpdateVariablesss.iloc[:minLen3] * 1000 
avgApplyBoundaryConditionsss = avgApplyBoundaryConditionsss.iloc[:minLen4] * 1000 
problemSizess = problemSizess.iloc[:minLen5] * 1000 

avgComputeFluxesck = avgComputeFluxesck.iloc[:minLen1] * 1000 
avgComputeVariablesck = avgComputeVariablesck.iloc[:minLen2] * 1000 
avgUpdateVariablesck = avgUpdateVariablesck.iloc[:minLen3] * 1000 
avgApplyBoundaryConditionsck = avgApplyBoundaryConditionsck.iloc[:minLen4] * 1000 
problemSizeck = problemSizeck.iloc[:minLen5] * 1000   

# Kernel to serial section comparison
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# ---- Subplot 1 ----
ax = axs[0, 0]
ax.scatter(problemSizess, avgComputeFluxesss, color='red', marker='*', label='Serial Compute Fluxes')
ax.scatter(problemSizeck, avgComputeFluxesck, color='black', marker='.', label='CUDA Compute Fluxes')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

# ---- Subplot 2 ----
ax = axs[0, 1]
ax.scatter(problemSizess, avgComputeVariablesss, color='violet', marker='*', label='Serial Compute Variables')
ax.scatter(problemSizeck, avgComputeVariablesck, color='orange', marker='.', label='CUDA Compute Variables')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

# ---- Subplot 3 ----
ax = axs[1, 0]
ax.scatter(problemSizess, avgUpdateVariablesss, color='pink', marker='*', label='Serial Update Variables')
ax.scatter(problemSizeck, avgUpdateVariablesck, color='blue', marker='.', label='CUDA Update Variables')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

# ---- Subplot 4 ----
ax = axs[1, 1]
ax.scatter(problemSizess, avgApplyBoundaryConditionsss, color='grey', marker='*', label='Serial Apply Boundary Conditions')
ax.scatter(problemSizeck, avgApplyBoundaryConditionsck, color='brown', marker='.', label='CUDA Apply Boundary Conditions')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size: (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------- CUDA Total to Serial Total performance comparison --------------------------------------------------------------
#Find Minimum Length
minLen6 = min(len(avgParallelElapsedTimect), len(avgSerialElapsedTime))
minLen7 = min(len(problemSizest), len(problemSizect))

# Truncate dataframes
avgSerialElapsedTime1 = avgSerialElapsedTime.iloc[:minLen6]
problemSizest1 = problemSizect.iloc[:minLen7]

avgParallelElapsedTimect = avgParallelElapsedTimect.iloc[:minLen6]
problemSizect = problemSizect.iloc[:minLen7]

# Speedup Comparison
speedupsc = avgSerialElapsedTime1 / avgParallelElapsedTimect
plt.figure(figsize=(10,6))
plt.scatter(problemSizest1, speedupsc, color = 'blue', marker = '.')    
plt.xlabel("Problem Size (N)")
plt.ylabel("Average Speedup") 
plt.show() 

# Serial & Parallel Total Runtime Comparison
fig, axs = plt.subplots(figsize=(10, 6))

# Parallel to serial comparison
ax = axs
ax.scatter(problemSizest1, avgSerialElapsedTime1, color = 'brown', marker = '*', label = 'Average Serial Elapsed Time')
ax.scatter(problemSizect, avgParallelElapsedTimect, color = 'blue', marker = '.', label = 'Average CUDA Elapsed Time')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (s)")
ax.legend()
plt.show()

# -------------------------------------------------------------- MPI Section to Serial Section performance comparison --------------------------------------------------------------
number_of_processors_to_analyzems = 48  # Change this to whichever size you want

subset5 = mSPData[numberOfProcessorsms == number_of_processors_to_analyzems]

# Sort by problem size
subset5 = subset5.sort_values(by='Problem size') 

problemSizems5 = subset5['Problem size']
numberOfProcessorsms5 = subset5['Number of processors']
avgComputeFluxesms5 = subset5['Avg compute fluxes time (s)']
avgComputeVariablesms5 = subset5['Avg compute variables time (s)']
avgUpdateVariablesms5 = subset5['Avg update variables time (s)']
avgApplyBoundaryConditionsms5 = subset5['Avg apply boundary conditions time (s)']
avgDataTransferms5 = subset5['Avg data transfer time (s)']

# Find the minimum length
minLen8 = min(len(avgComputeFluxesss), len(avgComputeFluxesms5))
minLen9 = min(len(avgComputeVariablesss), len(avgComputeVariablesms5))
minLen10 = min(len(avgUpdateVariablesss), len(avgUpdateVariablesms5))
minLen11 = min(len(avgApplyBoundaryConditionsss), len(avgApplyBoundaryConditionsms5))
minLen12 = min(len(problemSizess), len(problemSizems5))

# Truncate DataFrames
avgComputeFluxesss = avgComputeFluxesss.iloc[:minLen8] * 1000 
avgComputeVariablesss = avgComputeVariablesss.iloc[:minLen9] * 1000 
avgUpdateVariablesss = avgUpdateVariablesss.iloc[:minLen10] * 1000 
avgApplyBoundaryConditionsss = avgApplyBoundaryConditionsss.iloc[:minLen11] * 1000 
problemSizess = problemSizess.iloc[:minLen12] * 1000 

avgComputeFluxesms5 = avgComputeFluxesms5.iloc[:minLen8] * 1000 
avgComputeVariablesms5 = avgComputeVariablesms5.iloc[:minLen9] * 1000 
avgUpdateVariablesms5 = avgUpdateVariablesms5.iloc[:minLen10] * 1000 
avgApplyBoundaryConditionsms5 = avgApplyBoundaryConditionsms5.iloc[:minLen11] * 1000 
problemSizems5 = problemSizems5.iloc[:minLen12] * 1000   

# Kernel to serial section comparison
fig, axs = plt.subplots(2, 2, figsize=(12, 7))

# ---- Subplot 1 ----
ax = axs[0, 0]
ax.scatter(problemSizess, avgComputeFluxesss, color='red', marker='*', label='Serial Compute Fluxes')
ax.scatter(problemSizems5, avgComputeFluxesms5, color='black', marker='.', label='MPI Compute Fluxes')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

# ---- Subplot 2 ----
ax = axs[0, 1]
ax.scatter(problemSizess, avgComputeVariablesss, color='violet', marker='*', label='Serial Compute Variables')
ax.scatter(problemSizems5, avgComputeVariablesms5, color='orange', marker='.', label='MPI Compute Variables')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

# ---- Subplot 3 ----
ax = axs[1, 0]
ax.scatter(problemSizess, avgUpdateVariablesss, color='pink', marker='*', label='Serial Update Variables')
ax.scatter(problemSizems5, avgUpdateVariablesms5, color='blue', marker='.', label='MPI Update Variables')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

# ---- Subplot 4 ----
ax = axs[1, 1]
ax.scatter(problemSizess, avgApplyBoundaryConditionsss, color='grey', marker='*', label='Serial Apply Boundary Conditions')
ax.scatter(problemSizems5, avgApplyBoundaryConditionsms5, color='brown', marker='.', label='MPI Apply Boundary Conditions')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size: (N)")
ax.set_ylabel("Average Time to Solution (ms)")
ax.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------- MPI Total to Serial Total performance comparison --------------------------------------------------------------
number_of_processors_to_analyzemt = 48  # Change this to whichever size you want

subset6 = mTPData[numberOfProcessorsmt == number_of_processors_to_analyzemt]

# Sort by problem size
subset6 = subset6.sort_values(by='Problem size') 

problemSizemt6 = subset6['Problem size']
avgParallelElapsedTimemt6 = subset6['Avg elapsed time (s)']

#Find Minimum Length
minLen13 = min(len(avgParallelElapsedTimemt6), len(avgSerialElapsedTime))
minLen14 = min(len(problemSizemt6), len(problemSizest))

# Truncate dataframes
avgSerialElapsedTime2 = avgSerialElapsedTime.iloc[:minLen13]
problemSizest2 = problemSizest.iloc[:minLen14]

avgParallelElapsedTimemt6 = avgParallelElapsedTimemt6.iloc[:minLen13]
problemSizemt6 = problemSizemt6.iloc[:minLen14]

speedupsm = avgSerialElapsedTime2/avgParallelElapsedTimemt6
# Speedup Comparison
plt.figure(figsize=(10,6))
plt.scatter(problemSizest2, speedupsm, color = 'blue', marker = '.')    
plt.xlabel("Problem Size (N)")
plt.ylabel("Average Speedup") 
plt.show() 

# Serial & Parallel Total Runtime Comparison
fig, axs = plt.subplots(figsize=(10, 6))

# Parallel to serial comparison
ax = axs
ax.scatter(problemSizest2, avgSerialElapsedTime2, color = 'brown', marker = '*', label = 'Average Serial Elapsed Time')
ax.scatter(problemSizemt6, avgParallelElapsedTimemt6, color = 'blue', marker = '.', label = 'Average MPI Elapsed Time')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Average Time to Solution (s)")
ax.legend()
plt.show()

# -------------------------------------------------------------- CUDA Total to MPI Total performance comparison --------------------------------------------------------------
number_of_processors_to_analyze = 48  # Change this to whichever size you want

subset7 = mTPData[numberOfProcessorsmt == number_of_processors_to_analyze]

# Sort by problem size
subset7 = subset7.sort_values(by='Problem size') 
subset7['Avg power consumption'] = 6.25 * subset7['Number of processors'] * subset7['Avg elapsed time (s)']
problemSizemt7 = subset7['Problem size']
avgParallelElapsedTimemt7 = subset7['Avg elapsed time (s)']
avgPowerConsumptionmt = subset7['Avg power consumption']

#Find Minimum Length
minLen15 = min(len(avgParallelElapsedTimect), len(avgParallelElapsedTimemt7))
minLen16 = min(len(problemSizect), len(problemSizemt7))

# Truncate dataframes
avgParallelElapsedTimect = avgParallelElapsedTimect.iloc[:minLen15]
problemSizect = problemSizect.iloc[:minLen16]

avgPowerConsumptionct = 200 * avgParallelElapsedTimect
pTPData[avgPowerConsumptionct] = avgPowerConsumptionct

avgParallelElapsedTimemt = avgParallelElapsedTimemt.iloc[:minLen15]
problemSizemt7 = problemSizemt7.iloc[:minLen16]

fig, axes = plt.subplots(1, 2, figsize=(12, 7))

axes[0].plot(problemSizemt7, avgParallelElapsedTimemt7, marker = '.', color = 'blue')
axes[0].plot(problemSizect, avgParallelElapsedTimect, marker = '.', color = 'red')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel("Problem Size (N)")
axes[0].set_ylabel("Average Time to Solution (s)")
axes[0].legend()

axes[1].plot(problemSizemt7, avgPowerConsumptionmt, marker = '.', color = 'blue')
axes[1].plot(problemSizect,avgPowerConsumptionct, marker = '.', color = 'red')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel("Problem Size (N)")
axes[1].set_ylabel("Power Consumption (J)")
axes[1].legend()

plt.tight_layout()
plt.show()