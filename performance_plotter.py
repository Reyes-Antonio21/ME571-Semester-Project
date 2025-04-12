# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import LogFormatter, LogLocator

# -------------------------------------------------------------- Fitting functions --------------------------------------------------------------
def fitFunc1(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def fitFunc2(x, a, b, c):
    return a * x ** 2 + b * x + c

def fitFunc3(x, a, b):
    return a * x + b

def fitFunc4(x, d):
    return d *(67.0 * x ** 3 + 104.0 * x ** 2 + 77.0 * x)

def fitFunc5(x, a, b):
    return a * np.exp^(b * x)

# -------------------------------------------------------------- CSV Data -------------------------------------------------------------- 
# Path to the CSV files
# Note: The path should be updated to the location of your CSV files
path ='C:/Users/Antonio Reyes/OneDrive/Documents/Cuda Projects/Shallow_Water_Equations_Averaged_csv_Files/'
parallelKernelPerformance = 'Shallow_Water_Equations_Parallel_Kernel_Runtime_Performance.csv'
parallelTotalPerformance = 'Shallow_Water_Equations_Parallel_Total_Runtime_Performance.csv'
serialTotalPerformance = 'Shallow_Water_Equations_Serial_Total_Runtime_Performance.csv'
serialSectionPerformance = 'Shallow_Water_Equations_Serial_Section_Runtime_Performance.csv'

# -------------------------------------------------------------- parallel total performance --------------------------------------------------------------
# Read the CSV file
pTPData = pd.read_csv(path + parallelTotalPerformance, header = 0, names = ['Problem Size:','Average Elapsed Time:','Average Host-Device Data Transfer Time:'])
problemSizept = pTPData['Problem Size:']
avgParallelElapsedTime = pTPData['Average Elapsed Time:']
avgHostDeviceTransfer = pTPData['Average Host-Device Data Transfer Time:']

# Fit the model to the data
poptpt, pcovpt = curve_fit(fitFunc1, problemSizept, avgParallelElapsedTime)
apt, bpt, cpt, dpt = poptpt[0], poptpt[1], poptpt[2], poptpt[3]

# Predict y values
y_predpt = fitFunc1(problemSizept, apt, bpt, cpt, dpt)

# Compute R²
ss_respt = np.sum((avgParallelElapsedTime - y_predpt)**2)
ss_totpt = np.sum((avgParallelElapsedTime - np.mean(avgParallelElapsedTime))**2)
r_squaredpt = 1 - (ss_respt / ss_totpt)

print(f"R\u00B2: {r_squaredpt:.4f}")

plt.figure(figsize=(10,6))
plt.plot(problemSizept, y_predpt, color = 'pink', label = f'Elapsed Time Curve Fit, a = {apt:.4e}, b = {bpt:.4e}, c = {cpt:.4e}, d = {dpt:.4e}')
plt.scatter(problemSizept, avgParallelElapsedTime, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

# Fit the model to the data
poptdt, pcovdt = curve_fit(fitFunc2, problemSizept, avgHostDeviceTransfer)
adt, bdt, cdt = poptdt[0], poptdt[1], poptdt[2]

# Predict y values
y_preddt = fitFunc2(problemSizept, adt, bdt, cdt)

# Compute R\u00B2
ss_resdt = np.sum((avgHostDeviceTransfer - y_preddt)**2)
ss_totdt = np.sum((avgHostDeviceTransfer - np.mean(avgHostDeviceTransfer))**2)
r_squareddt = 1 - (ss_resdt / ss_totdt)

print(f"R\u00B2: {r_squareddt:.4f}")

plt.figure(figsize=(10,6))
plt.plot(problemSizept, y_preddt * 1000, color = 'green', label= f'Host-Device Data Transfer Curve Fit, a = {adt:.4e}, b = {bdt:.4e}, c = {cdt:.4e}')
plt.scatter(problemSizept, avgHostDeviceTransfer * 1000, color = 'blue', marker = '.', label = 'Average Host-Device Data Transfer Time')
plt.xlabel("Problem Size")
plt.ylabel("Host-Device Data Transfer Time (ms)")
plt.legend()
plt.show()

# -------------------------------------------------------------- parallel kernel performance --------------------------------------------------------------
# Read the CSV file
pKPData = pd.read_csv(path + parallelKernelPerformance, header = 0, names = ['Problem Size:','Average Elapsed Time:','Average Compute Fluxes Time:','Average Compute Variables Time:','Average Update Variables Time:','Average Apply Boundary Conditions Time:'])
problemSizepk = pKPData['Problem Size:']
avgComputeFluxpk = pKPData['Average Compute Fluxes Time:']
avgComputeVariablespk = pKPData['Average Compute Variables Time:']
avgUpdateVariablespk = pKPData['Average Update Variables Time:']
avgApplyBoundaryConditionspk = pKPData['Average Apply Boundary Conditions Time:']

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
sTPData = pd.read_csv(path + serialTotalPerformance, header = 0, names = ['Problem Size:','Average Elapsed Time:'])
problemSizest = sTPData['Problem Size:']
avgSerialElapsedTime = sTPData['Average Elapsed Time:']

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
sSPData = pd.read_csv(path + serialSectionPerformance, header = 0, names = ['Problem Size:', 'Average Elapsed Time', 'Average Compute Fluxes Time:','Average Compute Variables Time:','Average Update Variables Time:','Average Apply Boundary Conditions Time:']) 
problemSizess = sSPData['Problem Size:']
avgComputeFluxss = sSPData['Average Compute Fluxes Time:']
avgComputeVariablesss = sSPData['Average Compute Variables Time:']
avgUpdateVariablesss = sSPData['Average Update Variables Time:']
avgApplyBoundaryConditionsss = sSPData['Average Apply Boundary Conditions Time:']

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
ax.set_ylabel("Avg Time to Solution (ms)")
ax.legend()

# ---- Subplot 2 ----
ax = axs[0, 1]
ax.scatter(problemSizess, avgComputeVariablesss, color='violet', marker='*', label='Serial Compute Variables')
ax.scatter(problemSizepk, avgComputeVariablespk, color='orange', marker='.', label='Kernel Compute Variables')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Avg Time to Solution (ms)")
ax.legend()

# ---- Subplot 3 ----
ax = axs[1, 0]
ax.scatter(problemSizess, avgUpdateVariablesss, color='pink', marker='*', label='Serial Update Variables')
ax.scatter(problemSizepk, avgUpdateVariablespk, color='blue', marker='.', label='Kernel Update Variables')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size (N)")
ax.set_ylabel("Avg Time to Solution (ms)")
ax.legend()

# ---- Subplot 4 ----
ax = axs[1, 1]
ax.scatter(problemSizess, avgApplyBoundaryConditionsss, color='grey', marker='*', label='Serial Apply Boundary Conditions')
ax.scatter(problemSizepk, avgApplyBoundaryConditionspk, color='brown', marker='.', label='Kernel Apply Boundary Conditions')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Problem Size: (N)")
ax.set_ylabel("Avg Time to Solution (ms)")
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
plt.xlabel("Problem Size")
plt.ylabel("Speedup") 
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
ax.set_xlabel("Problem Size: (N)")
ax.set_ylabel("Avg Time to Solution (s)")
ax.legend()
plt.show()
