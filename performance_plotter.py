# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Fitting functions:
def fitFunc1(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def fitFunc2(x, a, b, c):
    return a * x ** 2 + b * x + c

# .CSV Data 
path ='C:/Users/anton/OneDrive/Documents/Cuda Projects/'
parallelKernelPerformance = 'Shallow_Water_Equations_Parallel_Kernel_Performance.csv'
parallelTotalPerformance = 'Shallow_Water_Equations_Parallel_Total_Performance.csv'
serialTotalPerformance = 'Shallow_Water_Equations_Serial_Total_Performance.csv'
serialSectionPerformance = 'Shallow_Water_Equations_Serial_Section_Performance.csv'

# parallel total performance 
pTPData = pd.read_csv(path + parallelTotalPerformance, header = 0, names = ['Problem Size:','Average Elapsed Time:','Average Host-Device Data Transfer Time:'])
problemSizept = pTPData['Problem Size:']
avgParallelElapsedTime = pTPData['Average Elapsed Time:']
avgHostDeviceTransfer = pTPData['Average Host-Device Data Transfer Time:']

poptpt, pcovpt = curve_fit(fitFunc1, problemSizept, avgParallelElapsedTime)
plt.figure(figsize=(10,6))
plt.plot(problemSizept, fitFunc1(problemSizept, *poptpt), color = 'pink', label = 'Elapsed Time Curve Fit')
plt.scatter(problemSizept, avgParallelElapsedTime, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.title("Average Parallelized Elapsed Time per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

poptdt, pcovdt = curve_fit(fitFunc2, problemSizept, avgHostDeviceTransfer)
plt.figure(figsize=(10,6))
plt.plot(problemSizept, fitFunc2(problemSizept, *poptdt) * 1000, color = 'green', label= 'Host-Device Data Transfer Curve Fit')
plt.scatter(problemSizept, avgHostDeviceTransfer * 1000, color = 'blue', marker = '.', label = 'Average Host-Device Data Transfer Time')
plt.title("Average Host-Device Data Transfer Time per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Host-Device Data Transfer Time (ms)")
plt.legend()
plt.show()

# parallel kernel performance
pKPData = pd.read_csv(path + parallelKernelPerformance, header = 0, names = ['Problem Size:','Average Elapsed Time:','Average Compute Fluxes Time:','Average Compute Variables Time:','Average Update Variables Time:','Average Apply Boundary Conditions Time:'])
problemSizepk = pKPData['Problem Size:']
avgComputeFluxpk = pKPData['Average Compute Fluxes Time:']
avgComputeVariablespk = pKPData['Average Compute Variables Time:']
avgUpdateVariablespk = pKPData['Average Update Variables Time:']
avgApplyBoundaryConditionspk = pKPData['Average Apply Boundary Conditions Time:']

poptpk1, pcovpk1 = curve_fit(fitFunc2, problemSizepk, avgComputeFluxpk)
plt.figure(figsize=(10,6))
plt.plot(problemSizepk, fitFunc2(problemSizepk, *poptpk1) * 1000, color = 'purple', label= 'Compute Fluxes Curve Fit')
poptpk2, pcovpk2 = curve_fit(fitFunc2, problemSizepk, avgComputeVariablespk)
plt.plot(problemSizepk, fitFunc2(problemSizepk, *poptpk2) * 1000, color = 'green', label= 'Compute Variables Curve Fit')
poptpk3, pcovpk3 = curve_fit(fitFunc2, problemSizepk, avgUpdateVariablespk)
plt.plot(problemSizepk, fitFunc2(problemSizepk, *poptpk3) * 1000, color = 'blue', label= 'Update Variables Curve Fit')
poptpk4, pcovpk4 = curve_fit(fitFunc2, problemSizepk, avgApplyBoundaryConditionspk)
plt.plot(problemSizepk, fitFunc2(problemSizepk, *poptpk4) * 1000, color = 'pink', label= 'Apply Boundary Conditions Curve Fit')
plt.scatter(problemSizepk, avgComputeFluxpk * 1000, color = 'orange', marker = '.', label = 'Compute Fluxes')
plt.scatter(problemSizepk, avgComputeVariablespk * 1000, color = 'grey', marker = '.', label = 'Compute Variables')
plt.scatter(problemSizepk, avgUpdateVariablespk * 1000, color = 'brown', marker = '.', label = 'Update Variables')
plt.scatter(problemSizepk, avgApplyBoundaryConditionspk * 1000, color = 'red', marker = '.', label = 'Apply Boundary Conditions')
plt.title("Average Kernel Execution Times per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Average Kernel Execution Time (ms)")
plt.legend()
plt.show()

# serial total performance  
sTPData = pd.read_csv(path + serialTotalPerformance, header = 0, names = ['Problem Size:','Average Elapsed Time:'])
problemSizest = sTPData['Problem Size:']
avgSerialElapsedTime = sTPData['Average Elapsed Time:']

poptst, pcovst = curve_fit(fitFunc1, problemSizest, avgSerialElapsedTime)
plt.figure(figsize=(10,6))
plt.plot(problemSizest, fitFunc1(problemSizest, *poptst), color = 'green', label= 'Elapsed Time Curve Fit')
plt.scatter(problemSizest,avgSerialElapsedTime, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.title("Average Serial Elapsed Time per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

# serial section performance
sSPData = pd.read_csv(path + serialSectionPerformance, header = 0, names = ['Problem Size:','Average Compute Fluxes Time:','Average Compute Variables Time:','Average Update Variables Time:','Average Apply Boundary Conditions Time:']) 
problemSizess = sSPData['Problem Size:']
avgComputeFluxss = sSPData['Average Compute Fluxes Time:']
avgComputeVariablesss = sSPData['Average Compute Variables Time:']
avgUpdateVariablesss = sSPData['Average Update Variables Time:']
avgApplyBoundaryConditionsss = sSPData['Average Apply Boundary Conditions Time:']

poptss1, pcovss1 = curve_fit(fitFunc2, problemSizess, avgComputeFluxss)
plt.figure(figsize=(10,6))
plt.plot(problemSizess, fitFunc2(problemSizess, *poptss1) * 1000, color = 'purple', label= 'Compute Fluxes Curve Fit')
poptss2, pcovss2 = curve_fit(fitFunc2, problemSizess, avgComputeVariablesss)
plt.plot(problemSizess, fitFunc2(problemSizess, *poptss2) * 1000, color = 'green', label= 'Compute Variables Curve Fit')
poptss3, pcovss3 = curve_fit(fitFunc2, problemSizess, avgUpdateVariablesss)
plt.plot(problemSizess, fitFunc2(problemSizess, *poptss3) * 1000, color = 'blue', label= 'Update Variables Curve Fit')
poptss4, pcovss4 = curve_fit(fitFunc2, problemSizess, avgApplyBoundaryConditionsss)
plt.plot(problemSizess, fitFunc2(problemSizess, *poptss4) * 1000, color = 'pink', label= 'Apply Boundary Conditions Curve Fit')
plt.scatter(problemSizess, avgComputeFluxss * 1000, color = 'orange', marker = '.', label = 'Compute Fluxes')
plt.scatter(problemSizess, avgComputeVariablesss * 1000, color = 'grey', marker = '.', label = 'Compute Variables')
plt.scatter(problemSizess, avgUpdateVariablesss * 1000, color = 'brown', marker = '.', label = 'Update Variables')
plt.scatter(problemSizess, avgApplyBoundaryConditionsss * 1000, color = 'red', marker = '.', label = 'Apply Boundary Conditions')
plt.title("Average Serial Section Execution Times per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Average Section Execution Time (ms)")
plt.legend()
plt.show()

# Data truncation for comparison
# Find the minimum length
min_len1 = min(len(avgComputeFluxss), len(avgComputeFluxpk))
min_len2 = min(len(avgComputeVariablesss), len(avgComputeVariablespk))
min_len3 = min(len(avgUpdateVariablesss), len(avgUpdateVariablespk))
min_len4 = min(len(avgApplyBoundaryConditionsss), len(avgApplyBoundaryConditionspk))
min_len5 = min(len(problemSizess), len(problemSizepk))

# Truncate both DataFrames
avgComputeFluxss = avgComputeFluxss.iloc[:min_len1]
avgComputeVariablesss = avgComputeVariablesss.iloc[:min_len2]
avgUpdateVariablesss = avgUpdateVariablesss.iloc[:min_len3]
avgApplyBoundaryConditionsss = avgApplyBoundaryConditionsss.iloc[:min_len4]
problemSizess = problemSizess.iloc[:min_len5]

avgComputeFluxpk = avgComputeFluxpk.iloc[:min_len1]
avgComputeVariablespk = avgComputeVariablespk.iloc[:min_len2]
avgUpdateVariablespk = avgUpdateVariablespk.iloc[:min_len3]
avgApplyBoundaryConditionspk = avgApplyBoundaryConditionspk.iloc[:min_len4]
problemSizepk = problemSizepk.iloc[:min_len5]

# Transform using logarithmic scale
avgComputeFluxss = np.log10(avgComputeFluxss.replace(0, np.nan))
avgComputeVariablesss = np.log10(avgComputeVariablesss.replace(0, np.nan))
avgUpdateVariablesss = np.log10(avgUpdateVariablesss.replace(0, np.nan))
avgApplyBoundaryConditionsss = np.log10(avgApplyBoundaryConditionsss.replace(0, np.nan))
problemSizess = np.log10(problemSizess.replace(0, np.nan))

avgComputeFluxpk = np.log10(avgComputeFluxpk.replace(0, np.nan))
avgComputeVariablespk = np.log10(avgComputeVariablespk.replace(0, np.nan))
avgUpdateVariablespk = np.log10(avgUpdateVariablespk.replace(0, np.nan))
avgApplyBoundaryConditionspk = np.log10(avgApplyBoundaryConditionspk.replace(0, np.nan))
problemSizepk = np.log10(problemSizepk.replace(0, np.nan))

# kernel to serial section comparison
plt.figure(figsize=(10,6))
plt.scatter(problemSizess, avgComputeFluxss, color = 'red', marker = '*', label = 'Serial Compute Fluxes')
plt.scatter(problemSizess, avgComputeVariablesss, color = 'violet', marker = '*', label = 'Serial Compute Variables')
plt.scatter(problemSizess, avgUpdateVariablesss, color = 'green', marker = '*', label = 'Serial Update Variables')
plt.scatter(problemSizess, avgApplyBoundaryConditionsss, color = 'grey', marker = '*', label = 'Serial Apply Boundary Conditions')
plt.scatter(problemSizepk, avgComputeFluxpk, color = 'black', marker = '.', label = 'Kernel Compute Fluxes')
plt.scatter(problemSizepk, avgComputeVariablespk, color = 'orange', marker = '.', label = 'Kernel Compute Variables')
plt.scatter(problemSizepk, avgUpdateVariablespk, color = 'blue', marker = '.', label = 'Kernel Update Variables')
plt.scatter(problemSizepk, avgApplyBoundaryConditionspk, color = 'brown', marker = '.', label = 'Kernel Apply Boundary Conditions')
plt.xlabel("Problem Size")
plt.ylabel("Average Section Execution Time (ms)")
plt.legend()
plt.show()