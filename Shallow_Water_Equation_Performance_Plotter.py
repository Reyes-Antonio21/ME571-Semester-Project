# Imports
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Fitting functions:
def fitFunc1(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def fitFunc2(x, a, b, c):
    return a * x ** 2 + b * x + c

# .CSV Data 
path ='C:/Users/Antonio Reyes/OneDrive/Documents/Cuda Projects/'
parallelKernelPerformance = 'Shallow_Water_Equations_Parallel_Kernel_Performance.csv'
parallelTotalPerformance = 'Shallow_Water_Equations_Parallel_Total_Performance.csv'
serialTotalPerformance = 'Shallow_Water_Equations_Serial_Total_Performance.csv'
serialSectionPerformance = 'Shallow_Water_Equations_Serial_Section_Performance.csv'

# parallel total performance 
pTPData = pd.read_csv(path + parallelTotalPerformance, header = 0, names = ['Problem Size:','Average Elapsed Time:','Average Host-Device Data Transfer Time:'])
problemSize = pTPData['Problem Size:']
avgParallelElapsedTime = pTPData['Average Elapsed Time:']
avgHostDeviceTransfer = pTPData['Average Host-Device Data Transfer Time:']

popt, pcov = curve_fit(fitFunc1, problemSize, avgParallelElapsedTime)
plt.figure(figsize=(10,6))
plt.plot(problemSize, fitFunc1(problemSize, *popt), color = 'pink', label = 'Elapsed Time Curve Fit')
plt.scatter(problemSize, avgParallelElapsedTime, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.title("Average Parallelized Elapsed Time per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

print(popt)

popt, pcov = curve_fit(fitFunc2, problemSize, avgHostDeviceTransfer)
plt.figure(figsize=(10,6))
plt.plot(problemSize, fitFunc2(problemSize, *popt) * 1000, color = 'green', label= 'Host-Device Data Transfer Curve Fit')
plt.scatter(problemSize, avgHostDeviceTransfer * 1000, color = 'blue', marker = '.', label = 'Average Host-Device Data Transfer Time')
plt.title("Average Host-Device Data Transfer Time per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Host-Device Data Transfer Time (ms)")
plt.legend()
plt.show()

# parallel kernel performance
pKPData = pd.read_csv(path + parallelKernelPerformance, header = 0, names = ['Problem Size:','Average Elapsed Time:','Average Compute Fluxes Time:','Average Compute Variables Time:','Average Update Variables Time:','Average Apply Boundary Conditions Time:'])
problemSize = pKPData['Problem Size:']
avgComputeFlux = pKPData['Average Compute Fluxes Time:']
avgComputeVariables = pKPData['Average Compute Variables Time:']
avgUpdateVariables = pKPData['Average Update Variables Time:']
avgApplyBoundaryConditions = pKPData['Average Apply Boundary Conditions Time:']

popt1, pcov1 = curve_fit(fitFunc2, problemSize, avgComputeFlux)
plt.figure(figsize=(10,6))
plt.plot(problemSize, fitFunc2(problemSize, *popt1) * 1000, color = 'purple', label= 'Compute Fluxes Curve Fit')
popt2, pcov2 = curve_fit(fitFunc2, problemSize, avgComputeVariables)
plt.plot(problemSize, fitFunc2(problemSize, *popt2) * 1000, color = 'green', label= 'Compute Variables Curve Fit')
popt3, pcov3 = curve_fit(fitFunc2, problemSize, avgUpdateVariables)
plt.plot(problemSize, fitFunc2(problemSize, *popt3) * 1000, color = 'blue', label= 'Update Variables Curve Fit')
popt4, pcov4 = curve_fit(fitFunc2, problemSize, avgApplyBoundaryConditions)
plt.plot(problemSize, fitFunc2(problemSize, *popt4) * 1000, color = 'pink', label= 'Apply Boundary Conditions Curve Fit')
plt.scatter(problemSize, avgComputeFlux * 1000, color = 'orange', marker = '.', label = 'Compute Fluxes')
plt.scatter(problemSize, avgComputeVariables * 1000, color = 'grey', marker = '.', label = 'Compute Variables')
plt.scatter(problemSize, avgUpdateVariables * 1000, color = 'brown', marker = '.', label = 'Update Variables')
plt.scatter(problemSize, avgApplyBoundaryConditions * 1000, color = 'red', marker = '.', label = 'Apply Boundary Conditions')
plt.title("Average Kernel Execution Times per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Average Kernel Execution Time (ms)")
plt.legend()
plt.show()

# serial total performance  
sTPData = pd.read_csv(path + serialTotalPerformance, header = 0, names = ['Problem Size:','Average Elapsed Time:'])
problemSize = sTPData['Problem Size:']
avgSerialElapsedTime = sTPData['Average Elapsed Time:']

popt, pcov = curve_fit(fitFunc1, problemSize, avgSerialElapsedTime)
plt.figure(figsize=(10,6))
plt.plot(problemSize, fitFunc1(problemSize, *popt), color = 'green', label= 'Elapsed Time Curve Fit')
plt.scatter(problemSize,avgSerialElapsedTime, color = 'blue', marker = '.', label = 'Average Elapsed Time')
plt.title("Average Serial Elapsed Time per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Average Time to Solution (s)")
plt.legend()
plt.show()

# serial section performance
sSPData = pd.read_csv(path + serialSectionPerformance, header = 0, names = ['Problem Size:','Average Compute Fluxes Time:','Average Compute Variables Time:','Average Update Variables Time:','Average Apply Boundary Conditions Time:']) 
problemSize = sSPData['Problem Size:']
avgComputeFlux = sSPData['Average Compute Fluxes Time:']
avgComputeVariables = sSPData['Average Compute Variables Time:']
avgUpdateVariables = sSPData['Average Update Variables Time:']
avgApplyBoundaryConditions = sSPData['Average Apply Boundary Conditions Time:']

popt1, pcov1 = curve_fit(fitFunc2, problemSize, avgComputeFlux)
plt.figure(figsize=(10,6))
plt.plot(problemSize, fitFunc2(problemSize, *popt1) * 1000, color = 'purple', label= 'Compute Fluxes Curve Fit')
popt2, pcov2 = curve_fit(fitFunc2, problemSize, avgComputeVariables)
plt.plot(problemSize, fitFunc2(problemSize, *popt2) * 1000, color = 'green', label= 'Compute Variables Curve Fit')
popt3, pcov3 = curve_fit(fitFunc2, problemSize, avgUpdateVariables)
plt.plot(problemSize, fitFunc2(problemSize, *popt3) * 1000, color = 'blue', label= 'Update Variables Curve Fit')
popt4, pcov4 = curve_fit(fitFunc2, problemSize, avgApplyBoundaryConditions)
plt.plot(problemSize, fitFunc2(problemSize, *popt4) * 1000, color = 'pink', label= 'Apply Boundary Conditions Curve Fit')
plt.scatter(problemSize, avgComputeFlux * 1000, color = 'orange', marker = '.', label = 'Compute Fluxes')
plt.scatter(problemSize, avgComputeVariables * 1000, color = 'grey', marker = '.', label = 'Compute Variables')
plt.scatter(problemSize, avgUpdateVariables * 1000, color = 'brown', marker = '.', label = 'Update Variables')
plt.scatter(problemSize, avgApplyBoundaryConditions * 1000, color = 'red', marker = '.', label = 'Apply Boundary Conditions')
plt.title("Average Serial Section Execution Times per Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Average Section Execution Time (ms)")
plt.legend()
plt.show()
