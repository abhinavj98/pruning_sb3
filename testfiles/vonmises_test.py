from scipy.stats import vonmises

sample_size = 100
loc = 0
kappa = 0.7
sample = vonmises(loc=loc, kappa=kappa).rvs(sample_size)

#Plot the vonmises distribution
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 100)
y = vonmises.pdf(x, loc=loc, kappa=kappa)
plt.plot(x, y, 'r-', lw=2)
plt.hist(sample, bins=50, density=True)

#Sample from the vonmises distribution
plt.title('Vonmises Distribution')
plt.xlabel('Angle')
plt.ylabel('Density')



plt.show()

#Compute probabillity of sample being between -pi/2 and pi/2 using cdf
prob = vonmises.cdf(np.pi/2, loc=loc, kappa=kappa) - vonmises.cdf(-np.pi/2, loc=loc, kappa=kappa)
print(prob)
