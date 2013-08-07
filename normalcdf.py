import math

def phi(x, mu=0.0, sigma=1.0):
    #note: if sigma<0: get survival prob
    x = (x - mu)/float(sigma)
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)/math.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)

    return 0.5*(1.0 + sign*y)

from ltqnorm import ltqnorm as phinv
if __name__ == "__main__":
	print 0, phi(0), phinv(phi(0))
	for x in range(-6,2):
		y = 10.**x
		p = phi(y)
		print y, p, phinv(p) if p<1. else "INF"
