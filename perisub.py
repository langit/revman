from random import * 

def doTIS(S, X):
    X[0] = S - X[1] - X[2]
    if X[0] > 0: X[2] += X[0]

def doNIS(S, X):
    X[0] = S - X[2]
    if X[0] > 0: X[2] += X[0]

def doAge(X):
    X[1] = X[2]
    X[2] = 0

def discUnif(D):
    D[1] = randint(0,25)
    D[2] = randint(0,25)

class PerishSubs:
    def __init__(me, pi, h, p, m, a):
        me.pi = (0,)+pi #pi[1] = piD, pi[2]=piU
        me.h  = h
        me.p = (0,)+p #p[1], p[2]
        me.m = m
        me.a = (0,)+a #a[1] = aD, a[2]=aU 

    def doServe(me, X, D): #serve demand
        for i in range(1,3):
            X[i] -= D[i] # X has extra stock
            D[i] = 0.0 #D has extra demand
            if X[i] < 0.0:
                D[i] = -X[i]
                X[i] = 0.0
        sub = [0.,0.,0.] #substitution as recourse
        for i in range(1,3):
            if D[i] > 0:
                sub[i] = me.pi[i]*D[i]
                if sub[i] > X[3-i]: sub[i] = X[3-i]
                X[3-i] -= sub[i]
                D[i] -= sub[i]
        return sub #lost Demand in D, unsold Inventory in X

    def findCost(me, sub, X, D): #lost Demand in D, unsold Inventory in X
        cost = 0.0 #found by formula (4)
        cost += me.h * X[2] #holding cost
        cost += me.m * X[1] #salvage cost
        cost += me.a[1] * sub[1]
        cost += me.a[2] * sub[2]
        cost += me.p[1] * D[1]
        cost += me.p[2] * D[2]
        return cost

    def simulate(me, S, old, T, warm, fr, rd): #fr, rd: are functions!
        X = [0., old, 0.]
        D = [0.,0.,0.]
        for i in range(warm):
            fr(S,X) #replenish inv.
            rd(D) #generate Demand
            print X, D,
            sub = me.doServe(X,D)
            print X, D, sub
            doAge(X)

        cost = 0.0
        for i in range(T):
            fr(S,X) #replenish inv.
            rd(D) #generate Demand
            sub = me.doServe(X,D)
            cost += me.findCost(sub,X,D)
            doAge(X)
        return cost/T

bestS = -1
bestC = 1e9
for S in range(10, 30):
    ps = PerishSubs((1.,1.), 1., (1., 4.), 2., (0., 3.))
    C = ps.simulate(S, S/2, 1000, 10, doTIS, discUnif)
    if C<bestC:
        bestC = C
        bestS = S
        print S, C
