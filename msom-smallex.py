

def findW(prof, x):
    m1 = min(prof[1], x[1])
    m0 = m1 + prof[0]
    m0 = min(m0, x[0]+x[1]) - m1
    return m0, m1

def revof(prof, f, bet, p, V, n):
    vc = n/(1-p)
    b0 = min(vc, prof[0])
    b1 = min(vc, b0+prof[1]) - b0
    return (b0*f[0]+b1*f[1])*(1 - p + p*bet)

def revon(w, f, bet, p, V, n):
    r = w[0]*f[0]+w[1]*f[1]
    r *= (1 - p + p*bet)
    r -= V * max(0, (1-p)*(w[0]+w[1]) - n)
    return r

def excal():
    f = (200.,100.)
    V = 300.
    n = 8
    beta = 0.2
    x = (5,5)

    plist = (0.1, 0.15, 0.2)
    profs = ((4,7), (5,7), (6,7))

    for prof in profs:
        for p in plist:
            w = findW(prof, x)
            ron = revon(w, f, beta, p, V, n)
            rof = revof(prof, f, beta, p, V, n)
            print prof, p, ":", w, ron, rof, ron/rof

excal()
