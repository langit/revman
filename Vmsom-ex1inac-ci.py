from Vmsom_ex1inac_conf import *
reps = 9000
glide = ScenaGlide(sina, sinb)
gmids = [glide.getGlide(i/8.0) for i in range(9)]
print [s.nid for s in gmids]
gmids = [ s if s.nid in (200.0, 500.0, 800.0) else None for s in gmids ]
print gmids
revSvcCI_rev1format(gmids, mymeth, reps, 'Vmsom-ex1inac-fscen.pkl')
