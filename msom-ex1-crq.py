from msom_ex1_conf import *

glide = ScenaGlide(sinb, sina)
gmid = glide.getGlide(0.5)
print "Nid:", gmid.nid
#scenaQuant(gmid, mymeth, 'msom-ex1-prange-cr', True)
scenaQuant(gmid, mymeth, None, True)
