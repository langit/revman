from msom_ex1_conf import *

glide = ScenaGlide(sinb, sina)
gmid = glide.getGlide(0.5)
print "Nid:", gmid.nid
scenaQuant(gmid, mymeth, 'msom-ex1-prange')
ylim(5500,7800)
savefig('msom-ex1-prange-qnt.eps')
#scenaQuant(gmid, mymeth)
