from Vmsom_ex1_conf import *

glide = ScenaGlide(sina, sinb)
gmid = glide.getGlide(0.5)
print "Nid:", gmid.nid
scenaQuant(gmid, mymeth, 'Vmsom-ex1')
#ylim(5500,7800)
#savefig('Vmsom-ex1-qnt.eps')
#scenaQuant(gmid, mymeth)
