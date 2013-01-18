from msom_ex2_conf import *

xlab = 'mean no-show rate'

glide = ScenaGlide(sina, sinb)
gmid = glide.getGlide(0.5)
print "Nid:", gmid.nid
scenaQuant(gmid, mymeth, 'msom-ex2-pshift')
ylim(5200,7700)
savefig('msom-ex2-pshift-qnt.eps')
#scenaQuant(gmid, mymeth)
