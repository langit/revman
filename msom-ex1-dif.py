from msom_ex1_conf import *
mymeth=("EMSR/SL", "OBSA/CR", "EMSR/CR", "CRSA/NV", "EMSR/NV", "DP/LBH")
glide = ScenaGlide(sinb, sina)
gmid = glide.getGlide(0.5)
print "Nid:", gmid.nid
diffRevTest(gmid, mymeth, 30, 5000)
