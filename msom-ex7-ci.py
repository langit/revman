from msom_ex7_conf import *
reps = 5000
glide = ScenaGlide(sinb, sina)

gmid = glide.getGlide(0.0)
ffmt = '%.2f'
print "\multicolumn{Example-4, S=", ffmt%gmid.nid,"}\\\\"
revSvcCI(gmid, mymeth, reps)

gmid = glide.getGlide(0.5)
print "\multicolumn{Example-4, S=", ffmt%gmid.nid,"}\\\\"
revSvcCI(gmid, mymeth, reps)

gmid = glide.getGlide(1.0)
print "\multicolumn{Example-4, S=", ffmt%gmid.nid,"}\\\\"
revSvcCI(gmid, mymeth, reps)
