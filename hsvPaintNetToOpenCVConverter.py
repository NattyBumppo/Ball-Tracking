lines = open('color.txt', 'r').readlines()

Hs = []
Ss = []
Vs = []

for line in lines:
    if line.strip() != '':

        # Should be a color in H S V format
        H, S, V = line.split()
        
        H = float(H)
        S = float(S)
        V = float(V)
        
        Hs.append(H)
        Ss.append(S)
        Vs.append(V)

# Find extremes
minH = min(Hs)
maxH = max(Hs)
minS = min(Ss)
maxS = max(Ss)
minV = min(Vs)
maxV = max(Vs)

# Convert from 0-360, 0-100, 0-100
# to 0-180, 0-255, 0-255
minHcv2 = minH * 0.5
maxHcv2 = maxH * 0.5
minScv2 = minS * 255.0 / 100.0
maxScv2 = maxS * 255.0 / 100.0
minVcv2 = minV * 255.0 / 100.0
maxVcv2 = maxV * 255.0 / 100.0

print 'Values converted to OpenCV format.'
print ''
print 'Minimum HSV bound: (%s, %s, %s)' % (minHcv2, minScv2, minVcv2)
print 'Maximum HSV bound: (%s, %s, %s)' % (maxHcv2, maxScv2, maxVcv2)