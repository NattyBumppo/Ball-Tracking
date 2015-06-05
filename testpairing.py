import itertools
import math

def distance(p, q):
    dist = math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
    return dist


p0 = (22,33)
p1 = (55,33)
p2 = (10,39)
p3 = (0,0)
p_arr = [p0, p1, p2, p3]

n = len(p_arr)

c0 = (1,1)
c1 = (10.1, 38)
c2 = (54,34)
c3 = (21,30)
c_arr = [c0,c1,c2,c3]

# Find minimal pairings
pairings = []
for pair in itertools.product(p_arr, c_arr):
    # print pair
    pairings.append((pair, distance(pair[0], pair[1])))

# for pair in pairings:
    # print pair

sorted_pairings = sorted(pairings, key=lambda item: item[1])
# print sorted_pairings

# Go through the list of sorted pairings and find matches for all p values
min_matches = []
for pairing in sorted_pairings:
    p = pairing[0][0] 
    if p not in [x[0][0] for x in min_matches]:
        min_matches.append(pairing)
        print "couldn't find", pairing[0]

print min_matches