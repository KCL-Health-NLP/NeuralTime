


# these functions are used to preprocess entities during the training : indeed, spacy does not accept entities overlapping

def merge_intervals(intervals):

    #intervals = [(x[0], x[1]) for x in entities]

    """
    A simple algorithm can be used:
    1. Sort the intervals in increasing order
    2. Push the first interval on the stack
    3. Iterate through intervals and for each one compare current interval
       with the top of the stack and:
       A. If current interval does not overlap, push on to stack
       B. If current interval does overlap, merge both intervals in to one
          and push on to stack
    4. At the end return stack
    """

    si = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for tup in si:
        if not merged:
            merged.append(tup)
        else:
            b = merged.pop()
            if b[1] >= tup[0]:
                new_tup = tuple([b[0], max(b[1], tup[1]), tup[2]]) # we choose randomly the label
                merged.append(new_tup)
            else:
                merged.append(b)
                merged.append(tup)
    return merged




def entities_overlap(ent1, ent2):
    # ent1 and en2 are two entities list.
    # it is assumed that inside ent1 and ent2, no entities overlap
    # the function finds the entities of ent2 which overlap with the entities of ent1 and discards them


    a = [(x[0], x[1]) for x in ent1]
    b = [(x[0], x[1]) for x in ent2]

    a.sort()
    b.sort()

    i = j = 0

    overlapping_entities = []

    while i < len(a) and j < len(b):
        a_left, a_right = a[i]
        b_left, b_right = b[j]

        if a_right <= b_left:
            i += 1
        elif b_right <= a_left:
            j += 1
        elif b_left <= a_left and b_right >= a_right:
            overlapping_entities += [ent2[j]]
            i += 1
            j += 1
        elif a_left <= b_left and a_right >= b_right:
            overlapping_entities += [ent2[j]]
            j += 1
        elif a_right >= b_left and a_right <= b_right and a_left <= b_right:
            overlapping_entities += [ent2[j]]
            i += 1
        elif a_left >= b_left and a_left <= b_right and a_right >= b_right:
            overlapping_entities += [ent2[j]]
            j += 1

    return [e for e in ent2 if e not in overlapping_entities]



def temp_entities_overlap(ent):
    # ent1 and en2 are two entities list.
    # it is assumed that inside ent1 and ent2, no entities overlap
    # the function finds the entities of ent2 which overlap with the entities of ent1 and discards them

    ent.sort()


    # build ranges first
    def expand(list):
        newList = []
        for r in list:
            newList.append(range(r[0], r[1]))
        print(newList)
        return newList

    def compare(l):
        toBeDeleted = []
        toKeep = []

        for el in l:
            if l.count(el)>1:
                toBeDeleted += [l.index(el)]

        for index1 in range(len(l)):
            for index2 in range(len(l)):
                if index1 == index2:
                    # we dont want to compare ourselfs
                    continue
                matches = [x for x in l[index1] if x in l[index2]]
                x = l[index1]
                y = l[index2]
                intersect = range(max(x[0], y[0]), min(x[-1], y[-1])+1)
                print(x, y, intersect)
                if len(list(intersect)) != 0:  # do we have overlap?
                    print('Overlap', x, y)
                    ## compare lengths and get rid of the longer one
                    if len(l[index1]) >= len(l[index2]):
                        toBeDeleted.append(index2)
                        break
                    elif len(l[index1]) < len(l[index2]):
                        toBeDeleted.append(index1)
        # distinct
        toBeDeleted = [toBeDeleted[i] for i, x in enumerate(toBeDeleted) if x not in toBeDeleted[i + 1:]]

        print(toBeDeleted)
        # remove items
        for i in toBeDeleted[::-1]:
            try:
                del l[i]
            except:
                print()
        return toBeDeleted

    to_delete = compare(expand(ent))
    return [e for e in ent if ent.index(e) not in to_delete]

#test_list = [(0,1, 'DURATION'), (2,6, 'TIME'), (3,4, 'DURATION'), (2,6, 'FREQUENCY'), (2711, 2718, 'TIME'),  (2712, 2719, 'FREQUENCY')]
#print(merge_intervals(test_list))

