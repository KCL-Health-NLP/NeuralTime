from collections import Counter
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score
import random

def get_tag_attrs(tag, attrs):
    values = {}
    
    for attr in attrs:
        val = tag.get(attr, None)
        values[attr] = val
    
    return values

def prf(tp, fp, fn):
    print('-- Calculating precision, recall and f-score')
    print('   tp:', tp)
    print('   fp:', fp)
    print('   fn:', fn)

    if tp + fp == 0.0 or tp + fn == 0.0:
        print('-- Warning: cannot calculate metrics with zero denominator')
        return 0.0, 0.0, 0.0

    if(tp>0):
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2 * p * r / (p + r)
    else:
        p = 0
        r = 0
        f = 0
    print('precision ' + str(p))
    print('recall ' + str(r))
    return p, r, f


def match_span(a1, a2, matching):
    # check if two annotated expressions match

    s1 = int(a1['start'])
    s2 = int(a2['start'])
    e1 = int(a1['end'])
    e2 = int(a2['end'])
    t1 = a1['text']
    t2 = a2['text']

    match_str = '{} {} {}\n{} {} {}'.format(s1, e1, t1, s2, e2, t2)
    
    # Exact match (strict matching)
    if s1 == s2 and e1 == e2:
        return True, match_str
       
    if matching == 'relaxed':

         # if one expression contains the other :
        if s1 <= s2 and e1 >= e2:
            # if a1 contains a2
            return True, match_str


        if s1 >= s2 and e1 <= e2:
            # if a2 contains a1
            return True, match_str

        # if the expressions overlap :
        if s1 <= s2 and e1 >= s2:
            return True, match_str

        if s1 >= s2 and s1 <= e2:
            return True, match_str

    return False, ''


def match_types(a1, a2, type_dict):

    # this functions computes the tp, fp and fn for every occurence of types provided
    # a1 and a2 are annotations dataframes in the standard format
    # a1 is considered the gold standard

    attr_agr = {}
    # for a in attrs_to_check:
    #    attr_agr[a] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}


    val1 = a1.get('type', None)
    val2 = a2.get('type', None)
    if val1 is not None and val2 is not None:
        if val1 not in type_dict.keys():
            type_dict[val1] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        if val2 not in type_dict.keys():
            type_dict[val2] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        if val1 == val2:
            type_dict[val1]['tp'] += 1
        else:
            # if the two types do not match, then it is a fp for the second type and a fn for the first
            type_dict[val1]['fp'] += 1
            type_dict[val2]['fn'] += 1
    return type_dict



def match_attributes(a1, a2, attrs_to_check):
    attr_agr = {}
    #for a in attrs_to_check:
    #    attr_agr[a] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    
    for attr in attrs_to_check:
        val1 = a1.get(attr, None)
        val2 = a2.get(attr, None)
        if val1 is not None and val2 is not None:
            if val1 == val2:
                scores = attr_agr.get(attr, {})
                tp = scores.get('tp', 0) + 1
                scores['tp'] = tp
                attr_agr[attr] = scores
            else:
                # this is fp and fn - weird
                scores = attr_agr.get(attr, {})
                fp = scores.get('fp', 0) + 1
                scores['fp'] = fp
                attr_agr[attr] = scores
                fn = scores.get('fn', 0) + 1
                scores['fn'] = fn
                attr_agr[attr] = scores
        elif val1 is None and val2 is not None:
            scores = attr_agr.get(attr, {})
            fp = scores.get('fp', 0) + 1
            scores['fp'] = fp
            attr_agr[attr] = scores
        elif val1 is not None and val2 is None:
            scores = attr_agr.get(attr, {})
            fn = scores.get('fn', 0) + 1
            scores['fn'] = fn
            attr_agr[attr] = scores
        else:
            scores = attr_agr.get(attr, {})
            tn = scores.get('tn', 0) + 1
            scores['tn'] = tn
            attr_agr[attr] = scores
    return attr_agr


def count_agreements_simple(annotations1, annotations2, matching, attrs_to_check, type_dict):
    matched1 = []
    matched2 = []
    non_matched1 = []
    non_matched2 = []

    tp = fp = fn = 0
    
    attr_agr = {}
    attr_vals1 = []
    attr_vals2 = []

    
    for tag1 in annotations1:
        for tag2 in annotations2:
            m, r = match_span(tag1, tag2, matching)
            if m:
                
                # span
                matched1.append(tag1)
                matched2.append(tag2)
                tp += 1

                # attributes
                a = match_attributes(tag1, tag2, attrs_to_check)
                type_dict = match_types(tag1, tag2, type_dict)

                for attr in a:
                    curr_agr = attr_agr.get(attr, {})
                    new_agr = a[attr]
                    c = dict(Counter(curr_agr) + Counter(new_agr))
                    attr_agr[attr] = c
                
                # testing
                vals1 = get_tag_attrs(tag1, attrs_to_check)
                vals2 = get_tag_attrs(tag2, attrs_to_check)
                attr_vals1.append(vals1)
                attr_vals2.append(vals2)
                
                break

    for tag2 in annotations2:
        if tag2 not in matched2:
            for tag1 in annotations1:
                if tag1 not in matched1:
                    m, r = match_span(tag2, tag1, matching)
                    if m:

                        # span
                        matched1.append(tag1)
                        matched2.append(tag2)
                        tp += 1
                        
                        # attributes
                        a = match_attributes(tag1, tag2, attrs_to_check)
                        type_dict = match_types(tag1, tag2, type_dict)

                        for attr in a:
                            curr_agr = attr_agr.get(attr, {})
                            new_agr = a[attr]
                            c = dict(Counter(curr_agr) + Counter(new_agr))
                            attr_agr[attr] = c
                        
                        # testing
                        vals1 = get_tag_attrs(tag1, attrs_to_check)
                        vals2 = get_tag_attrs(tag2, attrs_to_check)
                        attr_vals1.append(vals1)
                        attr_vals2.append(vals2)
                
                        break

    for tag1 in annotations1:
        if tag1 not in matched1:
            # span
            non_matched1.append(tag1)
            fn += 1

            t = tag1.get('type', None)
            if t not in type_dict.keys():
                type_dict[t] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 1}
            else :
                type_dict[t]['fn'] += 1

            if random.random()> 1:
                print('False negative')
                print(tag1)
                print()

    for tag2 in annotations2:
        if tag2 not in matched2:
            non_matched2.append(tag2)
            fp += 1

            t = tag2.get('type', None)
            if t not in type_dict.keys():
                type_dict[t] = {'tp': 0, 'tn': 0, 'fp': 1, 'fn': 0}
            else:
                type_dict[t]['fp'] += 1

            if random.random()> 1 :
                print('False positive')
                print(tag2)
                print()

    return tp, fp, fn, attr_agr, attr_vals1, attr_vals2, matched1, matched2, non_matched1, non_matched2, type_dict



#files1/2: file names for annotations1/2
#ann1/ann2: dataframes in the form ['doc', 'text', 'start', 'end', 'attribute1', 'attribute2', ..]
#attrs_to_check: list of attributes to compare

def batch_agreement(files1, ann1, ann2, matching='relaxed', attrs_to_check = []):
    
    if matching not in ['strict', 'relaxed']:
        raise ValueError('-- Invalid matching type "' + str(matching) + '". Use "strict" or "relaxed".')
    
    tp_g = fp_g = fn_g = 0.0
    
    attr_agr_g = {}
    attr_vals1_g = []
    attr_vals2_g = []
    
    matched_all = []
    non_matched_all = []

    type_dict = {}
    
    for f1 in files1:

                df1 = ann1[ann1['doc']==f1]
                df2 = ann2[ann2['doc']==f1]

                ann_list1 = df1.to_dict('records')
                ann_list2 = df2.to_dict('records')

                tp, fp, fn, attr_agr, attr_vals1, attr_vals2, matched1, matched2, non_matched1, non_matched2, type_dict = count_agreements_simple(ann_list1, ann_list2, 'relaxed', attrs_to_check, type_dict)



                tp_g += tp
                fp_g += fp
                fn_g += fn
                for attr in attr_agr:
                    curr_agr = attr_agr_g.get(attr, {})
                    new_agr = attr_agr[attr]
                    c = dict(Counter(curr_agr) + Counter(new_agr))
                    attr_agr_g[attr] = c
                
                # Used for scikit-learn calculations
                attr_vals1_g.extend(attr_vals1)
                attr_vals2_g.extend(attr_vals2)
                
                matched_all.append([f1, matched1, matched2])
                non_matched_all.append([f1, non_matched1, non_matched2])

    assert len(attr_vals1_g) == len(attr_vals2_g)

    p, r, f = prf(tp_g, fp_g, fn_g)

    print('f-score  : ' + str(f) + '\n')

    # Using scikit-learn (per-class results)
    if len(attrs_to_check)>0:
        
        for attr in attrs_to_check:

            sample1 = [k.get(attr, None) for k in attr_vals1_g]
            sample2 = [k.get(attr, None) for k in attr_vals2_g]
  
            sample1 = ['None' if v is None else v for v in sample1]
            sample2 = ['None' if v is None else v for v in sample2]

            
            scores = {}
            
            scores['weight'] = precision_recall_fscore_support(sample1, sample2, average='weighted')
            pt, rt, ft, st = scores['weight']
            print('p,r,f1', scores['weight']) 

            acc = accuracy_score(sample1, sample2)
            print('acc', acc)

            k = cohen_kappa_score(sample1, sample2)

            print('k', k)

    print(type_dict)
        
    return tp_g, fp_g, fn_g, p, r, f, pt, rt, ft, type_dict


