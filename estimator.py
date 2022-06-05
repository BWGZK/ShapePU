def EM_estimate(dic_prob):
    values = []
    labels = []
    for label in dic_prob.keys():
        g0 = dic_prob[label]["g"]
        p0 = dic_prob[label]["p"]
        p0_ = dic_prob[label]["n"]
        P0_ = g0.mean()
        labels.append(label)
        values.append((g0,p0,p0_,P0_))
    delta =1
    while delta > 1e-3:
        numes = []
        alpha_dict = {}
        for value in values:
            g0,p0,p0_,P0_ = value
            P0_ini = P0_
            nume0 = (P0_*p0)/g0.mean()
            numes.append(nume0)
        denom = sum(numes)
        delta = 0
        for i in range(len(numes)):
            nume = numes[i]
            g0,p0,p0_,P0_ini = values[i]
            P0_ = (nume/denom).mean()
            values[i] = (g0,p0,p0_,P0_)
            delta = delta + abs(P0_-P0_ini)
            alpha_dict.update({labels[i]:P0_})
    return alpha_dict