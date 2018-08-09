def average_branching_data(branchings, granularity=10, max_t=100, weighted_time_average=True):
    avg_branching = np.zeros((2, max_t, granularity**4))
    for i in range(len(branchings)):
        B, T = branchings[i][0].shape[:2]
        for b in range(B):
            for t in range(T):
                if branchings[i][1][b,t]==False:
                    avg_branching[0, t, branchings[i][0][b,t]]+=1
                    avg_branching[1, t]+= np.ones((granularity**4))
    if weighted_time_avegage:
        return (avg_branching[0]/np.sum(avg_branching[0])).reshape((max_t, granularity, granularity, granularity, granularity))
    else:
        avg_b = avg_branching[0]/np.clip(avg_branching[1], 0.1, np.inf)
        avg_b = avg_b/np.sum(avg_b)
        return avg_b.reshape((max_t, granularity, granularity, granularity, granularity))