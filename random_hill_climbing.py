import numpy as np
import pickle


def hill_climbing(f, num_iter=200, directions=5, initial_step_size=4, N=5, initial_vector=None, length=29, T=1,
                  best_N_reevaluate=False):
    # creates a list of similar vectors to those in best_N, for every vector v in best_N we create 'directions' similar
    # vectors, at different similarity levels. also, the similarity level decrease with the with te iteration number and
    # increase with the temperature T.
    def similar(i, best_N):
        similar = []
        for k in range(len(best_N)):
            orig = best_N[k][0]
            m = (initial_step_size - 0.1) / num_iter
            for j in range(directions):
                step = (initial_step_size - m*i + j/directions)**(T*(directions+j+1)/(2*directions))
                y = orig + np.random.uniform(-step, step, length)
                similar.append(y)
        return similar

    # checks if the new x,f tuple is good enough to be added to best_N, and adds it if so
    def update_best_N(tup ,best_N):
        min_f = min([tup_[1] for tup_ in best_N])
        if tup[1] > min_f:
            min_tup = [tup for tup in best_N if tup[1] == min_f][0]
            best_N = [tup for tup in best_N if tup[1] > min_f] + [tup]
            if len(best_N) < N:
                best_N.append(min_tup)
        return best_N

    if initial_vector is None:
        best_N_xs = [np.random.uniform(0, 1, length) for _ in range(N)]
        best_N = [(x, f(x)) for x in best_N_xs]
    else:
        best_N_xs = [initial_vector]
        best_N_xs.extend([initial_vector + np.random.uniform(0, 1, length) for _ in range(N-1)])
        best_N = [(x, f(x)) for x in best_N_xs]

    # the main loop
    for i in range(1, num_iter+1):
        if best_N_reevaluate:
            best_N = [(tup[0], f(tup[0])) for tup in best_N]
        print('################################# -- iter num = ', i, '-- #################################')
        # print('best_N - ', best_N)
        for y in similar(i, best_N):
            f_y = f(y)
            best_N = update_best_N((y, f_y), best_N)

        pickle.dump(best_N, open('best_N', 'wb'))  # saving best N every iteration
        max_f = max([tup[1] for tup in best_N])
        max_tup = [tup for tup in best_N if tup[1] >= max_f][0]
        pickle.dump(max_tup, open('random_hill_climbing_max_tup', 'wb'))

    max_f = max([tup[1] for tup in best_N])
    max_tup = [tup for tup in best_N if tup[1] >= max_f][0]
    print('\nfinished hill climbing\n')
    return max_tup