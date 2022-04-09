import numpy as np

'''
Exercise: Write a function that takes an annotation as those in true-ann1.fa and true-ann2.fa
and maps the annotation—sequence of N, C, and R, to HMM states. 
'''
#this functions return dictionaries, where keys are mapped to sequences of HMM states
def give_HMM_states_3(filename):
    annot_sequences = read_fasta_file(filename)
    hmm_states = {}

    for seq_name in annot_sequences:
        hmm_line = ""
        for i in range(0, len(annot_sequences[seq_name])):
            if annot_sequences[seq_name][i] == 'C':
                hmm_line += '0'
            elif annot_sequences[seq_name][i] == 'N':
                hmm_line += '1'
            else:
                hmm_line += '2'
        hmm_states[seq_name] = hmm_line
    return hmm_states

def give_HMM_states_7(filename):
    annot_sequences = read_fasta_file(filename)
    hmm_states = {}
    r1, r2, c1, c2 = False, False, False, False

    for seq_name in annot_sequences:
        hmm_line = ""
        for i in range(0, len(annot_sequences[seq_name])):
            if annot_sequences[seq_name][i] == 'C':
                if c1:
                    hmm_line += '1'
                    c1, c2 = False, True
                elif c2:
                    hmm_line += '2'
                    c1, c2 = False, False
                else:
                    hmm_line += '0'
                    c1 = True
            elif annot_sequences[seq_name][i] == 'N':
                hmm_line += '3'
            else:
                if r1:
                    hmm_line += '5'
                    r1, r2 = False, True
                elif r2:
                    hmm_line += '6'
                    r1, r2 = False, False
                else:
                    hmm_line += '4'
                    r1 = True                
        hmm_states[seq_name] = hmm_line
    return hmm_states


'''
Exercise: Implementing the Viterbi algorithm to get the Viterbi matrix.
'''
#we didn't use the given log() function, but instead used finfo for transforming the numbers into log space
def viterbi(obs, hmm):
    X = translate_observations_to_indices(obs)
    N = len(X) 
    K = len(hmm.init_probs)
    V = np.zeros((K, N)) #the Viterbi matrix
    B = np.zeros((K, N-1)).astype(np.int32) #the backtracking matrix

    tiny = np.finfo(0.).tiny
    trans_prob_log = np.log(hmm.trans_probs + tiny)
    init_prob_log = np.log(hmm.init_probs + tiny)
    emiss_prob_log = np.log(hmm.emission_probs + tiny)

    # Initialise the first column of V
    V[:, 0] = init_prob_log + emiss_prob_log[:, 0]
    
    for i in range(1, N):
        for j in range(0, K):
            temp_sum = trans_prob_log[:, j] + V[:, i - 1]
            V[j, i] = np.max(temp_sum) + emiss_prob_log[j, X[i]]
            B[j, i-1] = np.argmax(temp_sum)
    
    #Backtrack
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(V[:, -1])
    for i in range(N-2, 0, -1):
        S_opt[i] = B[int(S_opt[i+1]), i]

    return V, S_opt, B