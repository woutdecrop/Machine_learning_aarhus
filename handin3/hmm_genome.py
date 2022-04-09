import os, sys
import pathlib
import numpy as np
import compare_anns as ca


# Helper Functions
def read_fasta(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.

    Args:
        filename: the file you want to read, i.e. genome1.fa, genome2.fa, ..., true_ann5.fa
    """
    sequences_lines = {}
    current_sequence_lines = None
    result_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'data')
    my_path = os.path.join(result_path, filename)
    with open(my_path) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences


def write_fasta(filename, n_genome, annotation):
    """
    Write a FASTA file filname

    Args:
        filename: string with the filename where to write, should be like pred-ann6.fa
        n_genome: the number of genome you want to write, should be from 6 to 10
        annotation: decodification of the genome with N, C and R states
    """
    result_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    my_path = os.path.join(result_path, filename)

    s = '; genome' + str(n_genome) + ' FASTA file\n'
    s += '>pred-ann' + str(n_genome) + '\n'
    s += annotation + '\n'
    with open(my_path, 'w') as fp:
        fp.write(s)
        fp.close()


### Rest of the code
def translate_NCR_to_states(annotation, observations):
    """
    Args:
        annotation: string containing the NCR state, e.g. NNNCCCCCCCCCNNNNRRRRRRN
        observations: string containing the nucleobases, i.e. ATCG
    ATTENTION: STRING are required !
    
    return:
        A list of int corresponding to the hidden states of HMM
    """
    state = "N"
    out = []
    i = 0
    while i < (len(annotation) - 1):
        if state == "N":
            if annotation[i] == 'C':
                state = "StartC"
                i -= 1
            elif annotation[i] == 'R':
                state = "StopR"
                i -= 1
            elif annotation[i + 1] == 'C':
                state = "StartC"
                out.append(0)
            elif annotation[i + 1] == 'R':
                state = "StopR"
                out.append(0)
            elif annotation[i + 1] == 'N':
                state = "N"
                out.append(0)
            else:
                raise Exception("Wrong parser!")
        elif state == "StartC":
            if observations[i:i + 3] == 'ATG':
                out.extend([1, 2, 3])
            elif observations[i:i + 3] == 'TTG':
                out.extend([4, 5, 6])
            elif observations[i:i + 3] == 'GTG':
                out.extend([7, 8, 9])
            elif observations[i:i + 3] == 'ATA':
                out.extend([10, 11, 12])
            elif observations[i:i + 3] == 'ATT':
                out.extend([13, 14, 15])
            elif observations[i:i + 3] == 'GTT':
                out.extend([16, 17, 18])
            elif observations[i:i + 3] == 'ATC':
                out.extend([19, 20, 21])
            elif observations[i:i + 3] == 'CTG':
                out.extend([70, 71, 72])
            else:
                raise Exception("Wrong parser!")
            if annotation[i] != 'C':
                raise Exception("Wrong parser!")
            state = "C"
            i += 2
        elif state == "C":
            if annotation[i + 3] != 'C':
                state = "StopC"
                i -= 1  # don't move
            else:
                out.extend([22, 23, 24])
                i += 2
            if annotation[i] != 'C':
                raise Exception("Wrong parser!")
        elif state == "StopC":
            if observations[i:i + 3] == 'TAA':
                out.extend([25, 26, 27])
            elif observations[i:i + 3] == 'TGA':
                out.extend([28, 29, 30])
            elif observations[i:i + 3] == 'TAG':
                out.extend([31, 32, 33])
            else:
                err = observations[i:i + 3]
                raise Exception("Wrong parser!")
            if annotation[i] != 'C':
                raise Exception("Wrong parser!")
            i += 2
            state = "N"
        elif state == "StopR":
            if observations[i:i + 3] == 'TTA':
                out.extend([34, 35, 36])
            elif observations[i:i + 3] == 'TCA':
                out.extend([37, 38, 39])
            elif observations[i:i + 3] == 'CTA':
                out.extend([40, 41, 42])
            else:
                raise Exception("Wrong parser!")
            if annotation[i] != 'R':
                raise Exception("Wrong parser!")
            state = "R"
            i += 2
        elif state == "R":
            if annotation[i + 3] != 'R':
                state = "StartR"
                i -= 1  # don't move
            else:
                out.extend([43, 44, 45])
                i += 2
            if annotation[i] != 'R':
                raise Exception("Wrong parser!")
        elif state == "StartR":
            if observations[i:i + 3] == 'CAT':
                out.extend([46, 47, 48])
            elif observations[i:i + 3] == 'CAA':
                out.extend([49, 50, 51])
            elif observations[i:i + 3] == 'CAC':
                out.extend([52, 53, 54])
            elif observations[i:i + 3] == 'TAT':
                out.extend([55, 56, 57])
            elif observations[i:i + 3] == 'AAT':
                out.extend([58, 59, 60])
            elif observations[i:i + 3] == 'CAG':
                out.extend([61, 62, 63])
            elif observations[i:i + 3] == 'GAT':
                out.extend([64, 65, 66])
            elif observations[i:i + 3] == 'AAC':
                out.extend([67, 68, 69])
            else:
                raise Exception("Wrong parser!")
            if annotation[i] != 'R':
                raise Exception("Wrong parser!")
            state = "N"
            i += 2
        i += 1
    out.append(0)
    return out

def translate_states_to_NCR(states):
    """
    Args:
        states: list of int containing the states of HMM (found by Viterbi decoding)
    Return:
        string with high-level states, i.e. N,C and R
    """

    out = ''
    for z in states:
        if z == 0:
            out += 'N'
        elif (0 < z < 34) or (z == 70 or z == 71 or z == 72):
            out += 'C'
        else:
            out += 'R'
    return out


class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs


def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]


def training_by_counting(X, Z, K, D):
    """
    Args:
        X: LIST of lists observations in number format, e.g. [[1243212], [3241221], [231221]]
        Z: LIST of lists transactions in int, e.g. [[23,12,54,12,43], [12,53,11,2,1],[0,0,0,1,2,3]]
        K: number of hidden states
        D: number of observable symbols
    X and Z have to be coherent, i.e. X[i] must be the observations corresponding to Z[i]
    Returns:
        KxK matrix, a KxD matrix and a K vector containing counts cf. above
    """
    pi = np.zeros(K)
    trans_counts = np.zeros((K, K))
    emit_counts = np.zeros((K, D))

    pi[0] = 1  # always start in 0

    for t in range(len(Z)):
        gen = X[t]
        anno = Z[t]
        emit_counts[anno[0], gen[0]] += 1
        for i in range(1, len(anno)):
            trans_counts[anno[i-1], anno[i]] += 1
            emit_counts[anno[i], gen[i]] += 1

    row_sum_trans = np.sum(trans_counts, axis=1)
    row_sum_emit = np.sum(emit_counts, axis=1)

    E = np.zeros((K, D))
    T = np.zeros((K, K))
    for r in range(K):
        T[r:] = trans_counts[r, :] / row_sum_trans[r] if row_sum_trans[r] != 0 else trans_counts[r, :]
        E[r:] = emit_counts[r, :] / row_sum_emit[r] if row_sum_emit[r] != 0 else emit_counts[r, :]
    return pi, T, E


def viterbi(X, hmm):
    """
    Args:
        X: Sequence that you want to decode, should be in number format, e.g. [[1243212], [3241221], [231221]]
        hmm: Hidden Markov Model object
    :return: The most likely sequence of states given by X
    """
    N = len(X)  # N
    K = len(hmm.init_probs)  # I
    V = np.zeros((K, N))
    # B is the backtracking matrix
    B = np.zeros((K, N - 1)).astype(np.int32)

    tiny = np.finfo(0.).tiny

    trans_prob_log = np.log(hmm.trans_probs + tiny)
    init_prob_log = np.log(hmm.init_probs + tiny)
    emiss_prob_log = np.log(hmm.emission_probs + tiny)

    # Initialise the first column of V
    V[:, 0] = init_prob_log + emiss_prob_log[:, X[0]] #before was emiss_prob_log[:, 0]

    for n in range(1, N):
        # Implement the Viterbi algorithm
        if not n % int(N * 0.01):
            sys.stdout.write('\u001b[1000D' + str(int(n / (N * 0.01))) + "% ")  # keep track of execution
            sys.stdout.flush()
        temp_sum = np.transpose(trans_prob_log) + V[:, n - 1]
        V[:, n] = np.max(temp_sum, axis=1) + emiss_prob_log[:, X[n]]
        B[:, n - 1] = np.argmax(temp_sum, axis=1)
    print(" ")

    # Backtrack
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(V[:, -1])
    for i in range(N - 2, -1, -1):
        S_opt[i] = B[int(S_opt[i + 1]), i]

    return S_opt


def read_all():
    Z = []
    Z_ncr = []
    X = []
    for i in range(1, 6):
      x = read_fasta('genome' + str(i) + '.fa')
      z = read_fasta('true-ann' + str(i) + '.fa')
      # Create list of lists
      # X has lists of observation in number format, i.e. NOT ATGC
      # Z has lists of states as int, i.e. NOT NCR
      X.append(translate_observations_to_indices(x['genome' + str(i)]))
      Z_ncr.append(z['true-ann' + str(i)])
      Z.append(translate_NCR_to_states(z['true-ann' + str(i)], x['genome' + str(i)]))

    return Z, Z_ncr, X


def cross_validation():
    print("Reading genomes")
    Z, Z_ncr, X = read_all()
    print("Cross validating")
    best_hmm = None
    best_ac = -1
    for genome in range(1, 6):
        i = genome - 1
        X_training = X[0:i] + X[i + 1:]
        Z_training = Z[0:i] + Z[i + 1:]
        X_valid = X[i]
        Z_valid = Z_ncr[i]
        pi, T, E = training_by_counting(X_training, Z_training, 73, 4)
        HMM = hmm(pi, T, E)
        my_Z = viterbi(X_valid, HMM)
        my_Zncr = translate_states_to_NCR(my_Z)
        print("Genome ", genome)
        ac = ca.print_all(Z_valid, my_Zncr)
        if ac > best_ac:
            best_ac = ac
            best_hmm = HMM

    return best_hmm

def decode_genome(n_genome, hmm):
  """
  Create a FASTA file with the decodification of the genome
  Args:
    n_genome: number of the genome to decode
    hmm: Hidden Markov Model to use for viterbi decoding
  """
  x = read_fasta('genome' + str(n_genome) + '.fa')
  x = translate_observations_to_indices(x['genome' + str(n_genome)])
  my_Z = viterbi(x, hmm)
  my_Zncr = translate_states_to_NCR(my_Z)
  write_fasta('pred-ann' + str(n_genome) + '.fa', n_genome, my_Zncr)


if __name__ == '__main__':
    best_hmm = cross_validation()
    for n_genome in range(6, 11):
        print("Decoding genome ", n_genome)
        decode_genome(n_genome, best_hmm)
    print("DONE!")