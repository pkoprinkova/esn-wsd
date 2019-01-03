import cPickle
import argparse

from numpy import flip, zeros, tanh, dot, vstack, asarray, reshape, square, sqrt
from preprocess_data import load_embeddings, read_data, format_data

from calc_accuracy import calculate_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0', description='Train or evaluate a neural WSD model.',
                                     fromfile_prefix_chars='@')
    parser.add_argument('-embeddings_model', dest='embeddings_model', required=True, help='Location of the embeddings.')
    parser.add_argument('-test_data', dest='test_data', required=True, help='Path to the test data.')
    parser.add_argument('-dictionary', dest='dictionary', required=True, help='Path to the WordNet dictionary.')
    parser.add_argument('-sensekey2synset', dest='sensekey2synset', required=True, help='Path to the synset mappings.')
    parser.add_argument('-embeddings_size', dest='embeddings_size', required=True)
    parser.add_argument('-window_size', dest='window_size', required=True)
    parser.add_argument('-use_reservoirs', dest='use_reservoirs', required=False, default="True",
                        help="Use reseroirs or train directly on the word embeddings.")
    parser.add_argument('-bidirectional', dest='bidirectional', required=False, default="True",
                        help="Use a bidirectional architecture, or just one reservoir.")
    parser.add_argument('-res_size', dest='res_size', required=False, default=100, help="Size of the echo state reservoirs.")
    parser.add_argument('-only_open_class', dest='only_open_class', required=False, default="True")
    parser.add_argument('-save_path', dest='save_path', required=True, help='Path to the pickled model files.')

    args = parser.parse_args()
    embeddings_model = args.embeddings_model
    test_data_path = args.test_data
    sensekey2synset = args.sensekey2synset
    dictionary = args.dictionary
    embeddings_size = int(args.embeddings_size)
    window_size = int(args.window_size)
    use_reservoirs = args.use_reservoirs
    bidirectional = args.bidirectional
    if use_reservoirs == "True":
        res_size = int(args.res_size)
    else:
        res_size = 0
    if bidirectional == "True":
        total_res_size = res_size * 2
    else:
        total_res_size = res_size
    only_open_class = args.only_open_class
    save_path = args.save_path

    embeddings = load_embeddings(embeddings_model)  # load the embeddings
    f_sensekey2synset = cPickle.load(open(sensekey2synset, "rb"))  # get the mapping between synset keys and IDs
    test_data = read_data(test_data_path, f_sensekey2synset, only_open_class)

    inSize = embeddings_size * window_size
    outSize = embeddings_size


    with open(save_path, "rb") as input_file:
        _ = cPickle.load(input_file)
        resSparsity = cPickle.load(input_file)
        a = cPickle.load(input_file)
        Wout = cPickle.load(input_file)
        if use_reservoirs == "True":
            Win_fw = cPickle.load(input_file)
            Win_bw = cPickle.load(input_file)
            W_fw = cPickle.load(input_file)
            W_bw = cPickle.load(input_file)


    # test the trained ESN
    print 'Testing...'
    x = zeros((res_size, 1))
    y=zeros((outSize,1))
    states = []
    for i, text in enumerate(test_data):
        inputs, fw_states, bw_states = [], [], []
        Xts, Yts, lemmas, synsets, pos = format_data(test_data[i], embeddings, embeddings_size, window_size)
        testLen = len(Xts)
        u_fw = zeros((inSize, 1))
        if use_reservoirs == "True":
            u_bw = zeros((inSize, 1))
            x_fw = zeros((res_size, 1))
            x_bw = zeros((res_size, 1))
            state_fw = zeros((res_size + inSize, 1))
            state_bw = zeros((res_size + inSize, 1))
        X = zeros((testLen, inSize + total_res_size, 1))
        Y = zeros((testLen, outSize, 1))
        inputs, fw_states, bw_states = [], [], []
        for t in range(testLen):
            u_fw = reshape(asarray(Xts[t]), (inSize, 1))
            inputs.append(u_fw)
            if use_reservoirs == "True":
                u_bw = reshape(asarray(Xts[testLen-1-t]), (inSize, 1))
                x_fw = (1 - a) * x_fw + a * tanh(dot(Win_fw, u_fw) + dot(W_fw, x_fw))
                x_bw = (1 - a) * x_bw + a * tanh(dot(Win_bw, u_bw) + dot(W_bw, x_bw))
                fw_states.append(x_fw)
                bw_states.append(x_bw)
        features = zip(inputs, fw_states, flip(bw_states, 0))
        for j, ctx in enumerate(features):
            u = features[j][0]
            if use_reservoirs == "True":
                x_fw = features[j][1]
                if bidirectional == "True":
                    x_bw = features[j][2]
                    state = vstack((u, x_fw, x_bw))
                else:
                    state = vstack((u, x_fw))
            else:
                state = u
            y = dot( Wout, state )
            X[j] = state
            Y[j] = y
        states.append((X, Y, Yts, lemmas, synsets, pos))

    test_error = zeros((outSize,1))
    totalLen = 0
    correct, all = 0, 0
    for text in states:
        outs = text[1]
        golds = text[2]
        lemmas = text[3]
        synsets = text[4]
        pos = text[5]
        correct_text, all_text = calculate_accuracy(outs, synsets, lemmas, pos, embeddings, dictionary)
        correct += correct_text
        all += all_text
        for i, out in enumerate(outs):
            test_error += square((reshape(golds[i], (outSize, 1)) - out))
        totalLen += len(outs)
    accuracy = (correct * 100.0) / all
    test_error = sqrt(test_error) / totalLen
    print "The mean square error: " + str(sum(test_error)[0] / outSize)
    print "The accuracy score on the test data is: " + str(accuracy)



