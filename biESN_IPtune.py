import scipy.sparse.linalg
import cPickle
import argparse
import os

from numpy import flip, random, linalg, zeros, tanh, dot, vstack, sqrt, eye, asarray, reshape, sum

from preprocess_data import load_embeddings, read_data, format_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0', description='Train or evaluate a neural WSD model.',
                                     fromfile_prefix_chars='@')
    parser.add_argument('-embeddings_model', dest='embeddings_model', required=True, help='Location of the embeddings.')
    parser.add_argument('-train_data', dest='train_data', required=True, help='Path to the training data.')
    parser.add_argument('-sensekey2synset', dest='sensekey2synset', required=True, help='Path to the synset mappings.')
    parser.add_argument('-embeddings_size', dest='embeddings_size', required=True)
    parser.add_argument('-window_size', dest='window_size', required=True)
    parser.add_argument('-use_reservoirs', dest='use_reservoirs', required=False, default="True",
                        help="Use reseroirs or train directly on the word embeddings.")
    parser.add_argument('-bidirectional', dest='bidirectional', required=False, default="True",
                        help="Use a bidirectional architecture, or just one reservoir.")
    parser.add_argument('-res_size', dest='res_size', required=False, default=100,
                        help="Size of the echo state reservoirs.")
    parser.add_argument('-leak_rate', dest='leak_rate', required=False, default=1, help="Leaking rate.")
    parser.add_argument('-res_sparsity', dest='res_sparsity', required=False, default=0.5, help="Reservoir sparsity.")
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.0005, help="IP tuning learning rate.")
    parser.add_argument('-Gaus_mean', dest='Gaus_mean', required=False, default=0.0, help="Gaussian mean value.")
    parser.add_argument('-Gaus_sigma', dest='Gaus_sigma', required=False, default=0.0, help="Gaussian sigma value.")
    parser.add_argument('-IP_iterations', dest='IP_iterations', required=False, default=3, help="IP iterations number.")
    parser.add_argument('-only_open_class', dest='only_open_class', required=False, default="True")
    parser.add_argument('-training_iterations', dest='training_iterations', required=False, default="1")
    parser.add_argument('-save_path', dest='save_path', required=True)

    args = parser.parse_args()
    embeddings_model = args.embeddings_model
    train_data_path = args.train_data
    sensekey2synset = args.sensekey2synset
    embeddings_size = int(args.embeddings_size)
    window_size = int(args.window_size)
    use_reservoirs = args.use_reservoirs
    bidirectional = args.bidirectional
    res_size = int(args.res_size)
    a = float(args.leak_rate)  # leaking rate
    res_sparsity = float(args.res_sparsity)
    learning_rate = float(args.learning_rate)
    Gaus_mean = float(args.Gaus_mean)
    Gaus_sigma = float(args.Gaus_sigma)
    IP_iterations = int(args.IP_iterations)
    only_open_class = args.only_open_class
    training_iterations = int(args.training_iterations)
    save_path = args.save_path

    embeddings = load_embeddings(embeddings_model)  # load the embeddings
    f_sensekey2synset = cPickle.load(open(sensekey2synset, "rb"))  # get the mapping between synset keys and IDs
    train_data = read_data(train_data_path, f_sensekey2synset, only_open_class)  # read the training data

    inSize = embeddings_size * window_size
    outSize = embeddings_size
    # generate the ESN reservoirs
    name_add = str(res_size) + '_' + str(res_sparsity) + '_' + str(a)
    random.seed(42)
    Wout = (random.rand(outSize, total_res_size + inSize) - 0.5) * 1.0
    Win_fw = (random.rand(res_size, inSize) - 0.5) * 1
    Win_bw = (random.rand(res_size, inSize) - 0.5) * 1
    G_fw = ones((res_size, 1))
    G_bw = ones((res_size, 1))
    B_fw = zeros((res_size, 1))
    B_bw = zeros((res_size, 1))
    Wini = scipy.sparse.rand(res_size, res_size, density=res_sparsity)
    i, j, v = scipy.sparse.find(Wini)
    # W = Wini.toarray()
    W_fw = Wini.toarray()
    W_bw = Wini.toarray()
    # W[i, j] -= 0.5
    W_fw[i, j] -= 0.5
    W_bw[i, j] -= 0.5

    # normalizing and setting spectral radius
    print 'Computing spectral radius...'
    # rhoW = max(abs(linalg.eig(W)[0]))
    rhoW_fw = max(abs(linalg.eig(W_fw)[0]))
    rhoW_bw = max(abs(linalg.eig(W_bw)[0]))
    print 'done.'
    # W *= 1.25 / rhoW
    W_fw *= 1.25 / rhoW_fw
    W_bw *= 1.25 / rhoW_bw

    # IP tuning
    print 'IP tunning of reservoirs...'    
    for i,_ in enumerate(train_data):
        print 'calculating reservoir states on text ' + str(i+1) + '/' + str(len(train_data)) + '...'
        Xtr, Ytr,_, _, _ = format_data(train_data[i], embeddings, embeddings_size, window_size)
        trainLen = len(Xtr)
        u_fw = zeros((inSize, 1))
        u_bw = zeros((inSize, 1))
	x_fw = zeros((res_size, 1))
	x_bw = zeros((res_size, 1))
	net_fw = zeros((res_size, 1))
	net_bw = zeros((res_size, 1))
	dG_fw = zeros((res_size, 1))
	dG_bw = zeros((res_size, 1))
	dB_fw = zeros((res_size, 1))
	dB_bw = zeros((res_size, 1))
	
	for iter in range(IP_iterations):
		for t in range(trainLen):
			u_fw = reshape(asarray(Xtr[t]), (inSize, 1))
			u_bw = reshape(asarray(Xtr[trainLen-1-t]), (inSize, 1))
			net_fw = dot(Win_fw, u_fw) + dot(W_fw, x_fw)
			net_bw = dot(Win_bw, u_bw) + dot(W_bw, x_bw)
			x_fw = (1 - a) * x_fw + a * tanh(diag(G_fw)*net_fw + B_fw)
			x_bw = (1 - a) * x_bw + a * tanh(diag(G_bw)*net_bw + B_bw)
			dB_fw=-learning_rate*(-Gaus_mean/(Gaus_sigma**2)+(1/(Gaus_sigma**2))*dot(x_fw, (2.0*(Gaus_sigma**2)+1.0-dot(x_fw, x_fw)+Gaus_mean*x_fw)))
			dB_bw=-learning_rate*(-Gaus_mean/(Gaus_sigma**2)+(1/(Gaus_sigma**2))*dot(x_bw, (2.0*(Gaus_sigma**2)+1.0-dot(x_bw, x_bw)+Gaus_mean*x_bw)))
			dG_fw=learning_rate/G_fw+dot(dB_fw, net_fw)
			dG_bw=learning_rate/G_bw+dot(dB_bw, net_bw)
			G_fw+=dG_fw
			G_bw+=dG_bw
			B_fw+=dB_fw
			B_bw+=dB_bw		             
				
    print "...done."

    # f = open(os.path.join(save_path, 'trained_ESN3_' + name_add + '.cpickle'), "wb")
    # cPickle.dump(intermediate_data, f, protocol=2)
    # f.close()

    print 'Saving...'
    with open(os.path.join(save_path, 'IPtrained_ESN_' + name_add + '_parameters.txt'), 'w') as params:
        params.write(str(args) + '\n\n')
    f = open(os.path.join(save_path, 'IPtrained_ESN_' + name_add + '.cpickle'), "wb")
    cPickle.dump(Win_fw, f, protocol=2)
    cPickle.dump(Win_bw, f, protocol=2)
    cPickle.dump(W_fw, f, protocol=2)
    cPickle.dump(W_bw, f, protocol=2)
    cPickle.dump(G_fw, f, protocol=2)
    cPickle.dump(G_bw, f, protocol=2)
    cPickle.dump(B_fw, f, protocol=2)
    cPickle.dump(B_bw, f, protocol=2)
    f.close()
