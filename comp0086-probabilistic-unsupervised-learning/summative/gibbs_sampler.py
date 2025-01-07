# -*- coding: utf-8 -*-

"""
    File name: gibbs_sampler.py
    Description: a re-implementation of the Gibbs sampler for http://www.gatsby.ucl.ac.uk/teaching/courses/ml1
    Author: python: Roman Pogodin, MATLAB (original): Yee Whye Teh and Maneesh Sahani
    Date created: October 2018
    Python version: 3.6
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)


# todo: sample everything from self.rang_gen to control the random seed (works as numpy.random)
class GibbsSampler:
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          number of documents
        :param n_topics:        number of topics
        :param n_words:         number of words in vocabulary
        :param alpha:           dirichlet parameter on topic mixing proportions
        :param beta:            dirichlet parameter on topic word distributions
        :param random_seed:     random seed of the sampler
        """
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.n_words = n_words
        self.alpha = alpha
        self.beta = beta
        self.rand_gen = np.random.RandomState(random_seed)

        self.docs_words = np.zeros((self.n_docs, self.n_words))
        self.docs_words_test = None
        self.loglike = None
        self.loglike_test = None
        self.do_test = False

        self.A_dk = np.zeros((self.n_docs, self.n_topics))  # number of words in document d assigned to topic k
        self.B_kw = np.zeros((self.n_topics, self.n_words))  # number of occurrences of word w assigned to topic k
        self.A_dk_test = np.zeros((self.n_docs, self.n_topics))
        self.B_kw_test = np.zeros((self.n_topics, self.n_words))

        self.theta = np.ones((self.n_docs, self.n_topics)
                             ) / self.n_topics  # theta[d] is the distribution over topics in document d
        self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words  # phi[k] is the distribution words in topic k

        self.topics_space = np.arange(self.n_topics)
        self.topic_doc_words_distr = np.zeros((self.n_topics, self.n_docs, self.n_words))  # z_id|x_id, theta, phi

    def init_sampling(self, docs_words, docs_words_test=None,
                      theta=None, phi=None, n_iter=0, save_loglike=False):
        assert np.all(docs_words.shape == (self.n_docs, self.n_words)), "docs_words shape=%s must be (%d, %d)" % (
            docs_words.shape, self.n_docs, self.n_words)
        self.n_docs = docs_words.shape[0]

        self.docs_words = docs_words
        self.docs_words_test = docs_words_test

        self.do_test = (docs_words_test is not None)

        if save_loglike:
            self.loglike = np.zeros(n_iter)

            if self.do_test:
                self.loglike_test = np.zeros(n_iter)

        self.A_dk.fill(0.0)
        self.B_kw.fill(0.0)
        self.A_dk_test.fill(0.0)
        self.B_kw_test.fill(0.0)

        self.init_params(theta, phi)

    def init_params(self, theta=None, phi=None):
        if theta is None:
            self.theta = np.ones((self.n_docs, self.n_topics)) / self.n_topics
        else:
            self.theta = theta.copy()

        if phi is None:
            self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words
        else:
            self.phi = phi.copy()

        self.update_topic_doc_words()
        self.sample_counts()

    def run(self, docs_words, docs_words_test=None,
            n_iter=100, theta=None, phi=None, save_loglike=False):
        """
        docs_words is a matrix n_docs * n_words; each entry
        is a number of occurrences of a word in a document
        docs_words_test does not influence the updates and is used
        for validation
        """
        self.init_sampling(docs_words, docs_words_test,
                           theta, phi, n_iter, save_loglike)

        for iteration in range(n_iter):
            self.update_params()

            if save_loglike:
                self.update_loglike(iteration)

        return self.to_return_from_run()

    def to_return_from_run(self):
        return self.topic_doc_words_distr, self.theta, self.phi

    def update_params(self):
        """
        Samples theta and phi, then computes the distribution of
        z_id and samples counts A_dk, B_kw from it
        """
        # todo: sample theta and phi
        
        # Sample theta (document-topic distributions)
        for d in range(self.n_docs):
            # sample from Dirichlet distribution
            self.theta[d, :] = np.random.dirichlet(self.alpha + self.doc_topic_counts[d, :])
        # Sample phi (topic-word distributions)
        for k in range(self.n_topics):
            # sample from Dirichlet distribution
            self.phi[k, :] = np.random.dirichlet(self.beta + self.topic_word_counts[k, :])
        # Update topic-doc-word distribution
        self.update_topic_doc_words()
        # Sample counts (the z_id assignments)
        self.sample_counts()

    def update_topic_doc_words(self):
        """
        Computes the distribution of z_id|x_id, theta, phi
        """
        self.topic_doc_words_distr = np.repeat(
            self.theta.T[:, :, None], self.n_words, axis=2) * self.phi[:, None, :]
        self.topic_doc_words_distr /= self.theta.dot(self.phi)[None, :, :]

    def sample_counts(self):
        """
        For each document and each word, samples from z_id|x_id, theta, phi
        and adds the results to the counts A_dk and B_kw
        """
        self.A_dk.fill(0)
        self.B_kw.fill(0)

        if self.do_test:
            self.A_dk_test.fill(0)
            self.B_kw_test.fill(0)

        # todo: sample a topic for each (doc, word) and update A_dk, B_kw correspondingly
        # Iterate over each document
        for d in range(self.n_docs):
            # Iterate over each word in the document
            for n in range(self.doc_lengths[d]):
                # Current word index
                word_id = self.documents[d][n]
                
                # Calculate the probabilities of each topic for the current word
                topic_probs = np.zeros(self.n_topics)
                for k in range(self.n_topics):
                    topic_probs[k] = self.theta[d, k] * self.phi[k, word_id]
                
                # Normalize the probabilities
                topic_probs /= np.sum(topic_probs)
                
                # Sample a new topic for the current word
                new_topic = np.random.choice(self.n_topics, p=topic_probs)
                
                # Update the counts
                self.z[d,n] = new_topic
                self.A_dk[d, new_topic] += 1
                self.B_kw[new_topic, word_id] += 1
                
                # Update the counts for the test set
                if self.do_test:
                    self.A_dk_test[d, new_topic] += 1
                    self.B_kw_test[new_topic, word_id] += 1
            

    def update_loglike(self, iteration):
        """
        Updates loglike of the data, omitting the constant additive term
        with Gamma functions of hyperparameters.
        """
        loglike = 0.0

        # Compute log-likelihood for document-topic distributions
        for d in range(self.n_docs):
            # Compute the Dirichlet log-likelihood for the document
            loglike += gammaln(self.alpha + np.sum(self.A_dk[d]))  # log(Gamma(alpha + sum(A_dk[d])))

            for k in range(self.n_topics):
                loglike -= gammaln(self.alpha)  # -log(Gamma(alpha))
                loglike += gammaln(self.alpha + self.A_dk[d, k])  # +log(Gamma(alpha + A_dk[d, k]))

        # Compute log-likelihood for topic-word distributions
        for k in range(self.n_topics):
            # Compute the Dirichlet log-likelihood for the topic
            loglike += gammaln(self.beta + np.sum(self.B_kw[k]))  # log(Gamma(beta + sum(B_kw[k])))

            for w in range(self.n_words):
                loglike -= gammaln(self.beta)  # -log(Gamma(beta))
                loglike += gammaln(self.beta + self.B_kw[k, w])  # +log(Gamma(beta + B_kw[k, w]))

        # Store the log-likelihood for this iteration if needed
        self.loglike_history[iteration] = loglike

        def get_loglike(self):
            """Returns log-likelihood at each iteration."""
            if self.do_test:
                return self.loglike, self.loglike_test
            else:
                return self.loglike


class GibbsSamplerCollapsed(GibbsSampler):
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          number of documents
        :param n_topics:        number of topics
        :param n_words:         number of words in vocabulary
        :param alpha:           dirichlet parameter on topic mixing proportions
        :param beta:            dirichlet parameter on topic word distributions
        :param random_seed:     random seed of the sampler
        """
        super().__init__(n_docs, n_topics, n_words, alpha, beta, random_seed)

        # topics assigned to each (doc, word)
        self.doc_word_samples = np.ndarray((self.n_docs, self.n_words), dtype=object)
        self.doc_word_samples_test = self.doc_word_samples.copy()

    def init_params(self, theta=None, phi=None):
        # z_id are initialized uniformly
        for doc in range(self.n_docs):
            for word in range(self.n_words):
                if self.do_test:
                    additional_samples = self.docs_words_test[doc, word]
                else:
                    additional_samples = 0

                sampled_topics = self.rand_gen.choice(self.topics_space,
                                                      size=self.docs_words[doc, word] + additional_samples)

                sampled_topics_train = sampled_topics[:self.docs_words[doc, word]]
                self.doc_word_samples[doc, word] = sampled_topics_train.copy()  # now each cell is an np.array

                sample, counts = np.unique(sampled_topics_train, return_counts=True)

                self.A_dk[doc, sample] += counts
                self.B_kw[sample, word] += counts

                if self.do_test:
                    sampled_topics_test = sampled_topics[self.docs_words[doc, word]:]
                    self.doc_word_samples_test[doc, word] = sampled_topics_test.copy()

                    sample, counts = np.unique(sampled_topics_test, return_counts=True)

                    self.A_dk_test[doc, sample] += counts
                    self.B_kw_test[sample, word] += counts

    def update_params(self):
        """
        Computes the distribution of z_id.
        Sampling of A_dk, B_kw is done automatically as
        each new z_id updates these counters
        """
        # todo: sample a topic for each (doc, word) and update A_dk, B_kw correspondingly
        # Hint: you can update A_dk, B_kw after each sampling instead of re-computing the whole matrix
        pass

    def update_loglike(self, iteration):
        """
        Updates loglike of the data, omitting the constant additive term
        with Gamma functions of hyperparameters
        """
        # todo: implement log-like
        pass

    def to_return_from_run(self):
        return self.doc_word_samples


def read_data(filename):
    """
    Reads the text data and splits into train/test.
    Examples:
    docs_words_train, docs_words_test = read_data('./code/toyexample.data')
    nips_train, nips_test = read_data('./code/nips.data')
    :param filename:    path to the file
    :return:
    docs_words_train:   training data, [n_docs, n_words] numpy array
    docs_words_test:    test data, [n_docs, n_words] numpy array
    """
    data = pd.read_csv(filename, dtype=int, sep=' ', names=['doc', 'word', 'train', 'test'])

    n_docs = np.amax(data.loc[:, 'doc'])
    n_words = np.amax(data.loc[:, 'word'])

    docs_words_train = np.zeros((n_docs, n_words), dtype=int)
    docs_words_test = np.zeros((n_docs, n_words), dtype=int)

    docs_words_train[data.loc[:, 'doc'] - 1, data.loc[:, 'word'] - 1] = data.loc[:, 'train']
    docs_words_test[data.loc[:, 'doc'] - 1, data.loc[:, 'word'] - 1] = data.loc[:, 'test']

    return docs_words_train, docs_words_test


def main():
    print('Running toyexample.data with the standard sampler')

    docs_words_train, docs_words_test = read_data('./toyexample.data')
    n_docs, n_words = docs_words_train.shape
    n_topics = 3
    alpha = 1
    beta = 1
    random_seed = 0

    sampler = GibbsSampler(n_docs=n_docs, n_topics=n_topics, n_words=n_words,
                           alpha=alpha, beta=beta, random_seed=random_seed)

    topic_doc_words_distr, theta, phi = sampler.run(docs_words_train, docs_words_test,
                                                    n_iter=200, save_loglike=True)

    print(phi * [phi > 1e-2])

    like_train, like_test = sampler.get_loglike()

    plt.subplots(figsize=(15, 6))
    plt.plot(like_train, label='train')
    plt.plot(like_test, label='test')

    plt.ylabel('loglike')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

    print('Running toyexample.data with the collapsed sampler')

    sampler_collapsed = GibbsSamplerCollapsed(n_docs=n_docs, n_topics=n_topics, n_words=n_words,
                                              alpha=alpha, beta=beta, random_seed=random_seed)

    doc_word_samples = sampler_collapsed.run(docs_words_train, docs_words_test,
                                             n_iter=200, save_loglike=True)
    topic_counts = np.zeros((3, 6))
    for doc in range(doc_word_samples.shape[0]):
        for word in range(doc_word_samples.shape[1]):
            for topic in doc_word_samples[doc, word]:
                topic_counts[topic, word] += 1

    print(topic_counts)

    like_train, like_test = sampler_collapsed.get_loglike()

    plt.subplots(figsize=(15, 6))
    plt.plot(like_train, label='train')
    plt.plot(like_test, label='test')

    plt.ylabel('loglike')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
