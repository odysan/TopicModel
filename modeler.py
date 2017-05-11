from os import walk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import string
from gensim import corpora, models
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from collections import defaultdict
import optparse


def weights_graph(matrix, max_weight=None, ax=None):
    """Draw diagram for visualizing a weight matrix.  Y-axis represents topic ID, X-axis represents document ID.
    Darker squares imply that document represents that topic more than it represents other topics.  The lighter the
    square, the lower its representation of that topic.
    ie. each row is a topic, each column is a document, and darker squares mean that the document represents that
    topic very well.

    Heavily based on the hinton diagram here: http://matplotlib.org/examples/specialty_plots/hinton_demo.html"""
    ax = ax if ax is not None else plt.gca()

    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = (0,0,0, w)
        size = 0.9
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def process_file(filepath):
    """Reads the NSF paper and returns the abstract.  Each abstract is tokenized, stripped of its stop words, and
    stemmed.

    Returns string
    """
    found_abstract = False
    processed_line = []

    with open(filepath) as f:
        for line in f:
            if "Abstract    :" in line:
                found_abstract = True
                continue
            if found_abstract:
                try:
                    line = line.lower().translate(None, string.punctuation)
                    splitted_line = word_tokenize(line)
                    removed_stop_words = [word for word in splitted_line if word not in stopwords.words('english')]
                    stemmed_words = [stemmer.stem(i) for i in removed_stop_words]
                    processed_line.extend(stemmed_words)
                except:
                    return -1
    return processed_line


def process_file_callback(return_val):
    """Handles the return value of each child process.  Ignores all abstracts that failed parsing and do not contain
    more than 10 cleaned tokens.
    """
    if return_val != -1 and len(return_val) > 10:
        all_sentences.append(return_val)


def hash_topics_to_buckets(corp, lda_model, num_of_topics, min_docs):
    """Takes a non-random sample of all of the NSF papers and returns them for the sake of visualization.  Stops when
    there are at least min_docs (int) documents in each topic.

    Returns a dictionary of (k, v) = (topic id, tuple(document id, document's topic distribution))"""
    topics_hash = defaultdict(list)
    vis_topics = []
    arbitrary_id = 0
    for doc_vec in corp:
        max_topic_id = -9999
        max_topic_dist = -9999
        topic_probs = [0] * num_of_topics
        for (topic_id, topic_dist) in lda_model.get_document_topics(doc_vec, minimum_probability=0.005):
            if topic_dist > max_topic_dist:
                max_topic_dist = topic_dist
                max_topic_id = topic_id
            topic_probs[topic_id] = topic_dist
        arbitrary_id += 1
        topics_hash[max_topic_id].append((arbitrary_id, topic_probs))
        vis_topics.append((arbitrary_id, topic_probs))
        finished = True
        for x in range(0, num_of_topics):
            if len(topics_hash[x]) < min_docs:
                finished = False
        if finished:
            # (ids, distributions_matrix) = create_vis_data(vis_topics, num_of_topics)
            # visualize(ids, distributions_matrix)
            break
    return topics_hash


def create_vis_data(collection, num_of_topics):
    """Based on a dictionary or a list, creates a distribution matrix (using numpy) and list of id's associated with
    each record for the sake of visualization.

    Returns tuple(list of id's, dist matrix)"""
    list_of_ids = []
    dist_matrix = np.vstack([[0] * num_of_topics])
    if isinstance(collection, defaultdict):
        for key in sorted(collection):
            for value in collection[key]:
                dist_matrix = np.vstack([dist_matrix, value[1]])
                list_of_ids.append(value[0])
    elif isinstance(collection, list):
        for tup in collection:
            dist_matrix = np.vstack([dist_matrix, tup[1]])
            list_of_ids.append(tup[0])
    dist_matrix = np.delete(dist_matrix, (0), axis=0)
    return list_of_ids, dist_matrix


def visualize(ids, dists):
    """Visualizes distributions matrix."""
    weights_graph(dists)
    (num_cols, num_rows) = dists.shape
    plt.yticks(range(0, num_rows), range(0, num_rows))
    plt.ylabel('Topic ID')
    plt.xticks(range(0, len(ids)), ids)
    plt.xlabel('Document ID')
    plt.title('Document distributions per topic')
    plt.show()


if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest='data', default="./data", type=str,
                         help="The directory containing the NSF papers that we will be clustering. (default: ./data)")
    optparser.add_option("-t", "--numtopics", dest='num_of_topics', default=5, type=int,
                         help="Number of topics we want to group our papers into. (default: 5)")
    optparser.add_option("-i", "--iterlda", dest='iterations_lda', default=20, type=int,
                         help="Number of times the LDA algorithm should iterate for topic modeling. (default: 20")
    optparser.add_option("-w", "--numworkers", dest='num_of_workers', default=mp.cpu_count(), type=int,
                         help="Number of workers for multithreading (default: max available for CPU)")
    optparser.add_option("-k", "--mindocs", dest='min_docs', default=5, type=int,
                         help="Visualize a sample of all documents that reflect at least k of each topic. "
                              "(default: 5)")
    (opts, _) = optparser.parse_args()

    stemmer = PorterStemmer()
    pool = mp.Pool(opts.num_of_workers)
    all_sentences = []
    all_words = set()
    for (dirpath, dirnames, filenames) in walk(opts.data):
        if len(dirnames) == 0:
            for filename in filenames:
                if filename not in ["links.html", "index.html"]:
                    pool.apply_async(process_file, args=(dirpath + "/" + filename, ), callback=process_file_callback)
    pool.close()
    pool.join()
    print "Finished reading file."

    dictionary = corpora.Dictionary(all_sentences)
    print "Finished preparing dictionary."

    dictionary.filter_extremes()
    corpus = [dictionary.doc2bow(sentence) for sentence in all_sentences]
    print "Finished preparing corpus."

    ldamodel = models.LdaMulticore(corpus=corpus, num_topics=opts.num_of_topics, id2word=dictionary,
                                   passes=opts.iterations_lda, workers=opts.num_of_workers)
    print "Finished building LDA model."

    topics_hash = hash_topics_to_buckets(corpus, ldamodel, opts.num_of_topics, opts.min_docs)
    print "Finished sorting by topic."

    (ids, distributions_matrix) = create_vis_data(topics_hash, opts.num_of_topics)
    visualize(ids, distributions_matrix)
