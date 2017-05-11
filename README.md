# TopicModel

### How to run

Suppose you:
* Downloaded the National Science Foundation papers to the ~/Desktop/Papers directory
* Want to cluster them into 10 topics
* Want LDA to perform 3 iterations on the corpus
* Only want to use 2 workers when multithreading
* Want there to be a minimum of 5 docs per topic when you visualize it in a graph

If this is the case, then you will execute the program from the command-line like so:

`python modeler.py --data ~/Desktop/Papers --numtopics 10 --iterlda 3 --numworkers 2 --mindocs 5`
