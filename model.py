import tensorflow as tf

class MemNN:
	def __init__(self, params):
		self.query_raw = tf.placeholder(tf.string)
		self.answers_raw = tf.placeholder(tf.string)
		self.sentences_raw = tf.placeholder(tf.string)
		self.targets = tf.placeholder(tf.string)
		self.vocab2index = tf.contrib.lookup.index_table_from_tensor(params['vocab'], default_value=0)
		self.embeddings = tf.Variable(tf.random_uniform([params['vocab_size'], params['embedding_size']], -1., 1.))
		self.params = params

	def process_strings(self):
		query = self.vocab2index.lookup(self.query_raw)
		answers = self.vocab2index.lookup(self.answers_raw)
		sentences = self.vocab2index.lookup(self.sentences_raw)
		self.query = query
		self.answers = answers
		self.sentences = sentences


	def forward(self):
		query = tf.nn.embedding_lookup(self.embeddings, self.query)
		sentences = tf.nn.embedding_lookup(self.embeddings, self.sentences)
		query_averaged = tf.reduce_mean(query, axis=1)
		sentences_averaged = tf.reduce_mean(sentences,axis=1)
		p = tf.nn.softmax(tf.matmul(query_averaged, sentences_averaged))
		o = tf.reduce_sum(tf.multiply(p, sentences_averaged))
		W = tf.Variable(tf.random_uniform([self.params['embedding_size'], len(self.params['vocab'])], -1., 1.))
		a_cap = tf.nn.softmax(tf.matmul(tf.add(query_averaged, o), W))
		self.pred = a_cap

	def backward(self):
		answer = tf.one_hot(self.answers)
		loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=answer, logits=self.pred)
		optimizer = tf.train.AdamOptimizer(0.001)
		train_op = optimizer.minimize(loss)
		self.loss = loss
		self.train_op = train_op