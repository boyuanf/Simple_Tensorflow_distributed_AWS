import tensorflow as tf

cluster_spec = {
    "ps": ["localhost:2221", "localhost:2222"],
    "worker": ["localhost:2223", "localhost:2224"]}
server = tf.train.Server(cluster_spec, job_name="ps", task_index=1)

server.join()