import tensorflow as tf

cluster_spec = {
    "ps": ["ec2-52-77-227-193.ap-southeast-1.compute.amazonaws.com:2221", "ec2-54-169-126-11.ap-southeast-1.compute.amazonaws.com:2222"],
    "worker": ["ec2-52-77-227-193.ap-southeast-1.compute.amazonaws.com:2223", "ec2-54-169-126-11.ap-southeast-1.compute.amazonaws.com:2224"]}
server = tf.train.Server(cluster_spec, job_name="worker", task_index=1)

server.join()