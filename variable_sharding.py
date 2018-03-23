import tensorflow as tf

cluster_spec = {
    "ps": ["ec2-52-77-227-193.ap-southeast-1.compute.amazonaws.com:2221", "ec2-54-169-126-11.ap-southeast-1.compute.amazonaws.com:2222"],
    "worker": ["ec2-52-77-227-193.ap-southeast-1.compute.amazonaws.com:2223", "ec2-54-169-126-11.ap-southeast-1.compute.amazonaws.com:2224"]}

with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):  # the same as ps_tasks=1
#with tf.device(tf.train.replica_device_setter(ps_tasks=1)):
    v1 = tf.Variable(1.0)  # use ps 0
    v2 = tf.Variable(2.0)  # use ps 1
    t = tf.get_variable("t", shape=[20, 20])  # use ps 0
    q = tf.get_variable("q", shape=[20, 20])  # use ps 1
    s = v1 + v2  # default use task 0
    f = v1 * v2  # default use task 0
    with tf.device("/gpu:0"):
        p1 = 2 * s
        with tf.device("/task:1"):
            p2 = 3 * s
    
init = tf.global_variables_initializer()
    
with tf.Session("grpc://ec2-52-77-227-193.ap-southeast-1.compute.amazonaws.com:2221", config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    print("s: ", s.eval())
    print("p1: ", p1.eval())
    print("p2: ", p2.eval())
    print("v1 device: ", v1.device)
    print("v2 device: ", v2.device)
    print("t device: ", t.device)
    print("q device: ", q.device)
    print("s device: ", s.device)
    print("f device: ", f.device)
    print("p1 device: ", p1.device)
    print("p2 device: ", p2.device)
    
    