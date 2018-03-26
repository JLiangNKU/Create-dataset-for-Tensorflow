# Create-dataset-for-Tensorflow
create yourself dataset for tensorflow


test code





    getTrianList()
    dataroad = "/Users/zhuxiaoxiansheng/Desktop/Yaledata.txt"
    outputdir = "/Users/zhuxiaoxiansheng/Desktop/Yaledata"
    trainroad = trans2tfRecord(dataroad,outputdir)
    traindata,trainlabel = read_tfRecord(trainroad)

    image_batch,label_batch = tf.train.shuffle_batch([traindata,trainlabel],
                                            batch_size=100,capacity=2000,min_after_dequeue = 1000) 
    
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord = coord)
        train_steps = 10  
        # Retrieve a single instance:  
        try:  
            while not coord.should_stop(): 
                example,label = sess.run([image_batch,label_batch])  
                print(example.shape,label)  
  
                train_steps -= 1  
                print(train_steps)  
                if train_steps <= 0:  
                    coord.request_stop()    
  
        except tf.errors.OutOfRangeError:  
            print ('Done training -- epoch limit reached')  
        finally:  
            # When done, ask the threads to stop. 
            coord.request_stop()  
            # And wait for them to actually do it.  
            coord.join(threads)      
            
 output

 ![](https://github.com/LiangjunFeng/Create-dataset-for-Tensorflow/blob/master/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202018-03-26%20%E4%B8%8B%E5%8D%883.18.15.png)
