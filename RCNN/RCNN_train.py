#training the model.
#process--->1.load data. 2.create session. 3.feed data. 4.training 5.validation
import tensorflow as tf
import numpy as np
from RCNN_model import TextRCNN
from data_utils_keyphrase import load_data_new,create_vocabulary,create_vocabulary_label
import os
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import pickle

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",2,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.1,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training/evaluating.") 
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") 
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") 
tf.app.flags.DEFINE_string("ckpt_dir","text_rcnn_title_desc_checkpoint2/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",16,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",100,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") 
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("traning_data_path","keyphrases_train.txt","path of traning data.") 
tf.app.flags.DEFINE_string("word2vec_model_path","keyphrases-word2vec-model","word2vec's vocabulary and vectors") 
#1.load data. 2.create session. 3.feed data. 4.training 5.validation
def main(_):    
    if True:
        trainX, trainY, testX, testY = None, None, None, None
        vocabulary_word2index, vocabulary_index2word = create_vocabulary(word2vec_model_path=FLAGS.word2vec_model_path,name_scope="rcnn") #simple='simple'
        vocab_size = len(vocabulary_word2index)
        print("cnn_model.vocab_size:",vocab_size)
        vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label(name_scope="rcnn")
        train, test, _ = load_data_new(vocabulary_word2index, vocabulary_word2index_label,traning_data_path=FLAGS.traning_data_path) 
        trainX, trainY = train
        testX, testY = test
        # 2.Data preprocessing.Sequence padding.Post padding
        print("start padding & transform to one hot...")
        trainX=np.array([row + [0] * (FLAGS.sequence_length - len(row)) for row in trainX])
        testX=np.array([row + [0] * (FLAGS.sequence_length - len(row)) for row in testX])        
        print("trainX[0]:", trainX[0]) 
        # Converting labels to binary vectors
        print("end padding & transform to one hot...")
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textRCNN=TextRCNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size,FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.sequence_length,
                 vocab_size,FLAGS.embed_size,FLAGS.is_training)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textRCNN,word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch=sess.run(textRCNN.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])
                feed_dict = {textRCNN.input_x: trainX[start:end],textRCNN.dropout_keep_prob: 0.5}
                feed_dict[textRCNN.input_y] = trainY[start:end]                
                curr_loss,curr_acc,_=sess.run([textRCNN.loss_val,textRCNN.accuracy,textRCNN.train_op],feed_dict) 
                loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                if counter %3400==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter))) 

            #epoch increment
            print("going to increment epoch counter....")
            sess.run(textRCNN.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss, eval_acc=do_eval(sess,textRCNN,testX,testY,batch_size,vocabulary_index2word_label)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_acc))
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)
        
        test_loss, test_acc = do_eval(sess, textRCNN, testX, testY, batch_size,vocabulary_index2word_label)
    pass

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textRCNN,word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    
    word2vec_model = Word2Vec.load(word2vec_model_path)
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.wv.vocab, word2vec_model.wv[word2vec_model.wv.vocab]):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textRCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

def do_eval(sess,textCNN,evalX,evalY,batch_size,vocabulary_index2word_label):
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1}
        feed_dict[textCNN.input_y] = evalY[start:end]        
        curr_eval_loss, logits,curr_eval_acc= sess.run([textCNN.loss_val,textCNN.logits,textCNN.accuracy],feed_dict)
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

#get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=1):    
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    return index_list

def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

if __name__ == "__main__":
    tf.app.run()
