#prediction using model.
#process--->1.load data. 2.create session. 3.feed data. 4.predict
import tensorflow as tf
import numpy as np
from data_utils_keyphrase import load_data_predict,load_final_test_data,create_vocabulary,create_vocabulary_label
import os
import codecs
from RCNN_model import TextRCNN

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",2,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.1,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training/evaluating.") 
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") 
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") 
tf.app.flags.DEFINE_string("ckpt_dir","text_rcnn_title_desc_checkpoint2/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_length",16,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_string("predict_target_file","keyphrases_output.txt","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'keyphrases_source_test.txt',"target file path for final prediction") 
tf.app.flags.DEFINE_string("word2vec_model_path","keyphrases-word2vec-model","word2vec's vocabulary and vectors") 

#1.load data. 2.create session. 3.feed data. 4.predict
# 1.load data with vocabulary of words and labels

def main(_):
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word = create_vocabulary(simple='simple',word2vec_model_path=FLAGS.word2vec_model_path,name_scope="rcnn")
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_vocabulary_label(name_scope="rcnn")
    keyphraseid_keyphrase_lists=load_final_test_data(FLAGS.predict_source_file)
    keyphrase_string_list=[]
    for t in keyphraseid_keyphrase_lists:
        kid,keyphrase=t
        keyphrase_string_list.append(keyphrase)
    test= load_data_predict(vocabulary_word2index,vocabulary_word2index_label,keyphraseid_keyphrase_lists)
    testX=[]
    keyphrase_id_list=[]
    for tuplee in test:
        keyphrase_id,keyphrase_string=tuplee
        keyphrase_id_list.append(keyphrase_id)
        testX.append(keyphrase_string)
    # 2.Data preprocessing: Sequence padding
    print("start padding....")
    testX2=np.array([row + [0] * (FLAGS.sentence_length - len(row)) for row in testX]) # padding to max length    
    print("end padding...")
   # 3.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # 4.Instantiate Model
        textRCNN=TextRCNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.sentence_length,
                 vocab_size,FLAGS.embed_size,FLAGS.is_training)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir)) 
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 5.feed data, to get logits
        number_of_training_data=len(testX2);print("number_of_training_data:",number_of_training_data)
        index=0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'w', 'utf8')
        for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
            logits=sess.run(textRCNN.logits,feed_dict={textRCNN.input_x:testX2[start:end],textRCNN.dropout_keep_prob:1}) 
            print("start:",start,";end:",end)
            keyphrase_id_sublist=keyphrase_id_list[start:end]
            keyphrase_string_sublist=keyphrase_string_list[start:end]
            get_label_using_logits_batch(keyphrase_string_sublist, logits, vocabulary_index2word_label, predict_target_file_f)
            index=index+1
        predict_target_file_f.close()

# get label using logits
def get_label_using_logits_batch(keyphrase_string_sublist,logits_batch,vocabulary_index2word_label,f,top_number=1):
    for i,logits in enumerate(logits_batch):
        index=np.argsort(logits)[top_number] 
        label=vocabulary_index2word_label[index]            
        write_keyphrase_with_labels(keyphrase_string_sublist[i], label, f)
    f.flush()
    
# write keyphrase and labels to file system.
def write_keyphrase_with_labels(keyphrase_string,label,f):
    f.write(keyphrase_string+" __label__"+str(label)+"\n")

if __name__ == "__main__":
    tf.app.run()
