#prediction using model.
#process--->1.load data. 2.create session. 3.feed data. 4.predict

import tensorflow as tf
import numpy as np
from data_utils_keyphrase import load_data_predict,load_final_test_data,create_vocabulary,create_vocabulary_label
import os
import codecs
from CNN_model import TextCNN

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",2,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.1,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.") 
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") 
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") 
tf.app.flags.DEFINE_string("ckpt_dir","text_cnn_title_desc_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",16,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",15,"number of epochs.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") 
tf.app.flags.DEFINE_string("predict_target_file","keyphrases_output.txt","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'keyphrases_source_test.txt',"target file path for final prediction") 
tf.app.flags.DEFINE_string("word2vec_model_path","keyphrases-word2vec-model","word2vec's vocabulary and vectors") 
tf.app.flags.DEFINE_integer("num_filters", 300, "number of filters") 
tf.app.flags.DEFINE_string("ckpt_dir2","text_cnn_title_desc_checkpoint_exp/","checkpoint location for the model")

##############################################################################################################################################
filter_sizes=[1,2,3,4,5,6,7]#[1,2,3,4,5,6,7]
#1.load data. 2.create session. 3.feed data. 4.predict
def main(_):
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word = create_vocabulary(simple='simple',word2vec_model_path=FLAGS.word2vec_model_path,name_scope="cnn2")
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_vocabulary_label(name_scope="cnn2")
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
    testX2=np.array([row + [0] * (FLAGS.sentence_len - len(row)) for row in testX])
    #testX2 = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("end padding...")
   # 3.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # 4.Instantiate Model
        textCNN=TextCNN(filter_sizes,FLAGS.num_filters,FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,FLAGS.decay_rate,
                        FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training)
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
            logits=sess.run(textCNN.logits,feed_dict={textCNN.input_x:testX2[start:end],textCNN.dropout_keep_prob:1}) 
            # 6. get lable using logtis
            predicted_label=get_label_using_logits(logits[0],vocabulary_index2word_label)
            # 7. write question id and labels to file system.
            write_keyphrase_with_labels(keyphrase_string_list[index],predicted_label,predict_target_file_f)
            index=index+1
        predict_target_file_f.close()

# get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=1):
    #print(logits)
    index=np.argsort(logits)[-top_number] 
    #print(index)
    label=vocabulary_index2word_label[index]
    return label

# write keyphrase and labels to file system.
def write_keyphrase_with_labels(keyphrase_string,label,f):
    f.write(keyphrase_string+" __label__"+str(label)+"\n")

if __name__ == "__main__":
    tf.app.run()
