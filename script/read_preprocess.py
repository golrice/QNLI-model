import os 
import numpy as np 

# constant
word_embedding_path_50d = os.path.join(os.path.dirname(__file__), "..", "resources", "glove", "glove.6B.50d.txt")
word_embedding_path_100d = os.path.join(os.path.dirname(__file__), "..", "resources", "glove", "glove.6B.100d.txt")
word_embedding_path_200d = os.path.join(os.path.dirname(__file__), "..", "resources", "glove", "glove.6B.200d.txt")
word_embedding_path_300d = os.path.join(os.path.dirname(__file__), "..", "resources", "glove", "glove.6B.300d.txt")

save_words_path_50d = os.path.join(os.path.dirname(__file__), "..", "resources", "words", "save_words_50d")
save_words_path_100d = os.path.join(os.path.dirname(__file__), "..", "resources", "words", "save_words_100d")
save_words_path_200d = os.path.join(os.path.dirname(__file__), "..", "resources", "words", "save_words_200d")
save_words_path_300d = os.path.join(os.path.dirname(__file__), "..", "resources", "words", "save_words_300d")

save_vector_path_50d = os.path.join(os.path.dirname(__file__), "..", "resources", "vector", "save_vector_50d")
save_vector_path_100d = os.path.join(os.path.dirname(__file__), "..", "resources", "vector", "save_vector_100d")
save_vector_path_200d = os.path.join(os.path.dirname(__file__), "..", "resources", "vector", "save_vector_200d")
save_vector_path_300d = os.path.join(os.path.dirname(__file__), "..", "resources", "vector", "save_vector_300d")

embedding_path = (word_embedding_path_50d, word_embedding_path_100d, word_embedding_path_200d, word_embedding_path_300d)
save_words_path = (save_words_path_50d, save_words_path_100d, save_words_path_200d, save_words_path_300d)
save_vector_path = (save_vector_path_50d, save_vector_path_100d, save_vector_path_200d, save_vector_path_300d)
dim = (50, 100, 200, 300)

# global variables
word_embedding_dict = {}

def read_word_embedding(embedding_path, save_words_path, save_vector_path):
    with open(embedding_path, 'r',encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], np.float32)
            word_embedding_dict[word] = vector

        np.save(save_words_path, np.array(list(word_embedding_dict.keys())))
        np.save(save_vector_path, np.array(list(word_embedding_dict.values())))


#####################################################################################################################

for i in range(len(embedding_path)):
    read_word_embedding(embedding_path[i], save_words_path[i], save_vector_path[i])
