from spread_opt import *
from preprocessing_utils import *
from clustering import *

from sklearn.metrics import accuracy_score


import warnings

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/home/bryanportillo_lt/Documents/py_env/venv/network_intrusion_dataset/Train_data.csv'
   
    #ids_train_file = 'e:/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'
    
    #synth_clust = clustering()

    #synth_clust.synthetic_data_tester()
    
    data_obj = data()

    data_obj.load_data(ids_train_file, 'train')

    data_obj.load_labels('train', from_data=True)

    data_obj.split_data(.5)

    data_obj.encode_categorical('protocol_type', 'data')

    data_obj.encode_categorical('service', 'data')

    data_obj.encode_categorical('flag', 'data')

    data_obj.encode_categorical('class', 'labels')

    #print(data_obj.train_data.tail(10))
    #print(data_obj.train_labels.tail(5))

    data_obj.scale_data('min_max')

    #print(data_obj.train_data.tail(10))

    init_path = './results/orig_data_visualization/'

    os.makedirs(init_path, exist_ok=True)

    #data_obj.lower_dimensional_embedding(data_obj.train_data, 'train', 'Original Train Data: 3-Dimensions', init_path)

    #data_obj.lower_dimensional_embedding(data_obj.test_data, 'test', 'Original Test Data: 3-Dimensions', init_path)

    data_obj.generate_graphs('train')

    data_obj.generate_graphs('test')

    init_path = './results/plain_data/'

    os.makedirs(init_path, exist_ok=True)
    
    clustering_meths = [           'Kmeans',
                                   'Agglomerative',
                                   'Spectral',
                                   'Birch',
                                   'BisectingKmeans',
                                   'GaussianMixture'
                                   ]

    data_obj.lower_dimensional_embedding(data_obj.train_graph ,'train', 'Original Train Graph: 3-Dimensions', init_path)

    data_obj.lower_dimensional_embedding(data_obj.test_graph, 'test', 'Original Test Graph: 3-Dimensions', init_path)
    
    #plain_clustering = clustering(data_obj.train_data, data_obj.train_labels, data_obj.test_data, data_obj.test_labels, "full", "40_dim_no_proj", clustering_methods=clustering_meths, workers = -1)

    rng = np.random.default_rng()

    #prec_gamma = rng.random(size=(1, 41)) 

    #prec_gamma = np.random.randint(0, 200, (1, 41))

    #prec_gamma = prec_gamma.astype(float)

    prec_gamma = np.var(data_obj.train_data, axis=0).values

    print(prec_gamma)

    

    aew_train = aew(data_obj.train_graph, data_obj.train_data, data_obj.train_labels, prec_gamma)

    aew_train.generate_optimal_edge_weights(20)

    #aew_train.generate_edge_weights()

    prec_gamma = np.var(data_obj.test_data, axis=0).values

    aew_test = aew(data_obj.test_graph, data_obj.test_data, data_obj.test_labels, prec_gamma)

    #print(np.isnan(aew_train.similarity_matrix).any())

    #print(len(aew_train.similarity_matrix[0]))

    #print(len(aew_train.similarity_matrix))

    aew_test.generate_edge_weights()

    #print(np.isnan(aew_test.similarity_matrix).any())

    #print(len(aew_test.similarity_matrix[0]))

    #print(len(aew_test.similarity_matrix))

    
    clustering_with_adj_matr_prec_kmeans = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', n_jobs=-1)

    print("Kmeans Train: ", accuracy_score(clustering_with_adj_matr_prec_kmeans.fit_predict(aew_train.eigenvectors), aew_train.labels))

    clustering_with_adj_matr_prec_disc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='discretize', n_jobs=-1)

    print("Discretize Train: ", accuracy_score(clustering_with_adj_matr_prec_disc.fit_predict(aew_train.eigenvectors), aew_train.labels))

    clustering_with_adj_matr_prec_clust = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='cluster_qr', n_jobs=-1)

    print("Cluster_qr Train: ", accuracy_score(clustering_with_adj_matr_prec_clust.fit_predict(aew_train.eigenvectors), aew_train.labels))

    clustering_with_adj_matr_prec_kmeans1 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', n_jobs=-1)

    print("Kmeans Test: ", accuracy_score(clustering_with_adj_matr_prec_kmeans1.fit_predict(aew_test.eigenvectors), aew_test.labels))

    clustering_with_adj_matr_prec_disc1 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='discretize', n_jobs=-1)

    print("Discretize Test: ", accuracy_score(clustering_with_adj_matr_prec_disc1.fit_predict(aew_test.eigenvectors), aew_test.labels))

    clustering_with_adj_matr_prec_clust1 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='cluster_qr', n_jobs=-1)

    print("Cluster_qr Test: ", accuracy_score(clustering_with_adj_matr_prec_clust1.fit_predict(aew_test.eigenvectors), aew_test.labels))
    
    
    plain_graph_clustering = clustering(aew_train.eigenvectors, aew_train.labels, aew_test.eigenvectors, aew_test.labels, "full", "40_dim_no_proj_graph_data", clustering_methods=clustering_meths,  workers = -1)

    plain_graph_clustering.generate_clustering()

'''
    num_components = [3, 8, 12, 15, 20, 40]

    for num_comp in num_components:

        print("Current number of components: ", num_comp)

        data_obj.train_projection, _ = data_obj.downsize_data(aew_train.eigenvectors, 'train', num_comp)

        data_obj.test_projection, _ = data_obj.downsize_data(aew_test.eigenvectors, 'test', num_comp)
        init_path = './results/orig_data_visualization/num_comp_' + str(num_comp) + '/'

        os.makedirs(init_path, exist_ok=True)

        for projection in data_obj.train_projection.keys():

            #print("Train NaNs: ", np.count_nonzero(np.isnan(data_obj.train_projection[projection])))    

            #print("Test NaNs: ", np.count_nonzero(np.isnan(data_obj.test_projection[projection])))

            data_obj.lower_dimensional_embedding(data_obj.train_projection[projection], 'train', 'Train Mappings Base: 3-Dimensions', init_path)

            data_obj.lower_dimensional_embedding(data_obj.test_projection[projection], 'test', 'Test Mappings Base: 3-Dimensions', init_path)

            clustering_graph_data = clustering(data_obj.train_projection[projection], data_obj.train_labels, data_obj.test_projection[projection], data_obj.test_labels, num_comp, projection, clustering_methods=clustering_meths, workers = -1)

            clustering_graph_data.generate_clustering()
    '''
