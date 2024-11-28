from aew_sm import *
#from aew_gpu_3 import *
from preprocessing_utils import *
from clustering import *
from aew_surface_plotter import *

from sklearn.metrics import accuracy_score


import warnings

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    #ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/home/bryanportillo_lt/Documents/py_env/venv/network_intrusion_dataset/Train_data.csv'
   
    #ids_train_file = 'e:/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    #cm1_file = '/media/mint/NethermostHallV2/py_env/venv/bug_detection_datasets/jm1.csv'

    cm1_file = 'sq_ds/jm1.csv'

    #cm1_file = 'sq_ds/kc2.csv'


    '''
    opt_cycles = [2, 5, 10, 25, 30, 35, 40, 45,50]

    opt_cycles = [5]
    
    for rep in range(5):

        for cycle in opt_cycles:
    
            synthetic_data_tester(rep, cycle)
    '''     
    
        #synthetic_data_tester(rep)

    '''
    data_obj = data(datapath = ids_train_file)

    data_obj.load_data(50)

    data_obj.load_labels()

    data_obj.encode_categorical('protocol_type', 'data')

    data_obj.encode_categorical('service', 'data')

    data_obj.encode_categorical('flag', 'data')

    data_obj.encode_categorical('class', 'labels')

    data_obj.scale_data('min_max')

    data_obj.generate_graphs(10)

    aew_obj = aew(data_obj.graph.toarray(), data_obj.data, data_obj.labels, 'var')

    aew_obj.similarity_matrix = aew_obj.generate_optimal_edge_weights(5)

    #aew_obj.similarity_matrix = aew_obj.generate_edge_weights(aew_obj.gamma)

    #aew_obj.get_eigenvectors(2, .90)

    #pca = PCA(n_components=2)

    #aew_obj.data = pd.DataFrame(pca.fit_transform(data_obj.data))

    #plot_error_surface(aew_obj)
    '''

    data_obj = data(cm1_file, graph_type='stratified')

    #data_obj.load_data()

    #data_obj.load_labels()

    data_obj.encode_categorical('defects', 'labels')

    data_obj.scale_data('min_max')

    #data_obj.generate_graphs(100)

    for rep in range(5):

        diag_base = str(rep) + "," 

        dir_name = 'results_' + str(rep)  

        os.makedirs(str('./'+dir_name+'/plain_data/'), exist_ok=True)

        #print("Strat data: ", data_obj.stratified_data)

        #print("Strat labesl: ", data_obj.stratified_labels)

        #print("Strat data len: ", data_obj.stratified_data.shape)

        data_obj.generate_graphs(100, rep)

        print("Strat  graph: ", data_obj.graph)

        aew_obj = aew(data_obj.graph, data_obj.stratified_data[rep], data_obj.stratified_labels, rep)
 
        aew_obj.generate_optimal_edge_weights(1)

        #error_str = diag_base + str(aew_obj.final_error) + "\n"

        #test_diag_file.write(error_str)

        ###### Test 3d Data

        aew_obj.get_eigenvectors('lowest_var', .90)

        ###### Original Data Test

        visualizer_obj = visualizer(data_obj.labels, 3)

        visualizer_obj.lower_dimensional_embedding(data_obj.data.to_numpy(), "orig_data_3.html", str("./"+dir_name+"/plain_data/"))

        ###### Eigenvector Data Test

        clustering_obj = clustering(base_data = data_obj.data.to_numpy(), data=aew_obj.eigenvectors, labels=aew_obj.labels, path_name = str("./"+dir_name+"/"), name_append="eigenvector_3d_data", workers=-1)

        clustering_obj.generate_spectral()    

        clustering_obj.generate_kmeans()

        clustering_obj.generate_gaussianmixture()
