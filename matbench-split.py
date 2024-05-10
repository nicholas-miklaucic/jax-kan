import random
import numpy as np
# import torch
# torch.manual_seed(1234)
# torch.cuda.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(1234)
random.seed(1234)



from matbench.bench import MatbenchBenchmark
mb = MatbenchBenchmark(autoload=False, subset=["matbench_steels"])
for task in mb.tasks:
    task.load()
    for fold in task.folds:
        print(f'fold no: {fold}')
        train_inputs, train_outputs = task.get_train_and_val_data(fold)        
        test_inputs = task.get_test_data(fold, include_target=False)        
        print(train_inputs.shape)
        print(train_outputs.shape)
        print(test_inputs.shape)
        # predictions = 0 #all test sample predictions here
        # params = my_model.get_parameters_as_dictionary()
        # task.record(fold, predictions, params=params)
# mb.to_file("results.json.gz")
