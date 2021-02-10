
## Data Pre-Processing

* `courier_behavior_evaluation.py`: evaluate the confirmation behaviors of different couriers based on waybills.

* `stay_point_extraction.py`: extract stay points from couriers' trajectories.

* `ground_truth_construction.py`: construct the ground-truth *address-delivery location* labels.

* `location_candidate_generation.py`: generate stay points based on stay points of couriers.

* `inverted_indexing.py`: generate the inverted index, and simulate the delayed confirmation behaviors.

## Delivery Location Discovery

* `feature_extraction.py`: extract features and generate learning samples (for classification and selection).

* `dataset.py`: dataset for training.

* `models.py`: the implementation of different models.

* `train_selection_models.py`: train LocMatcher and Pointer Network model.

* `train_classification_mlp.py` and `train_classification_others.py`: train classification-based models.

* `discovery_algo.py`: the implementation of different inference methods. 

* `evaluate.py`: evaluate the performance of different methods.
