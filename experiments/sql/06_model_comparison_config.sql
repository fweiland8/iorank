insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (601, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'bounding-box', null, null, null,
        'baseline-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (602, 'KITTI', 'dorn-ranker', 'model_file=dorn-model.pt', null, null, null, null, null, null, null, null,
        null, null,
        null, null, 0, 5, false);


insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (604, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet,spatial-mask,handcrafted', null,
        'mask_size_range:
- 32
- 96
reduced_size_range:
- 32
- 256',
        null,
        'torch-fate-ranker', null,
        'n_hidden_joint_layers_range:
- 2
- 32
n_hidden_joint_units_range:
- 8
- 64
n_hidden_set_layers_range:
- 2
- 8
n_hidden_set_units_range:
- 2
- 32',
        null, 50, 5, false);
