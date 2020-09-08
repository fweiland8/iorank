insert into thesis_felix.e2e_configuration (id, dataset, n_cells, predictions_per_cell,
                                            input_size, backbone_name, max_n_epochs,
                                            lr_decay_epoch, od_loss_factor, or_loss_factor,
                                            batch_size, n_experiments, include_spatial_mask,
                                            feature_reduction)
values (501, 'KITTI', 7, 2, 448, 'mobilenet', 40, 15, 5, 1, 8, 5, false, 'conv');

insert into thesis_felix.e2e_configuration (id, dataset, n_cells, predictions_per_cell,
                                            input_size, backbone_name, max_n_epochs,
                                            lr_decay_epoch, od_loss_factor, or_loss_factor,
                                            batch_size, n_experiments, include_spatial_mask,
                                            feature_reduction)
values (502, 'KITTI', 11, 2, 448, 'mobilenet', 40, 15, 5, 1, 8, 5, false, 'conv');

insert into thesis_felix.e2e_configuration (id, dataset, n_cells, predictions_per_cell,
                                            input_size, backbone_name, max_n_epochs,
                                            lr_decay_epoch, od_loss_factor, or_loss_factor,
                                            batch_size, n_experiments, include_spatial_mask,
                                            feature_reduction)
values (503, 'KITTI', 7, 2, 448, 'resnet', 40, 15, 5, 1, 8, 5, false, 'conv');

insert into thesis_felix.e2e_configuration (id, dataset, n_cells, predictions_per_cell,
                                            input_size, backbone_name, max_n_epochs,
                                            lr_decay_epoch, od_loss_factor, or_loss_factor,
                                            batch_size, n_experiments, include_spatial_mask,
                                            feature_reduction)
values (504, 'KITTI', 11, 2, 448, 'resnet', 40, 15, 5, 1, 8, 5, false, 'conv');

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (505, 'KITTI', 'iorank', null, 'yolo', 'backbone_name=mobilenet,n_cells=7,predictions_per_cell=2', null, null,
        'resnet', 'reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, true);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (506, 'KITTI', 'iorank', null, 'yolo', 'backbone_name=resnet,n_cells=7,predictions_per_cell=2', null, null,
        'resnet', 'reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, true);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (507, 'KITTI', 'iorank', null, 'yolo', 'backbone_name=mobilenet,n_cells=11,predictions_per_cell=2', null, null,
        'resnet', 'reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, true);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (508, 'KITTI', 'iorank', null, 'yolo', 'backbone_name=resnet,n_cells=11,predictions_per_cell=2', null, null,
        'resnet', 'reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, true);


