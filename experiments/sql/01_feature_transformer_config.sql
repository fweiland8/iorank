insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (101, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'autoencoder',
        'n_latent_features=4096,reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (102, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'alexnet', 'reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (103, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet', 'reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (104, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'raw', 'reduced_size=64', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (105, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'spatial-mask', 'mask_size=64', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (106, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'handcrafted', null, null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (107, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'autoencoder,label',
        'n_latent_features=4096,reduced_size=128|', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (108, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'alexnet,label', 'reduced_size=128|', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);


insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (109, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet,label', 'reduced_size=128|', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (110, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'raw,label', 'reduced_size=64|', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (111, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'spatial-mask,label', 'mask_size=64|', null,
        null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (112, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'handcrafted,label', null, null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);


insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (113, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet,spatial-mask',
        'reduced_size=128|mask_size=64', null,
        null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);


insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (114, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet,spatial-mask,handcrafted',
        'reduced_size=128|mask_size=64|', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);


insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (115, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet,spatial-mask,handcrafted,label',
        'reduced_size=128|mask_size=64||', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (116, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'autoencoder',
        'input_type=blacked,n_latent_features=4096,reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (117, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'alexnet',
        'input_type=blacked,reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (118, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet',
        'input_type=blacked,reduced_size=128', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (119, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'raw',
        'input_type=blacked,reduced_size=64', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (120, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet,handcrafted',
        'reduced_size=128|', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (130, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'spatial-mask,handcrafted',
        null, null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (131, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet,handcrafted',
        'reduced_size=256|', null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);
